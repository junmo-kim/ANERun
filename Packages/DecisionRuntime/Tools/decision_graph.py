"""Qwen3.5 text stages for DecisionRuntime's fixed-width scan/cache ABI.

The runtime supplies embeddings in units of 1/64 and constructs rotary tables
and visibility masks. Only the final stage returns normalized, unscaled rows.
Checkpoint tensors are loaded on demand so export needs only one stage in RAM.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from text_norm import BoundedRMSNorm

CHUNK = 128
RESIDUAL_SCALE = 64
TILE = 16
RECURRENCE_SCALE = 64


def precise_silu(x):
    # Evaluate sigmoid using nonpositive exponents so small negative gates survive in FP16.
    return (x * x.clamp_max(0).exp()) / (1 + (-x.abs()).exp())


class CheckpointReader:
    def __init__(self, path):
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.files = json.loads(index.read_text())["weight_map"]
            if any(Path(name).name != name for name in self.files.values()):
                raise ValueError("Expected local checkpoint shard names")
        else:
            name = "model.safetensors"
            with safe_open(self.path / name, framework="pt") as tensors:
                self.files = dict.fromkeys(tensors.keys(), name)

    def get_tensor(self, name):
        with safe_open(self.path / self.files[name], framework="pt") as tensors:
            return tensors.get_tensor(name)


@dataclass(frozen=True)
class Config:
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    taps: int
    eps: float
    layer_types: tuple


@dataclass
class Backbone:
    cfg: Config
    reader: CheckpointReader
    prefix: str
    dtype: torch.dtype
    scaled_recurrence: bool
    valen_recurrence: bool = False

    def weight(self, name, shape, *, offset=0):
        tensor = self.reader.get_tensor(self.prefix + name)
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name}: expected {shape}, found {tuple(tensor.shape)}")
        if offset:
            tensor = tensor.float() + offset
        return tensor.to(self.dtype)


def load_weights(path, *, dtype=torch.float16, scaled_recurrence=False, valen_recurrence=False):
    path = Path(path)
    root = json.loads((path / "config.json").read_text())
    t = root.get("text_config", root)
    rope = t["rope_parameters"]
    if (t.get("hidden_act", "silu") != "silu" or t.get("attention_bias", False)
            or not t.get("attn_output_gate", True) or rope.get("rope_type", "default") != "default"
            or rope["rope_theta"] != 10_000_000):
        raise ValueError("Unsupported activation, projection or rotary configuration")
    reader = CheckpointReader(path)
    prefix, = (p for p in ("model.language_model.", "backbone.language_model.", "model.")
               if p + "norm.weight" in reader.files)
    decision = path / "decision_config.json"
    if decision.is_file():
        loop = json.loads(decision.read_text()).get("midloop")
        if loop:
            if loop["blocks"] != [[12, 17], [18, 23]] or loop["passes"] != 2:
                raise ValueError("Unsupported GLiDE execution schedule")
    cfg = Config(t["hidden_size"], t["intermediate_size"], t["num_attention_heads"],
                 t["num_key_value_heads"], t["head_dim"],
                 int(t["head_dim"] * rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 1))),
                 t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"],
                 t["linear_value_head_dim"], t["linear_conv_kernel_dim"], t["rms_norm_eps"],
                 tuple(t["layer_types"]))
    if (cfg.heads % cfg.kv_heads or cfg.value_heads % cfg.key_heads or cfg.rotary_dim % 2
            or not 0 < cfg.rotary_dim <= cfg.head_dim or not 1 < cfg.taps <= CHUNK
            or set(cfg.layer_types) - {"linear_attention", "full_attention"}
            or len(cfg.layer_types) != t["num_hidden_layers"] or cfg.layer_types[-1] != "full_attention"):
        raise ValueError("Unsupported Qwen3.5 text dimensions or layer order")
    return Backbone(cfg, reader, prefix, dtype, scaled_recurrence, valen_recurrence)


def stage_ranges(cfg, *, pair_before_attention=False):
    """Limit recurrent scans to one layer per physical Core AI source asset."""
    ranges = []
    start = 0
    for i, kind in enumerate(cfg.layer_types):
        if (pair_before_attention and kind == "linear_attention"
                and cfg.layer_types[i + 1:i + 2] == ("full_attention",)):
            continue
        ranges.append((start, i))
        start = i + 1
    return ranges


def execution_schedule(ranges, loop):
    """Group complete stages between the checkpoint's inclusive midloop boundaries."""
    groups = []
    repeats = {start: end for start, end in loop["blocks"]} if loop else {}
    i = 0
    while i < len(ranges):
        start = ranges[i][0]
        end = repeats.pop(start, None)
        first = i
        if end is not None:
            while i < len(ranges) and ranges[i][1] < end:
                i += 1
            if i == len(ranges) or ranges[i][1] != end:
                raise ValueError("Recurrent block ends inside an exported stage")
        groups.append({"stages": list(range(first, i + 1)), "passes": loop["passes"] if end is not None else 1})
        i += 1
    if repeats:
        raise ValueError("Recurrent block starts inside an exported stage")
    return groups


def projection(bb, name, n_in, n_out):
    return pointwise(bb.weight(name + ".weight", (n_out, n_in)))


def pointwise(w):
    n_out, n_in = w.shape
    layer = nn.Conv2d(n_in, n_out, 1, bias=False, device="meta")
    layer.weight = nn.Parameter(w[:, :, None, None], requires_grad=False)
    return layer


def norm(bb, name, dim, axis, *, offset=1, input_scale=1.0):
    layer = BoundedRMSNorm(dim, bb.cfg.eps, axis, input_scale=input_scale)
    layer.weight = nn.Parameter(bb.weight(name + ".weight", (dim,), offset=offset), requires_grad=False)
    return layer


def heads(x, count):
    # [1,C,1,W] -> [1,H,W,D]
    return x.reshape(1, count, -1, CHUNK).transpose(2, 3)


def channels(x):
    return x.transpose(2, 3).reshape(1, -1, 1, CHUNK)


class FeedForward(nn.Module):
    def __init__(self, bb, prefix):
        super().__init__()
        cfg = bb.cfg
        self.silu = precise_silu if bb.scaled_recurrence else F.silu
        self.norm = norm(bb, prefix + "post_attention_layernorm", cfg.hidden, 1)
        p = prefix + "mlp."
        # Large feed-forward projections exceed the ANE convolution channel
        # limit. Partition the intermediate dimension and add partial outputs.
        width = cfg.intermediate if cfg.intermediate <= 8192 else 4096
        shape = (cfg.intermediate, cfg.hidden)
        self.gate = nn.ModuleList(pointwise(w) for w in bb.weight(p + "gate_proj.weight", shape).split(width))
        self.up = nn.ModuleList(pointwise(w) for w in bb.weight(p + "up_proj.weight", shape).split(width))
        self.down = nn.ModuleList(pointwise(w.contiguous()) for w in
                                  bb.weight(p + "down_proj.weight", shape[::-1]).split(width, dim=1))

    def forward(self, residual):
        x = self.norm(residual)
        parts = [down(self.silu(gate(x)) * up(x)) for gate, up, down in zip(self.gate, self.up, self.down)]
        return residual + sum(parts)



class FullQuery(nn.Module):
    def __init__(self, bb, prefix):
        super().__init__()
        c = bb.cfg
        self.cfg = c
        self.input_norm = norm(bb, prefix + "input_layernorm", c.hidden, 1)
        p = prefix + "self_attn."
        self.q = projection(bb, p + "q_proj", c.hidden, 2 * c.heads * c.head_dim)
        self.k = projection(bb, p + "k_proj", c.hidden, c.kv_heads * c.head_dim)
        self.v = projection(bb, p + "v_proj", c.hidden, c.kv_heads * c.head_dim)
        self.q_norm = norm(bb, p + "q_norm", c.head_dim, 3)
        self.k_norm = norm(bb, p + "k_norm", c.head_dim, 3)

    def rotate(self, x, cos, sin):
        half = self.cfg.rotary_dim // 2
        a, b = x[..., :half], x[..., half:2 * half]
        return torch.cat((a * cos - b * sin, b * cos + a * sin, x[..., 2 * half:]), -1)

    def forward(self, residual, cos, sin):
        x = self.input_norm(residual)
        q, gate = heads(self.q(x), self.cfg.heads).chunk(2, -1)
        k = heads(self.k(x), self.cfg.kv_heads)
        cos, sin = cos.transpose(2, 3), sin.transpose(2, 3)
        return (self.rotate(self.q_norm(q), cos, sin), gate,
                self.rotate(self.k_norm(k), cos, sin), heads(self.v(x), self.cfg.kv_heads))


class DeltaLayer(nn.Module):
    """Block solution of the causal gated delta recurrence with carried K x V memory."""
    def __init__(self, bb, prefix, *, recurrence_scale=RECURRENCE_SCALE):
        super().__init__()
        c = bb.cfg
        self.cfg = c
        self.input_norm = norm(bb, prefix + "input_layernorm", c.hidden, 1)
        self.ff = FeedForward(bb, prefix)
        p = prefix + "linear_attn."
        self.mix = 2 * c.key_heads * c.key_dim + c.value_heads * c.value_dim
        self.qkv = projection(bb, p + "in_proj_qkv", c.hidden, self.mix)
        self.z = projection(bb, p + "in_proj_z", c.hidden, c.value_heads * c.value_dim)
        self.rate = projection(bb, p + "in_proj_a", c.hidden, c.value_heads)
        self.beta = projection(bb, p + "in_proj_b", c.hidden, c.value_heads)
        self.out = projection(bb, p + "out_proj", c.value_heads * c.value_dim, c.hidden)
        # L2 normalization is RMS normalization scaled by 1/sqrt(D):
        # sum(x^2) + eps = D * (mean(x^2) + eps/D).
        self.unit_norm = BoundedRMSNorm(c.key_dim, 1e-6 / c.key_dim, axis=3)
        self.unit_norm.weight = nn.Parameter(torch.full((c.key_dim,), c.key_dim ** -0.5,
                                                        dtype=bb.dtype), requires_grad=False)
        self.scaled_recurrence = bb.scaled_recurrence
        self.recurrence_scale = recurrence_scale
        self.valen_recurrence = bb.valen_recurrence
        self.gate_norm = norm(bb, p + "norm", c.value_dim, 3, offset=0,
                              input_scale=recurrence_scale if self.scaled_recurrence else 1)
        self.register_buffer("kernel", bb.weight(p + "conv1d.weight", (self.mix, 1, c.taps)))
        self.register_buffer("bias", bb.weight(p + "dt_bias", (c.value_heads,)).reshape(1,-1,1,1))
        self.register_buffer("rate_scale", -bb.weight(p + "A_log", (c.value_heads,)).float().exp()
                             .to(bb.dtype).reshape(1,-1,1,1))
        eye = torch.eye(TILE, dtype=bb.dtype)
        self.register_buffer("eye", eye)
        self.register_buffer("lower", torch.ones_like(eye).tril())
        self.register_buffer("strict_lower", self.lower - eye)
        self.history_size = self.mix * (c.taps - 1)
        self.memory_size = c.value_heads * c.key_dim * c.value_dim
        self.history_channels = math.ceil(self.history_size / CHUNK)
        self.state_channels = self.history_channels + math.ceil(self.memory_size / CHUNK)

    def tiles(self, x):
        # [1,H,W,D] -> [W/T,H,T,D]
        return x.reshape(x.shape[1], CHUNK // TILE, TILE, -1).transpose(0, 1)

    def unit_heads(self, x):
        c = self.cfg
        y = self.unit_norm(heads(x, c.key_heads))
        groups = c.value_heads // c.key_heads
        if groups > 1:
            # Replicate whole heads within their group using rank-four ops.
            y = torch.cat([y.transpose(2, 3)] * groups, dim=2)
            y = y.reshape(1, c.value_heads, c.key_dim, CHUNK).transpose(2, 3)
        return self.tiles(y)

    def delta(self, q, k, v, beta, rate, memory):
        # innovations U_i = beta_i (V_i - K_i M_{i-1} exp(g_i)).
        # Within a tile the cross-token terms form a strictly lower triangular
        # L. (I-L)^-1 is the finite product of (I+L^(2^j)).
        log_prefix = self.lower @ rate
        decay = (log_prefix - log_prefix.transpose(2, 3)).clamp_max(0).exp() * self.lower
        weighted_k = beta * k
        power = -(weighted_k @ k.transpose(2, 3)) * decay * self.strict_lower
        solve = self.eye + power
        for _ in range(1, int(math.log2(TILE))):
            # Core AI's ANE lowering requires distinct matmul SSA operands.
            right = torch.cat(power.chunk(2, dim=-1), dim=-1)
            power = power @ right
            solve = solve + power @ solve
        prefix = log_prefix.exp()
        innovations = solve @ (beta * v)
        incoming = solve @ (weighted_k * prefix)
        q = q * (self.cfg.key_dim ** -0.5)
        scores = (q @ k.transpose(2, 3)) * decay
        end_keys = (k * decay[:, :, -1:, :].transpose(2, 3)).transpose(2, 3)
        outputs = []
        for i in range(CHUNK // TILE):
            sl = slice(i, i + 1)
            update = innovations[sl] - incoming[sl] @ memory
            outputs.append((q[sl] * prefix[sl]) @ memory + scores[sl] @ update)
            memory = prefix[sl, :, -1:, :] * memory + end_keys[sl] @ update
        return torch.cat(outputs, dim=2), memory

    @staticmethod
    def unpack(state, shape):
        size = math.prod(shape)
        if size % CHUNK:
            state = state.reshape(-1)[:size]
        return state.reshape(shape)

    @staticmethod
    def pack(tensor):
        extra = -tensor.numel() % CHUNK
        if extra:
            tensor = F.pad(tensor.reshape(-1), (0, extra))
        return tensor.reshape(1, -1, 1, CHUNK)

    def forward(self, residual, state):
        c = self.cfg
        # Slice channel planes directly. Flattening a full-size matrix into one
        # dimension exceeds the ANE slice limit even when the final shape fits.
        history = self.unpack(state[:, :self.history_channels], (1, self.mix, 1, c.taps - 1))
        memory = self.unpack(state[:, self.history_channels:], (1, c.value_heads, c.key_dim, c.value_dim))
        x = self.input_norm(residual)
        extended = torch.cat((history, self.qkv(x)), dim=3)
        conv = sum(extended[..., tap:tap+CHUNK] * self.kernel[:, :, tap].reshape(1,-1,1,1)
                   for tap in range(c.taps))
        kd = c.key_heads * c.key_dim
        q, k, v = self.ff.silu(conv).split((kd, kd, c.value_heads * c.value_dim), dim=1)
        beta = self.beta(x).sigmoid().transpose(2, 3)
        a = self.rate(x).transpose(2, 3) + self.bias
        # Stable softplus uses only exponentials of nonpositive values.
        rate = (a.clamp_min(0) + torch.log1p((-a.abs()).exp())) * self.rate_scale
        # Delta is linear in both values and incoming memory. Keep the small
        # context in scaled units through gate normalization, then restore state units.
        scale = self.recurrence_scale
        if self.valen_recurrence:
            # Preserve operand evaluation order in the qualified Valen graph.
            out, memory = self.delta(
                self.unit_heads(q), self.unit_heads(k), self.tiles(heads(v * scale, c.value_heads)),
                self.tiles(beta), self.tiles(rate), memory * scale,
            )
        else:
            if self.scaled_recurrence:
                v = v * scale
                memory = memory * scale
            out, memory = self.delta(self.unit_heads(q), self.unit_heads(k), self.tiles(heads(v, c.value_heads)),
                                     self.tiles(beta), self.tiles(rate), memory)
        out = self.gate_norm(out) * self.ff.silu(heads(self.z(x), c.value_heads))
        if self.scaled_recurrence and not self.valen_recurrence:
            memory = memory / scale
        history = self.pack(extended[..., 1 - c.taps:])
        packed = torch.cat((history, self.pack(memory / scale if self.valen_recurrence else memory)), dim=1)
        return self.ff(residual + self.out(channels(out))), packed


class ScanStage(nn.Module):
    def __init__(self, backbone, start, end):
        super().__init__()
        cfg = backbone.cfg
        full = cfg.layer_types[end] == "full_attention"
        self.layers = nn.ModuleList(
            DeltaLayer(backbone, f"layers.{i}.", recurrence_scale=(
                256 if backbone.valen_recurrence and i == 0 else RECURRENCE_SCALE))
            for i in range(start, end + int(not full))
        )
        self.channels = max(1, sum(layer.state_channels for layer in self.layers))
        self.query = FullQuery(backbone, f"layers.{end}.") if full else None

    def forward(self, hidden_in, *args):
        state = args[-1]
        residual = hidden_in / RESIDUAL_SCALE
        states = []
        offset = 0
        for layer in self.layers:
            residual, updated = layer(residual, state[:, offset:offset+layer.state_channels])
            states.append(updated)
            offset += layer.state_channels
        if states:
            packed = torch.cat(states, dim=1)
            hidden_out = residual * RESIDUAL_SCALE
        else:
            # A projection-only scan has two identity results. Materialize both
            # inside the ANE region; passthrough buffers bypass its entry point.
            hidden_out = hidden_in.clamp_min(0) + hidden_in.clamp_max(0)
            packed = state.clamp_min(0) + state.clamp_max(0)
        if self.query is None:
            return hidden_out, packed
        return hidden_out, *self.query(residual, args[0], args[1]), packed


class AttendStage(nn.Module):
    def __init__(self, backbone, end, capacity):
        super().__init__()
        c = backbone.cfg
        p = f"layers.{end}."
        self.cfg = c
        self.capacity = capacity
        self.out = projection(backbone, p + "self_attn.o_proj", c.heads * c.head_dim, c.hidden)
        self.ff = FeedForward(backbone, p)
        self.final = norm(backbone, "norm", c.hidden, 1) if end == len(c.layer_types)-1 else None

    def forward(self, hidden_in, query, gate, key, value, mask):
        c = self.cfg
        grouped = query.reshape(1, c.kv_heads, -1, c.head_dim)
        scores = (grouped * (c.head_dim ** -0.5)) @ key.transpose(2, 3)
        probs = (scores.reshape(1, c.heads, CHUNK, self.capacity) + mask).softmax(dim=3)
        attended = probs.reshape_as(scores) @ value
        x = hidden_in / RESIDUAL_SCALE + self.out(channels(attended.reshape_as(query) * gate.sigmoid()))
        x = self.ff(x)
        return self.final(x) if self.final is not None else x * RESIDUAL_SCALE
