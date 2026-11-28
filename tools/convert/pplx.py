"""pplx-embed-v1-0.6b (bidirectional Qwen3) text encoder re-implemented for Core ML / ANE, following the
ANEEmbed eg2.py conventions:
  * RMSNorm divides each row by its max-abs first (fp16 overflow guard; exact in real arithmetic).
  * Fold attention layout (query heads sharing a K/V head folded into rows); no repeat_interleave.
  * RoPE tables are graph constants at absolute positions 0..S-1.
  * The mask is a runtime input (bias), never a graph constant.
Output: mean-pooled final hidden state (unnormalized float, as sentence-transformers before the int8
quantizer). The host applies int8 = clamp(round(tanh(x) * 127)) or normalizes.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

QUERY_ROWS = 2048
QUERY_BLOCK = 128
KEY_BLOCK = 4096
HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 16, 8, 128, 1024
# Keep normalized weights 32 times larger in the value matmul so small,
# useful weights do not flush to zero on the ANE. For the pinned weights,
# sqrt(HIDDEN) * max row norm of W_v * RMS weights is below 1372.
# The scaled value average thus fits in fp16 before scaling back down.
INVERSE_WEIGHT_SCALE = 1 / 32


def rms_safe(x, w=None, eps=1e-6):
    """RMSNorm without fp16 subnormals: out = x * (rsqrt(mean((x/s)^2) + eps/s^2) / s), s = max|x|.
    rms() computes x/s first, which for a row holding a massive activation (s ~ 1e4) pushes the ordinary
    channels below the fp16 normal range."""
    s = torch.maximum(x.abs().amax(-1, keepdim=True), torch.tensor(1e-4, dtype=x.dtype))
    y = x / s
    # Divide the square root first. Forming s*s in fp16 can underflow, and
    # the Core ML reciprocal rewrite adds an epsilon to the denominator.
    eps_root = torch.full_like(s, eps ** 0.5) / s
    inv = torch.rsqrt((y * y).mean(-1, keepdim=True) + eps_root * eps_root) / s
    out = x * inv
    return out * w if w is not None else out


# Residual-stream scale per layer (powers of two, exact in fp16). pplx-embed-v1-0.6b has massive activations:
# fp32 max |residual| ~3.5e4 from layer 10 and ~2.8e5 from layer 19 (fp16 max 65504). Every reader of the
# residual is an RMSNorm (scale invariant), so the stream is kept at alpha_i * x by rescaling x at the start of
# layer i and folding alpha_i into o_proj / down_proj.
PPLX_SCALES = ((10, 1 / 8), (19, 1 / 64))


def alphas(L, stages=PPLX_SCALES):
    a = []
    for i in range(L):
        v = 1.0
        for start, val in stages:
            if i >= start:
                v = val
        a.append(v)
    return a


def rot_half(x, h):
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_tables(n, hd, theta):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    fr = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    return emb.cos(), emb.sin()


class Weights(nn.Module):
    def __init__(self, hf_dir):
        super().__init__()
        with open(f"{hf_dir}/config.json") as source:
            cfg = json.load(source)
        f = safe_open(f"{hf_dir}/model.safetensors", "pt")
        keys = set(f.keys())
        pre = "model." if "model.norm.weight" in keys else ""

        def g(k):
            return nn.Parameter(f.get_tensor(pre + k).float(), requires_grad=False)

        self.D = cfg["hidden_size"]
        self.nh = cfg["num_attention_heads"]
        self.kvh = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.theta = cfg["rope_theta"]
        assert not cfg.get("use_sliding_window") and all(t == "full_attention" for t in cfg.get("layer_types", []))
        self.norm = g("norm.weight")
        self.layers = nn.ModuleList()
        for i in range(cfg["num_hidden_layers"]):
            m = nn.Module()
            m.input_layernorm = g(f"layers.{i}.input_layernorm.weight")
            m.post_attention_layernorm = g(f"layers.{i}.post_attention_layernorm.weight")
            for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                setattr(m, k, g(f"layers.{i}.self_attn.{k}.weight"))
            for k in ("gate_proj", "up_proj", "down_proj"):
                setattr(m, k, g(f"layers.{i}.mlp.{k}.weight"))
            self.layers.append(m)


def attend_fold(q, k, v, bias, kvh, mode, key_block=None, query_rows=None):
    nh, sq, hd, sk = (int(t) for t in (q.shape[1], q.shape[2], q.shape[3], k.shape[2]))
    g = nh // kvh
    qf = q.reshape(1, kvh, g * sq, hd)
    if key_block is not None:
        # Score each key block directly so the full Q-block by S tensor
        # never reaches ANE memory. This route takes a broadcast key mask.
        maxima, means, counts = [], [], []
        ones = torch.ones(key_block, 1, dtype=q.dtype)
        for start in range(0, sk, key_block):
            sl = slice(start, start + key_block)
            sj = torch.matmul(qf, k[:, :, sl].transpose(-1, -2))
            if query_rows is None:
                sj = sj + bias[..., sl]
            else:
                # The host encodes each key relative to the query tile. All row
                # numbers and sentinels are exact in fp16, even at 32K context.
                blocked = bias[..., sl] > query_rows
                sj = (sj.reshape(1, nh, sq, key_block) + blocked.to(sj.dtype) * -10000).reshape_as(sj)
            maximum = sj.amax(-1, keepdim=True)
            exp = torch.exp(sj - maximum)
            count = torch.matmul(exp, ones)
            # A block mean stays in the range of its values. Accumulating
            # exp @ V before dividing can saturate on the ANE in fp16.
            means.append(torch.matmul(exp / (count * INVERSE_WEIGHT_SCALE), v[:, :, sl]))
            maxima.append(maximum)
            counts.append(count)
        maximum = maxima[0]
        for block_max in maxima[1:]:
            maximum = torch.maximum(maximum, block_max)
        weights = [torch.exp(block_max - maximum) * count for block_max, count in zip(maxima, counts)]
        total = sum(weights)
        out = sum(mean * (weight / total) for mean, weight in zip(means, weights)) * INVERSE_WEIGHT_SCALE
        return out.reshape(1, nh, sq, hd)
    s = torch.matmul(qf, k.transpose(-1, -2)).reshape(1, nh, sq, sk) + bias
    if mode == "mmsum":
        # A matmul denominator avoids the loss of precision observed in the
        # ANE softmax at long key lengths.
        e = torch.exp(s - s.amax(-1, keepdim=True)).reshape(1, kvh, g * sq, sk)
        den = torch.matmul(e, torch.ones(sk, 1, dtype=e.dtype))
        out = torch.matmul(e / (den * INVERSE_WEIGHT_SCALE), v) * INVERSE_WEIGHT_SCALE
        return out.reshape(1, nh, sq, hd)
    if mode != "softmax":
        raise ValueError(f"unsupported attention mode {mode}")
    w = torch.softmax(s, dim=-1).reshape(1, kvh, g * sq, sk)
    return torch.matmul(w, v).reshape(1, nh, sq, hd)


def attend_blocked(q, k, v, bias, kvh, qblock, mode, key_block=None):
    """Query rows in static blocks of qblock against all keys: the score tensor per op is [.., qblock, S]
    instead of [.., S, S]. bias is [1,1,1,S] (key bias, one sequence) or [1,1,S,S]."""
    S = int(q.shape[2])
    outs = []
    for b in range(S // qblock):
        sl = slice(b * qblock, (b + 1) * qblock)
        bb = bias if int(bias.shape[2]) == 1 else bias[:, :, sl]
        outs.append(attend_fold(q[:, :, sl], k, v, bb, kvh, mode, key_block))
    return torch.cat(outs, dim=2)


def layer_safe(W, m, x, c, s, bias, o_w, d_w, qblock, mode):
    """layer() with rms_safe and pre-scaled o_proj / down_proj weights (o_w, d_w)."""
    nh, kvh, hd = W.nh, W.kvh, W.hd
    h = rms_safe(x, m.input_layernorm)
    q = rms_safe(F.linear(h, m.q_proj).reshape(1, -1, nh, hd).transpose(1, 2), m.q_norm)
    k = rms_safe(F.linear(h, m.k_proj).reshape(1, -1, kvh, hd).transpose(1, 2), m.k_norm)
    v = F.linear(h, m.v_proj).reshape(1, -1, kvh, hd).transpose(1, 2)
    q = (q * c + rot_half(q, hd // 2) * s) * (hd ** -0.5)
    k = k * c + rot_half(k, hd // 2) * s
    o = (attend_blocked(q, k, v, bias, kvh, qblock, mode) if qblock else attend_fold(q, k, v, bias, kvh, mode))
    o = o.transpose(1, 2).reshape(1, -1, nh * hd)
    x = x + F.linear(o, o_w)
    h = rms_safe(x, m.post_attention_layernorm)
    return x + F.linear(F.silu(F.linear(h, m.gate_proj)) * F.linear(h, m.up_proj), d_w)


class Bucket(nn.Module):
    """embed_S: inputs_embeds [1,S,D], bias [1,1,S,S], pool [K,S] -> pooled [K,D] (unnormalized mean)"""

    def __init__(self, W, S, *, mode, qblock=None, stages=PPLX_SCALES):
        super().__init__()
        self.W, self.qblock, self.mode = W, qblock, mode
        c, s = rope_tables(S, W.hd, W.theta)
        self.register_buffer("cos", c, persistent=False)
        self.register_buffer("sin", s, persistent=False)
        self.al = alphas(len(W.layers), stages)
        self.o_w = nn.ParameterList([nn.Parameter(m.o_proj * a, requires_grad=False)
                                     for m, a in zip(W.layers, self.al)])
        self.d_w = nn.ParameterList([nn.Parameter(m.down_proj * a, requires_grad=False)
                                     for m, a in zip(W.layers, self.al)])

    def forward(self, inputs_embeds, bias, pool):
        x = inputs_embeds
        prev = 1.0
        for i, m in enumerate(self.W.layers):
            if self.al[i] != prev:
                x = x * (self.al[i] / prev)
                prev = self.al[i]
            x = layer_safe(self.W, m, x, self.cos, self.sin, bias, self.o_w[i], self.d_w[i], self.qblock, self.mode)
        h = rms_safe(x, self.W.norm)[0]
        return torch.matmul(pool, h)


class Chunk(nn.Module):
    """Layers [lo, hi) of the staged encoder as one function, for compiling a long model in parts.
    first: inputs (inputs_embeds, bias); otherwise (hidden, bias), hidden being the previous part's output
    (residual stream at that part's scale). last: also takes pool and returns pooled [K,D] fp32; otherwise returns
    hidden [1,S,D]."""

    def __init__(self, W, S, lo, hi, *, mode, qblock):
        super().__init__()
        self.W, self.lo, self.hi, self.qblock, self.mode = W, lo, hi, qblock, mode
        self.last = hi == len(W.layers)
        c, s = rope_tables(S, W.hd, W.theta)
        self.register_buffer("cos", c, persistent=False)
        self.register_buffer("sin", s, persistent=False)
        al = alphas(len(W.layers))
        self.prev0 = al[lo - 1] if lo > 0 else 1.0
        self.al = al[lo:hi]
        self.o_w = nn.ParameterList([nn.Parameter(W.layers[i].o_proj * al[i], requires_grad=False) for i in range(lo, hi)])
        self.d_w = nn.ParameterList([nn.Parameter(W.layers[i].down_proj * al[i], requires_grad=False) for i in range(lo, hi)])

    def forward(self, x, bias, pool=None):
        prev = self.prev0
        for j, i in enumerate(range(self.lo, self.hi)):
            if self.al[j] != prev:
                x = x * (self.al[j] / prev)
                prev = self.al[j]
            x = layer_safe(self.W, self.W.layers[i], x, self.cos, self.sin, bias, self.o_w[j], self.d_w[j], self.qblock, self.mode)
        if self.last:
            return torch.matmul(pool, rms_safe(x, self.W.norm)[0])
        return x


class Pre(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, xs):
        m = self.m
        h = rms_safe(xs, m.input_layernorm)
        q = rms_safe(F.linear(h, m.q_proj).reshape(1, QUERY_ROWS, HEADS, HEAD_DIM), m.q_norm).reshape(1, QUERY_ROWS, HEADS * HEAD_DIM)
        k = rms_safe(F.linear(h, m.k_proj).reshape(1, QUERY_ROWS, KV_HEADS, HEAD_DIM), m.k_norm).reshape(1, QUERY_ROWS, KV_HEADS * HEAD_DIM)
        return q, k, F.linear(h, m.v_proj)

class Post(nn.Module):
    def __init__(self, m, a):
        super().__init__()
        self.m = m
        self.o_w = nn.Parameter(m.o_proj * a, requires_grad=False)
        self.d_w = nn.Parameter(m.down_proj * a, requires_grad=False)

    def forward(self, xs, o):
        m = self.m
        a = F.linear(o, self.o_w)
        h = rms_safe(xs + a, m.post_attention_layernorm)
        return a, F.linear(F.silu(F.linear(h, m.gate_proj)) * F.linear(h, m.up_proj), self.d_w)

class Attention(nn.Module):
    def forward(self, q, k, v, bias):
        return attend_blocked(q, k, v, bias, KV_HEADS, QUERY_BLOCK, mode="mmsum", key_block=KEY_BLOCK)


class CausalAttention(nn.Module):
    def forward(self, q, k, v, bias):
        key_block = min(KEY_BLOCK, int(k.shape[2]))
        blocks = []
        for start in range(0, QUERY_ROWS, QUERY_BLOCK):
            rows = torch.arange(start, start + QUERY_BLOCK, dtype=q.dtype).reshape(1, 1, QUERY_BLOCK, 1)
            blocks.append(attend_fold(q[:, :, start:start + QUERY_BLOCK], k, v, bias, KV_HEADS,
                                      "mmsum", key_block=key_block, query_rows=rows))
        return torch.cat(blocks, dim=2)
