"""EmbeddingGemma 2 text encoder, re-implemented for Core ML / Apple Neural Engine (ANE) conversion.

Everything here mirrors transformers' EmbeddingGemma2TextModel (+ mean pooling + L2 normalize) and is checked
against it (see verify_hf.py). Differences are numerically neutral rewrites that make the fp16 ANE graph safe:
  * RMSNorm divides each row by its max-abs first (fp16 overflow guard; exact in real arithmetic).
  * Attention uses the "fold" layout: the query heads sharing one K/V head are folded into the row dimension
    ([1,kvh,g*S,hd] @ [1,kvh,hd,S]). The obvious repeat_interleave layout makes the ANE write zeros for query
    heads 1..3 of the global layers (5/11/17/23, kvh=1) at large S. Do not change this without re-running the
    self-test on the ANE.
  * RoPE tables are graph constants at absolute positions 0..S-1. A runtime positions / rope input miscompiles.
  * Every mask is a runtime input (bias tensor). Graph-constant masks miscompile on the ANE for S > 512.
  * The per-layer-embedding (PLE) projection is computed per layer from a [512,512] slice of the [12288,512]
    weight, so every function carries identical weight constants and the multifunction merge stores them once.
  * Sliding layers of the windowed functions attend block by block: query block b (B = 512 rows) sees the key
    rows b*B - 512 .. b*B + B + 512 (static, clipped slices; the band mask is an input). Padding the keys in the
    graph (F.pad) made the layer about 3x slower on the ANE, so the slices are clipped instead.
  * Global layers of the long functions (winl) attend in blocks: query blocks of GLOBAL_QB rows, each against
    key blocks of global_kb(S) keys with a flash-style combine, every softmax denominator a matmul with a ones
    vector. The ANE softmax loses accuracy as the key length grows, and one [g*S, S] score tensor is slow: on an
    M4, winl8192 went from 1457 ms to 1240 ms, and the minimum cos vs fp32 from 0.99989 (9 texts) to 0.999979
    (3 texts); see README.

Modules (all fp32 here; converted with compute_precision=FLOAT16):
  Bucket(S)           embed_S          up to k_for(S) packed sequences of total length <= S, full S x S attention
  Window(S)           winp{S}_b512     packed sequences, total <= S: sliding layers over [S, 1536] bands,
                                       global layers over [S, S] (global_bias)
                      winl{S}_b512     one sequence of n <= S rows: sliding layers over bands, global layers
                                       over all S keys with key_bias; pool [1, S] (2^-10 on valid rows)
  Reference(S)        fp32 ground truth (attention_mask input, constant band), HF-equivalent
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

NEG = -1e4            # additive mask value (fp16-safe)
WINDOW = 512          # config.sliding_window; HF bidirectional sliding mask keeps |i - j| <= 512
BUCKETS = (64, 128, 256, 512, 768, 1024, 1280)   # embed_S: full attention, packing
BLOCK = 512           # windowed functions: query rows per sliding-attention block
HALO = 512            # key rows on each side of a block (== WINDOW)
BAND = BLOCK + 2 * HALO   # band_bias columns: keys b*BLOCK - HALO .. b*BLOCK + BLOCK + HALO - 1
WINDOW_PACK = (1536, 2048)                # winp{S}_b512: windowed, packing
WINDOW_LONG = (3072, 4096, 6144, 8192)    # winl{S}_b512: windowed, one sequence per call
GLOBAL_QB = 128       # winl global layers: query rows per block
SCALE = math.sqrt(512)


def global_kb(S):
    """winl global layers: keys per block (a divisor of S)"""
    return 2048 if S % 2048 == 0 else 1024


def k_for(S):
    """pooled slots (packed sequences) per bucket call"""
    return max(8, S // 32)


def rms(x, w=None, eps=1e-6):
    s = torch.maximum(x.abs().amax(-1, keepdim=True), torch.tensor(1e-4, dtype=x.dtype))
    y = x / s
    out = y * torch.rsqrt((y * y).mean(-1, keepdim=True) + eps / (s * s))
    return out * w if w is not None else out


def rot_half(x, h):
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_tables(n, hd, theta):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    fr = torch.arange(n, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    return emb.cos(), emb.sin()


GELU_C = 0.7978845608028654  # sqrt(2 / pi)
# Functions whose MLPs use the written-out GELU. It costs 0-2% at 64/128 rows and about 10% in the long functions;
# 256-2048 keep Core ML's gelu op (written out it cost 13-20% there).
EXACT_GELU_SIZES = (64, 128) + WINDOW_LONG


def gelu(x):
    """GELU, tanh approximation, written out: Core ML's gelu op loses about ten times more precision in fp16 on the
    Neural Engine than this form (measured on the vision MLP: relative error 0.0144 vs 0.0012)."""
    return x * (0.5 + 0.5 * torch.tanh(x * (GELU_C + GELU_C * 0.044715 * x * x)))


def attend_fold(q, k, v, bias, kvh):
    """q [1,4,Sq,hd], k/v [1,kvh,Sk,hd], bias broadcastable to [1,4,Sq,Sk] -> [1,4,Sq,hd]"""
    nh, sq, hd, sk = (int(t) for t in (q.shape[1], q.shape[2], q.shape[3], k.shape[2]))  # static when traced
    g = nh // kvh
    qf = q.reshape(1, kvh, g * sq, hd)
    s = torch.matmul(qf, k.transpose(-1, -2)).reshape(1, nh, sq, sk) + bias
    w = torch.softmax(s, dim=-1).reshape(1, kvh, g * sq, sk)
    return torch.matmul(w, v).reshape(1, nh, sq, hd)


class Weights(nn.Module):
    """fp32 copies of the text-model weights (language_model.* in model.safetensors)."""

    def __init__(self, hf_dir):
        super().__init__()
        cfg = json.load(open(f"{hf_dir}/config.json"))["text_config"]
        self.cfg = cfg
        f = safe_open(f"{hf_dir}/model.safetensors", "pt")

        def g(k):
            return nn.Parameter(f.get_tensor("language_model." + k).float(), requires_grad=False)

        self.L = cfg["num_hidden_layers"]
        assert cfg["sliding_window"] == WINDOW and cfg["hidden_size"] == 512
        ple_w = g("ple.per_layer_model_projection.weight")          # [24*512, 512]
        self.ple_norm = g("ple.per_layer_projection_norm.weight")
        self.norm = g("norm.weight")
        self.proj = g("embedding_projection.weight")                  # [768, 512]
        self.layers = nn.ModuleList()
        for i, t in enumerate(cfg["layer_types"]):
            m = nn.Module()
            for k in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                      "post_feedforward_layernorm", "ple_block.post_per_layer_input_norm"):
                setattr(m, k.replace(".", "_"), g(f"layers.{i}.{k}.weight"))
            for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                setattr(m, k, g(f"layers.{i}.self_attn.{k}.weight"))
            for k in ("gate_proj", "up_proj", "down_proj"):
                setattr(m, k, g(f"layers.{i}.mlp.{k}.weight"))
            m.ple_gate = g(f"layers.{i}.ple_block.per_layer_input_gate.weight")
            m.ple_proj = g(f"layers.{i}.ple_block.per_layer_projection.weight")
            m.ple_w = nn.Parameter(ple_w[i * 512:(i + 1) * 512].clone(), requires_grad=False)
            m.scalar = g(f"layers.{i}.layer_scalar")
            m.hd = int(m.q_norm.shape[0])
            m.kvh = int(m.k_proj.shape[0]) // m.hd
            m.swa = t == "sliding_attention"
            m.theta = cfg["rope_parameters"][t]["rope_theta"]
            self.layers.append(m)
        self.global_layers = [i for i, m in enumerate(self.layers) if not m.swa]


def layer(W, i, x, x0, c, s, attend, exact_gelu=True):
    """Decoder layer i over x [1,S,512]; x0 = inputs_embeds (per-layer input).
    attend(q, k, v, kvh) -> [1,4,S,hd] is the attention (masks are bound by the caller)."""
    m = W.layers[i]
    hd, kvh = m.hd, m.kvh
    h = rms(x, m.input_layernorm)
    q = rms(F.linear(h, m.q_proj).reshape(1, -1, 4, hd).transpose(1, 2), m.q_norm)
    k = rms(F.linear(h, m.k_proj).reshape(1, -1, kvh, hd).transpose(1, 2), m.k_norm)
    v = rms(F.linear(h, m.v_proj).reshape(1, -1, kvh, hd).transpose(1, 2))
    q = q * c + rot_half(q, hd // 2) * s
    k = k * c + rot_half(k, hd // 2) * s
    o = attend(q, k, v, kvh).transpose(1, 2).reshape(1, -1, 4 * hd)
    x = x + rms(F.linear(o, m.o_proj), m.post_attention_layernorm)
    h = rms(x, m.pre_feedforward_layernorm)
    act = gelu if exact_gelu else (lambda t: F.gelu(t, approximate="tanh"))
    h = F.linear(act(F.linear(h, m.gate_proj)) * F.linear(h, m.up_proj), m.down_proj)
    x = x + rms(h, m.post_feedforward_layernorm)
    ple = rms(F.linear(x0, m.ple_w) * (512 ** -0.5), W.ple_norm)
    gte = act(F.linear(x, m.ple_gate)) * ple
    x = x + rms(F.linear(gte, m.ple_proj), m.ple_block_post_per_layer_input_norm)
    return x * m.scalar


def full_attention(bias):
    """all keys, one additive bias broadcastable to [1,4,S,S]"""
    def attend(q, k, v, kvh):
        return attend_fold(q, k, v, bias, kvh)
    return attend


def blocked_attention(key_bias, qb, kb):
    """Global layers of winl: query blocks of qb rows in the fold layout, each against key blocks of kb keys.
    Per key block j: m_j = max of the scores, e_j = exp(s_j - m_j), numerator e_j @ v_j and denominator
    e_j @ 1 (a matmul, not a reduce_sum); the blocks are combined with weights exp(m_j - max_j m_j).
    key_bias [1,1,1,S]. Equal to full_attention in fp32. A text runs in the smallest winl size that holds it
    (host.long_size), so the last key block always holds a real key; a block of padded keys alone would get the
    weight exp(-1e4 + ...) = 0 anyway."""
    def attend(q, k, v, kvh):
        nh, S, hd, sk = (int(t) for t in (q.shape[1], q.shape[2], q.shape[3], k.shape[2]))
        assert S % qb == 0 and sk % kb == 0, f"blocked_attention: {S} rows / {qb}, {sk} keys / {kb} do not divide"
        g = nh // kvh
        ones = torch.ones(kb, 1, dtype=q.dtype)
        outs = []
        for b in range(S // qb):
            qf = q[:, :, b * qb:(b + 1) * qb].reshape(1, kvh, g * qb, hd)
            ms, nums, dens = [], [], []
            for j in range(sk // kb):
                ks = slice(j * kb, (j + 1) * kb)
                s = torch.matmul(qf, k[:, :, ks].transpose(-1, -2)) + key_bias[..., ks]
                m = s.amax(-1, keepdim=True)
                e = torch.exp(s - m)
                ms.append(m)
                nums.append(torch.matmul(e, v[:, :, ks]))
                dens.append(torch.matmul(e, ones))
            top = ms[0]
            for m in ms[1:]:
                top = torch.maximum(top, m)
            num, den = 0, 0
            for m, n_, d_ in zip(ms, nums, dens):
                w = torch.exp(m - top)
                num = num + n_ * w
                den = den + d_ * w
            outs.append((num / den).reshape(1, nh, qb, hd))
        return torch.cat(outs, dim=2)
    return attend


def band_attention(band, S):
    """Block-local attention: query block b (rows b*BLOCK ..) against key rows b*BLOCK - HALO .. b*BLOCK + BLOCK +
    HALO, clipped to [0, S). band [1,1,S,BAND]: row r, column c <-> key row (r // BLOCK) * BLOCK - HALO + c
    (columns outside [0, S) are never read). Unrolled over the S / BLOCK blocks on static slices."""
    def attend(q, k, v, kvh):
        outs = []
        for b in range(S // BLOCK):
            lo = b * BLOCK - HALO
            clo, chi = max(lo, 0), min(lo + BAND, S)
            bb = band[:, :, b * BLOCK:(b + 1) * BLOCK, clo - lo:chi - lo]
            outs.append(attend_fold(q[:, :, b * BLOCK:(b + 1) * BLOCK], k[:, :, clo:chi], v[:, :, clo:chi], bb, kvh))
        return torch.cat(outs, dim=2)
    return attend


def head(W, x, pool):
    """x [1,S,512], pool [K,S] -> [K,768] L2-normalized (all-zero pool rows give zero rows)"""
    h = rms(x, W.norm)[0]
    e = F.linear(torch.matmul(pool, h), W.proj)
    s = torch.maximum(e.abs().amax(-1, keepdim=True), torch.tensor(1e-4, dtype=e.dtype))
    e = e / s
    return e * torch.rsqrt((e * e).sum(-1, keepdim=True) + 1e-6)


class _Tables(nn.Module):
    def __init__(self, W, n):
        super().__init__()
        for name, hd in (("swa", 256), ("glb", 512)):
            th = next(m.theta for m in W.layers if (m.hd == hd))
            c, s = rope_tables(n, hd, th)
            self.register_buffer(f"cos_{name}", c, persistent=False)
            self.register_buffer(f"sin_{name}", s, persistent=False)

    def get(self, m):
        n = "swa" if m.swa else "glb"
        return getattr(self, f"cos_{n}"), getattr(self, f"sin_{n}")


class Bucket(nn.Module):
    """embed_S: inputs_embeds [1,S,512], global_bias [1,1,S,S], local_bias [1,1,S,S], pool [K,S] -> [K,768]"""

    def __init__(self, W, S):
        super().__init__()
        self.W, self.S = W, S
        self.t = _Tables(W, S)

    def forward(self, inputs_embeds, global_bias, local_bias, pool):
        x = inputs_embeds
        glb, loc = full_attention(global_bias), full_attention(local_bias)
        for i, m in enumerate(self.W.layers):
            c, s = self.t.get(m)
            x = layer(self.W, i, x, inputs_embeds, c, s, loc if m.swa else glb, self.S in EXACT_GELU_SIZES)
        return head(self.W, x, pool)


class Window(nn.Module):
    """winp{S}_b512 (packed): inputs_embeds [1,S,512], band_bias [1,1,S,BAND], global_bias [1,1,S,S], pool [K,S]
    winl{S}_b512 (long):    inputs_embeds [1,S,512], band_bias [1,1,S,BAND], key_bias [1,1,1,S],    pool [1,S]
    -> embedding [K or 1, 768]. Sliding layers use band_attention. Global layers attend to all S keys with the
    third input: winp with full_attention over global_bias, winl with blocked_attention over key_bias."""

    def __init__(self, W, S):
        super().__init__()
        assert S % BLOCK == 0
        self.W, self.S = W, S
        self.t = _Tables(W, S)

    def forward(self, inputs_embeds, band_bias, global_or_key_bias, pool):
        x = inputs_embeds
        loc = band_attention(band_bias, self.S)
        if int(global_or_key_bias.shape[2]) == 1:      # winl: key_bias [1,1,1,S], one sequence
            glb = blocked_attention(global_or_key_bias, GLOBAL_QB, global_kb(self.S))
        else:                                           # winp: global_bias [1,1,S,S], packed sequences
            glb = full_attention(global_or_key_bias)
        for i, m in enumerate(self.W.layers):
            c, s = self.t.get(m)
            x = layer(self.W, i, x, inputs_embeds, c, s, loc if m.swa else glb, self.S in EXACT_GELU_SIZES)
        return head(self.W, x, pool)


class Reference(nn.Module):
    """fp32 ground truth for ONE sequence: inputs_embeds [1,S,512], attention_mask [1,S] -> [1,768].
    Standard (non-fold) attention with repeat_interleave, constant sliding band, mask-based mean pooling;
    equals sentence-transformers EmbeddingGemma 2 at cos 1.000000 (verify_hf.py)."""

    def __init__(self, W, S):
        super().__init__()
        self.W, self.S = W, S
        self.t = _Tables(W, S)
        i = torch.arange(S)
        self.register_buffer("band", torch.where((i[:, None] - i[None, :]).abs() <= WINDOW, 0.0, NEG)
                             .reshape(1, 1, S, S), persistent=False)

    def forward(self, inputs_embeds, attention_mask):
        W, S = self.W, self.S
        x = inputs_embeds
        pad = ((1.0 - attention_mask) * NEG).reshape(1, 1, 1, S)
        for i, m in enumerate(W.layers):
            c, s = self.t.get(m)
            hd, kvh = m.hd, m.kvh
            h = rms(x, m.input_layernorm)
            q = rms(F.linear(h, m.q_proj).reshape(1, S, 4, hd).transpose(1, 2), m.q_norm)
            k = rms(F.linear(h, m.k_proj).reshape(1, S, kvh, hd).transpose(1, 2), m.k_norm)
            v = rms(F.linear(h, m.v_proj).reshape(1, S, kvh, hd).transpose(1, 2))
            q = q * c + rot_half(q, hd // 2) * s
            k = (k * c + rot_half(k, hd // 2) * s).repeat_interleave(4 // kvh, dim=1)
            v = v.repeat_interleave(4 // kvh, dim=1)
            bias = pad + self.band if m.swa else pad
            w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + bias, dim=-1)
            o = torch.matmul(w, v).transpose(1, 2).reshape(1, S, 4 * hd)
            x = x + rms(F.linear(o, m.o_proj), m.post_attention_layernorm)
            h = rms(x, m.pre_feedforward_layernorm)
            h = F.linear(F.gelu(F.linear(h, m.gate_proj), approximate="tanh") * F.linear(h, m.up_proj), m.down_proj)
            x = x + rms(h, m.post_feedforward_layernorm)
            ple = rms(F.linear(inputs_embeds, m.ple_w) * (512 ** -0.5), W.ple_norm)
            gte = F.gelu(F.linear(x, m.ple_gate), approximate="tanh") * ple
            x = x + rms(F.linear(gte, m.ple_proj), m.ple_block_post_per_layer_input_norm)
            x = x * m.scalar
        h = rms(x, W.norm)
        wpool = (attention_mask / attention_mask.sum(-1, keepdim=True)).reshape(1, 1, S)
        e = F.linear(torch.matmul(wpool, h), W.proj)[:, 0, :]
        return e / e.norm(dim=-1, keepdim=True)


_REFS = {}


def ref_embed(W, rows):
    """fp32 reference embedding of one sequence given its input rows [n,512] (fp32)"""
    n = len(rows)
    S = 1 << max(6, (n - 1).bit_length())
    if S not in _REFS:
        _REFS[S] = Reference(W, S).eval()
    x = torch.zeros(1, S, 512)
    x[0, :n] = torch.as_tensor(np.asarray(rows, np.float32))
    msk = torch.zeros(1, S)
    msk[0, :n] = 1
    with torch.no_grad():
        return _REFS[S](x, msk)[0].numpy()
