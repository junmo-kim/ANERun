"""Reference implementation of the ANEEmbed host contract (pure numpy + Core ML; no torch).

This is the executable spec the Swift runtime mirrors. It only reads the shipped artifacts
(embeddings.bf16, position_embeddings.f16, tokenizer.json, the .mlmodelc packages).

Text, <= 1280 rows:     pack sequences into embed_S, S in BUCKETS (pack_bucket) -> one embedding row each.
Text, 1281..2048 rows:  windowed packing function winp{1536,2048}_b512 (pack_window); shorter sequences can
                        share the call. pack() picks the right feed for any packing size.
Text, 2049..8192 rows:  one call of winl{S}_b512, S = smallest of WINDOW_LONG >= n (long_feed), one document.
Image:                  HF Gemma4ImageProcessor pixels -> vision_{70,140,280} -> image tokens, wrapped as
                        <bos><|image> tokens <image|><eos> and embedded like a text sequence (packable).

Windowed functions take the sliding-window mask as a band, band_bias [1,1,S,1536]: row r, column c is query row
r against key row (r // 512) * 512 - 512 + c; 0 if both rows belong to the same sequence and |r - key| <= 512,
or key == r; else -1e4. Columns whose key row is outside [0, S) are never read by the graph.
"""
import math

import numpy as np

NEG = -1e4
WINDOW = 512
BUCKETS = (64, 128, 256, 512, 768, 1024, 1280)
BLOCK, HALO = 512, 512
BAND = BLOCK + 2 * HALO
WINDOW_PACK = (1536, 2048)
WINDOW_LONG = (3072, 4096, 6144, 8192)
PACK_SIZES = BUCKETS + WINDOW_PACK
SCALE = math.sqrt(512)
BOS, EOS, PAD, BOI, EOI, IMAGE = 2, 1, 0, 255999, 258882, 258880
DOC_PREFIX = "title: none | text: "


def k_for(S):
    return max(8, S // 32)


def function_name(S, long=False):
    """embed_S (S in BUCKETS), winp{S}_b512 (S in WINDOW_PACK), winl{S}_b512 (long=True, S in WINDOW_LONG)"""
    if long:
        assert S in WINDOW_LONG
        return f"winl{S}_b{BLOCK}"
    if S in BUCKETS:
        return f"embed_{S}"
    assert S in WINDOW_PACK
    return f"winp{S}_b{BLOCK}"


class TokenTable:
    """embeddings.bf16: 262144 rows x 512 bf16 (little endian). Input row = bf16 -> fp32 * sqrt(512)."""

    def __init__(self, path):
        raw = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, 512)
        self.raw = raw

    def rows(self, ids):
        u = self.raw[np.asarray(ids, np.int64)].astype(np.uint32) << 16
        return u.view(np.float32) * np.float32(SCALE)          # fp32 [n,512]


class Tokenizer:
    def __init__(self, path):
        from tokenizers import Tokenizer as T
        self.t = T.from_file(path)

    def ids(self, text, prefix=DOC_PREFIX):
        return [BOS] + self.t.encode(prefix + text, add_special_tokens=False).ids + [EOS]


# ---------------------------------------------------------------- packing functions (<= 2048 rows per call)

def bucket_for(total):
    """smallest packing size (embed_S or winp) that holds `total` rows"""
    return next(S for S in PACK_SIZES if S >= total)


def _segments(rows, S):
    K = k_for(S)
    assert len(rows) <= K and sum(len(r) for r in rows) <= S
    x = np.zeros((1, S, 512), np.float32)
    seg = np.full(S, -1)
    pool = np.zeros((K, S), np.float32)
    off = 0
    for k, r in enumerate(rows):
        n = len(r)
        x[0, off:off + n] = r
        seg[off:off + n] = k
        pool[k, off:off + n] = 1.0 / n
        off += n
    i = np.arange(S)
    same = ((seg[:, None] == seg[None, :]) & (seg[None, :] >= 0)) | (i[:, None] == i[None, :])
    g = np.where(same, 0.0, NEG).astype(np.float16).reshape(1, 1, S, S)
    return x, seg, pool, same, g


def pack_bucket(rows, S):
    """rows: list of fp32 [n_k,512] sequences (sum n_k <= S, count <= K). Returns the embed_S feed (fp16).
    Sequences occupy consecutive rows from 0; RoPE is relative so restarting positions is unnecessary
    (the functions use absolute positions 0..S-1 internally)."""
    x, seg, pool, same, g = _segments(rows, S)
    i = np.arange(S)
    loc = np.where(same & (np.abs(i[:, None] - i[None, :]) <= WINDOW), 0.0, NEG).astype(np.float16).reshape(1, 1, S, S)
    return {"inputs_embeds": x.astype(np.float16), "global_bias": g, "local_bias": loc, "pool": pool.astype(np.float16)}


def band_bias(S, seg):
    """seg [S]: sequence id per row (-1 = padding) -> band_bias [1,1,S,BAND] fp16 (see the module docstring)"""
    nb = S // BLOCK
    qa = np.arange(S).reshape(nb, BLOCK)[:, :, None]
    ka = (np.arange(nb) * BLOCK - HALO)[:, None, None] + np.arange(BAND)[None, None, :]
    kin = (ka >= 0) & (ka < S)
    kseg = np.where(kin, seg[np.clip(ka, 0, S - 1)], -1)
    qseg = seg[qa]
    ok = ((np.abs(qa - ka) <= WINDOW) & (kseg == qseg) & (kseg >= 0)) | (qa == ka)
    return np.where(ok, 0.0, NEG).astype(np.float16).reshape(1, 1, S, BAND)


def pack_window(rows, S):
    """winp{S}_b512 feed: as pack_bucket, with band_bias instead of local_bias."""
    x, seg, pool, _, g = _segments(rows, S)
    return {"inputs_embeds": x.astype(np.float16), "band_bias": band_bias(S, seg), "global_bias": g,
            "pool": pool.astype(np.float16)}


def pack(rows, S):
    """feed of the packing function of size S (embed_S or winp{S}_b512)"""
    return pack_bucket(rows, S) if S in BUCKETS else pack_window(rows, S)


# ---------------------------------------------------------------- long texts (2049..8192 rows, one per call)

def long_size(n):
    return next(S for S in WINDOW_LONG if S >= n)


LONG_POOL_WEIGHT = 1.0 / 1024


def long_pool(n, S):
    """[1,S] fp16: 2^-10 on the n valid rows. NOT 1/n: with 1/n (1.2e-4 at n=8192) the fp16 products in the
    in-graph pooling underflow and cos vs fp32 drops to ~0.9996; the output is L2-normalized, so the scale is free.
    2^-10 keeps the pooled sum <= 8x the mean (1/16 overflowed fp16 in tests)."""
    return np.where(np.arange(S) < n, LONG_POOL_WEIGHT, 0.0).astype(np.float16).reshape(1, S)


def long_feed(rows, S):
    """winl{S}_b512 feed for one sequence rows [n,512] (n <= S): token rows, band, key_bias, pool"""
    n = len(rows)
    assert n <= S
    x = np.zeros((1, S, 512), np.float16)
    x[0, :n] = rows
    seg = np.where(np.arange(S) < n, 0, -1)
    return {"inputs_embeds": x, "band_bias": band_bias(S, seg),
            "key_bias": np.where(np.arange(S) < n, 0.0, NEG).astype(np.float16).reshape(1, 1, 1, S),
            "pool": long_pool(n, S)}


# ---------------------------------------------------------------- images

class Vision:
    """vision_{B} feed from HF Gemma4ImageProcessor outputs (pixel_values [P,768] in [0,1], position ids [P,2]
    as (x, y), -1 for padding, P = 9*B)."""

    def __init__(self, table_path):
        self.table = np.fromfile(table_path, np.float16).reshape(2, 1024, 768).astype(np.float32)

    def feed(self, pixel_values, pos, budget):
        P = budget * 9
        pv = np.asarray(pixel_values, np.float32).reshape(P, 768)
        pos = np.asarray(pos, np.int64).reshape(P, 2)
        valid = (pos[:, 0] >= 0)
        nv = int(valid.sum())
        cols = int(pos[valid, 0].max()) + 1
        pe = np.zeros((P, 768), np.float32)
        pe[valid] = self.table[0, pos[valid, 0]] + self.table[1, pos[valid, 1]]
        pool = np.zeros((budget, P), np.float32)
        for p in np.nonzero(valid)[0]:
            xx, yy = pos[p]
            pool[(xx // 3) + (cols // 3) * (yy // 3), p] = 1.0 / 9.0
        return {"patches": pv[None].astype(np.float16), "positions": pos[None].astype(np.float16),
                "position_embeddings": pe[None].astype(np.float16), "valid": valid[None].astype(np.float16),
                "pool": pool.astype(np.float16)}, nv // 9


def image_sequence(table, tokens):
    """<bos><|image> tokens <image|><eos> rows (special rows scaled, image tokens as produced)"""
    return np.concatenate([table.rows([BOS, BOI]), np.asarray(tokens, np.float32), table.rows([EOI, EOS])])


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
