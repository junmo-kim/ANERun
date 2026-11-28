"""Numerical verification of the shipped package on CPU+ANE vs the fp32 torch reference (eg2.Reference, which
equals HF sentence-transformers at cos 1.000000, see verify_hf.py). Inputs are built only from shipped artifacts
through host.py (the documented host contract).

Usage: verify.py text <dist> <corpus.json> --hf-dir <snapshot>    embed_S and winp: single full/half, packed short,
                                                                  mixed long + short, two long
       verify.py long <dist> <corpus.json> --hf-dir <snapshot>    winl: one text of 2049..8192 tokens per call,
                                                                  incl. the lengths at and just over each size
       verify.py pplx <dist> <corpus.json> --hf-dir <snapshot> [--binary PATH]
                Swift HTTP output vs quantized fp32 means (corpus entries supply raw vectors or text for HF).
corpus.json comes from corpus.py or a list of evaluation documents, and the snapshot from fetch.py models.
Prints one line per case and a final SUMMARY json {category: {n, min, median}}."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("text", "long", "pplx"))
ap.add_argument("dist")
ap.add_argument("corpus")
ap.add_argument("--hf-dir", required=True, help="pinned snapshot for the selected model")
ap.add_argument("--binary", help="anerun Swift executable (pplx)")
ap.add_argument("--timeout", type=float, default=1800, help="seconds to wait for Swift readiness or one HTTP call (pplx)")
args = ap.parse_args()
if args.mode == "pplx":
    import math
    from pplx_verify import run

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        ap.error("--timeout must be a finite positive number")
    sys.exit(run(args.dist, args.corpus, args.hf_dir, args.binary, args.timeout))

import coremltools as ct  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import host  # noqa: E402
from eg2 import Weights, ref_embed  # noqa: E402

torch.set_num_threads(8)
mode, dist, corpus = args.mode, args.dist, args.corpus
W = Weights(args.hf_dir)
tab = host.TokenTable(f"{dist}/embeddings.bf16")
tok = host.Tokenizer(f"{dist}/tokenizer.json")
docs = json.load(open(corpus))
ids = [tok.ids(d["text"]) for d in docs]
PKG = f"{dist}/EmbeddingGemma2Text.mlmodelc"
CU = ct.ComputeUnit.CPU_AND_NE
res = {}


def fn(name):
    return ct.models.CompiledMLModel(PKG, compute_units=CU, function_name=name)


def text_rows(d, n, start=0):
    body = ids[d % len(ids)][1:-1]
    start = start % max(1, len(body) - n)
    seq = [host.BOS] + body[start:start + n - 2] + [host.EOS]
    assert len(seq) == n, (len(seq), n)
    return tab.rows(seq)


allv = []


def add(cat, c, total=True):
    res.setdefault(cat, []).append(c)
    if total:
        allv.append(c)


def summary():
    s = {k: {"n": len(v), "min": round(min(v), 6), "median": round(float(np.median(v)), 6)} for k, v in res.items()}
    s["ALL"] = {"n": len(allv), "min": round(min(allv), 6), "median": round(float(np.median(allv)), 6)}
    print("SUMMARY", json.dumps(s), flush=True)


if mode == "text":
    MIXED = {1024: 600, 1280: 850, 1536: 1100, 2048: 1500}   # + 422 rows of short texts <= S
    for S in host.PACK_SIZES:
        name = host.function_name(S)
        m = fn(name)
        K = host.k_for(S)
        rng = np.random.default_rng(S)
        cases = []
        for d in (0, 1, 2, 3, 5, 6):                 # en, ko, ja, zh, ko-constitution, ja-constitution
            cases.append(("single_full", [(d, S)]))
            cases.append(("single_half", [(d, S // 2 + 5)]))
        for trial in range(3):
            lens, tot = [], 0
            while len(lens) < K:
                n = int(rng.integers(6, max(8, min(S // 4, 200))))
                if tot + n > S:
                    break
                lens.append(n)
                tot += n
            cases.append(("packed_short", [((trial * 5 + j) % len(ids), n) for j, n in enumerate(lens)]))
        if S >= 1024:
            cases.append(("mixed_long_short", [(1, MIXED[S]), (4, 120), (11, 64), (6, 200), (8, 31), (2, 7)]))
            cases.append(("two_long", [(5, S // 2 - 40), (9, S // 2 - 30)]))
        for cat, spec in cases:
            rows = [text_rows(d, n, start=53 * d + n) for d, n in spec]
            t = time.perf_counter()
            out = m.predict(host.pack(rows, S))["embedding"]
            dt = (time.perf_counter() - t) * 1000
            cs = [host.cos(out[k], ref_embed(W, r)) for k, r in enumerate(rows)]
            zero = float(np.abs(out[len(rows):]).max()) if len(rows) < K else 0.0
            for c in cs:
                add(cat, c)
                add(name, c, total=False)
            print(f"{name} {cat} lens={[n for _, n in spec]} cos={[round(c, 6) for c in cs]} unused-max|x|={zero:.3g} "
                  f"predict={dt:.0f} ms", flush=True)
        del m
    summary()

if mode == "long":
    fns = {}
    t0 = time.time()
    for S in host.WINDOW_LONG:
        fns[S] = fn(host.function_name(S, long=True))
    print(f"loaded {len(fns)} long functions in {time.time() - t0:.1f}s", flush=True)
    lengths = [2049, 2600, 3072, 3073, 3500, 4096, 4097, 5000, 6144, 6145, 7000, 8191, 8192]
    j = 0
    for L in lengths:
        cand = [i for i, x in enumerate(ids) if len(x) >= L]
        S = host.long_size(L)
        for r in range(4 if L in (2049, 3073, 4097, 6145, 8192) else 3):
            d = cand[(j + r * 3) % len(cand)]
            seq = ids[d][:L - 1] + [host.EOS]
            rows = tab.rows(seq)
            t = time.perf_counter()
            e = np.asarray(fns[S].predict(host.long_feed(rows, S))["embedding"][0], np.float32)
            dt = (time.perf_counter() - t) * 1000
            c = host.cos(e, ref_embed(W, rows))
            add(f"winl{S}", c)
            print(f"long n={L} winl{S} doc={docs[d]['name']} cos={c:.6f} predict={dt:.0f} ms", flush=True)
        j += 1
    summary()
