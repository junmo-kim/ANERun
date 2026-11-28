"""Golden self-test set: generation (fp32 torch / HF, CPU) and the reference runner (Core ML, CPU+ANE).

    python selftest.py run <dist_dir> [--functions a,b,...]     # what the runtime does at load (exit 1 on failure)

Every test builds its inputs deterministically from data in selftest.json (token ids, a procedural image) plus
the shipped embeddings.bf16 / position_embeddings.f16, runs ONE function, and compares with fp32 expectations
in selftest.bin (little-endian float32, offsets in selftest.json). Inputs:

  bucket embed_S     sequences (token id lists incl. <bos>/<eos>) packed with host.pack_bucket
                     -> compare each embedding row (cosine) and require unused rows ~0
  winp winp{S}_b512  the same with host.pack_window (band_bias instead of local_bias); one sequence spans
                     several 512-row blocks so the band and the block boundaries are exercised
  winl winl{S}_b512  document = first `doc_rows` ids of `doc_ids` (n > the next smaller long size or 2048, so
                     the last block is partial and padding rows exist), fed with host.long_feed
                     -> compare the embedding (cosine)
  vision_{B}         procedural image (see pattern()) of gx x gy patches -> compare image tokens

Expectations: text = eg2.Reference fp32 (equal to Hugging Face, verify_hf.py), vision = Hugging Face fp32 image
features. Vision compares the digest D = image_tokens[valid tokens] @ R, R = digest matrix [512,8] from
selftest.bin. Metrics: cosine (text: per embedding row; vision: of flattened D) and relative error
|Da - Dr|_F / |Dr|_F (vision).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host  # noqa: E402

WINP_LENGTHS = {1536: (1300, 37, 150), 2048: (1700, 37, 200, 9)}
WINL_ROWS = {3072: 2600, 4096: 3500, 6144: 5900, 8192: 8000}
VISION_GRIDS = {70: (27, 18), 140: (36, 30), 280: (48, 45)}   # patch grid (cols, rows): 54 / 120 / 240 tokens
DIGEST_DIMS = 8


def pattern(gx, gy, budget):
    """Procedural image as Gemma4ImageProcessor-style outputs: pixel_values [9B,768] (row-major patches, each
    [16 rows][16 cols][RGB]), position ids [9B,2] (x, y; -1 padding). Integer formula, exact in any language:
    value(X, Y, c) = ((5X + 3Y + 70c + 90 * ((X // 24 + Y // 24) % 2)) % 256) / 255, X/Y = pixel column/row."""
    P = 9 * budget
    pv = np.zeros((P, 768), np.float32)
    pos = np.full((P, 2), -1, np.int64)
    r, c, ch = np.meshgrid(np.arange(16), np.arange(16), np.arange(3), indexing="ij")
    for py in range(gy):
        for px in range(gx):
            p = py * gx + px
            X, Y = px * 16 + c, py * 16 + r
            v = (5 * X + 3 * Y + 70 * ch + 90 * ((X // 24 + Y // 24) % 2)) % 256
            pv[p] = (v.astype(np.float32) / 255.0).reshape(-1)
            pos[p] = (px, py)
    return pv, pos


def framed(doc, lens):
    """consecutive slices of doc, each framed as <bos> ... <eos>, of the given total lengths"""
    seqs, off = [], 0
    for n in lens:
        seqs.append([host.BOS] + doc[off:off + n - 2] + [host.EOS])
        off += n
    return seqs


def bucket_sequences(doc, S):
    return framed(doc, {64: (40, 11, 9), 128: (80, 30, 12)}.get(S, (S * 5 // 8, 37, S // 8 + 3)))


# ------------------------------------------------------------------------------------------- generation

def generate(hf_dir, out_dir):
    import torch
    import build
    from eg2 import Weights, ref_embed
    torch.set_num_threads(8)
    W = Weights(hf_dir)
    tab = host.TokenTable(f"{out_dir}/embeddings.bf16")
    tok = host.Tokenizer(f"{out_dir}/tokenizer.json")
    text = open(f"{out_dir}/LICENSE").read()
    doc = []
    while len(doc) < 8192:
        doc += tok.ids(text, prefix="")[1:-1]
    doc = doc[:8192]
    rng = np.random.default_rng(20261114)
    R = rng.standard_normal((512, DIGEST_DIMS)).astype(np.float32) / np.sqrt(512)
    blob = bytearray()

    def put(a):
        a = np.ascontiguousarray(a, "<f4")
        off = len(blob)
        blob.extend(a.tobytes())
        return {"offset": off, "shape": list(a.shape)}

    tests = []

    def run_torch(mod, feed):
        with torch.no_grad():
            return mod(*[torch.from_numpy(np.asarray(v, np.float32)) for v in feed.values()]).numpy()

    for name, kind, args, _, _ in build.function_specs():
        S = args[0]
        if kind in ("bucket", "winp"):
            seqs = bucket_sequences(doc, S) if kind == "bucket" else framed(doc, WINP_LENGTHS[S])
            rows = [tab.rows(q) for q in seqs]
            exp = np.stack([ref_embed(W, r) for r in rows])
            # sanity: the fp32 graph itself equals the reference
            got = run_torch(build.make_module(W, kind, args), host.pack(rows, S))
            assert min(host.cos(got[k], exp[k]) for k in range(len(seqs))) > 0.9999
            tests.append({"function": name, "kind": kind, "sequences": seqs, "expected": put(exp)})
            print(f"[selftest] {name} lens={[len(q) for q in seqs]}", flush=True)
        else:
            n = WINL_ROWS[S]
            rows = tab.rows(doc[:n])
            exp = ref_embed(W, rows)
            got = run_torch(build.make_module(W, kind, args), host.long_feed(rows, S))[0]
            assert host.cos(got, exp) > 0.9999
            tests.append({"function": name, "kind": kind, "doc_rows": n, "expected": put(exp)})
            print(f"[selftest] {name} doc_rows={n}", flush=True)
    # vision: fp32 HF vision tower + embedder on the procedural image
    from transformers import EmbeddingGemma2Model
    hfm = EmbeddingGemma2Model.from_pretrained(hf_dir, dtype=torch.float32, use_safetensors=True, local_files_only=True).eval()
    for B, (gx, gy) in VISION_GRIDS.items():
        pv, pos = pattern(gx, gy, B)
        with torch.no_grad():
            f = hfm.get_image_features(pixel_values=torch.from_numpy(pv)[None],
                                       image_position_ids=torch.from_numpy(pos)[None]).pooler_output[0].numpy()
        ntok = gx * gy // 9
        assert f.shape == (ntok, 512), f.shape
        tests.append({"function": f"vision_{B}", "kind": "vision", "package": "EmbeddingGemma2Vision",
                      "grid": [gx, gy], "budget": B, "valid_tokens": ntok, "expected": put(f @ R),
                      "expected_full_l2": float(np.linalg.norm(f))})
        print(f"[selftest] vision_{B} tokens={ntok}", flush=True)
    meta = {
        "format": "aneembed-selftest/1",
        "bin": "selftest.bin", "dtype": "float32", "byte_order": "little",
        "digest": put(R),
        "doc_ids_source": "LICENSE (Apache License 2.0 text shipped with the package), tokenized without prefix, "
                          "<bos>/<eos> stripped, repeated, first 8192 ids",
        "doc_ids": doc,
        "tolerance": TOLERANCE,
        "procedure": __doc__.strip().split("\n\n", 1)[1],
        "tests": tests,
    }
    open(f"{out_dir}/selftest.bin", "wb").write(bytes(blob))
    json.dump(meta, open(f"{out_dir}/selftest.json", "w"), indent=1)
    print(f"[selftest] {len(tests)} tests, selftest.bin {len(blob)} bytes", flush=True)


TOLERANCE = {   # min cosine / max relative error; measured values are >= 0.9999 (see README)
    "bucket": {"min_cos": 0.9990, "zero_rows_max_abs": 1e-3},
    "winp": {"min_cos": 0.9990, "zero_rows_max_abs": 1e-3},
    "winl": {"min_cos": 0.9990},
    "vision": {"min_cos": 0.9990, "max_rel_err": 0.05},
}


# ------------------------------------------------------------------------------------------- runner

def function_size(name):
    """embed_768 -> 768, winp1536_b512 -> 1536, winl8192_b512 -> 8192"""
    return int(name.split("_")[1]) if name.startswith("embed_") else int(name[4:].split("_")[0])


def metrics(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return host.cos(a, b), float(np.linalg.norm(a - b) / np.linalg.norm(b))


def run(dist, only=None, compute_units="ane", verbose=True):
    import coremltools as ct
    meta = json.load(open(f"{dist}/selftest.json"))
    blob = open(f"{dist}/selftest.bin", "rb").read()

    def get(e):
        return np.frombuffer(blob, "<f4", int(np.prod(e["shape"])), e["offset"]).reshape(e["shape"])

    R = get(meta["digest"])
    tab = host.TokenTable(f"{dist}/embeddings.bf16")
    vis = host.Vision(f"{dist}/position_embeddings.f16")
    doc = meta["doc_ids"]
    tol = meta["tolerance"]
    cu = ct.ComputeUnit.CPU_ONLY if compute_units == "cpu" else ct.ComputeUnit.CPU_AND_NE
    results, ok_all = [], True
    for t in meta["tests"]:
        fn, kind = t["function"], t["kind"]
        if only and fn not in only:
            continue
        pkg = t.get("package", "EmbeddingGemma2Text")
        m = ct.models.CompiledMLModel(f"{dist}/{pkg}.mlmodelc", compute_units=cu, function_name=fn)
        exp = get(t["expected"])
        r = {"function": fn}
        if kind in ("bucket", "winp"):
            S = function_size(fn)
            out = m.predict(host.pack([tab.rows(q) for q in t["sequences"]], S))["embedding"]
            k = len(t["sequences"])
            r["min_cos"] = min(host.cos(out[i], exp[i]) for i in range(k))
            r["zero_rows_max_abs"] = float(np.abs(out[k:]).max()) if k < len(out) else 0.0
            ok = r["min_cos"] >= tol[kind]["min_cos"] and r["zero_rows_max_abs"] <= tol[kind]["zero_rows_max_abs"]
        elif kind == "winl":
            S = function_size(fn)
            e = m.predict(host.long_feed(tab.rows(doc[:t["doc_rows"]]), S))["embedding"]
            r["min_cos"] = host.cos(e, exp)
            ok = r["min_cos"] >= tol[kind]["min_cos"]
        else:
            B = t["budget"]
            pv, pos = pattern(*t["grid"], B)
            feed, ntok = vis.feed(pv, pos, B)
            y = m.predict(feed)["image_tokens"][0, :ntok].astype(np.float32)
            r["min_cos"], r["rel_err"] = metrics(y @ R, exp)
            ok = r["min_cos"] >= tol[kind]["min_cos"] and r["rel_err"] <= tol[kind]["max_rel_err"]
        r["pass"] = bool(ok)
        ok_all &= ok
        results.append(r)
        if verbose:
            print(json.dumps(r), flush=True)
        del m
    return ok_all, results


PPLX_FORMAT = "aneembed-pplx-selftest/1"


def generate_pplx(hf_dir, out_dir):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from pplx_build import function_specs
    from pplx_reference import means, tokenizer

    text = (
        "A small coastal town mapped its bus routes, then asked residents which stops "
        "they used on weekday mornings and where they needed better connections.",
        "서울 도서관은 주말에 어린이와 가족을 위한 독서 모임을 열고 새 책을 소개합니다.",
    )
    tok = tokenizer(hf_dir)
    sequences = [tok.encode(t, add_special_tokens=False).ids[:32] for t in text]
    references = list(means(hf_dir, sequences))
    if any(v.shape != (1024,) or not np.isfinite(v).all() for v in references):
        raise ValueError("invalid pplx fp32 canary mean")
    cases, data = [], bytearray()
    direct = []
    for spec in function_specs():
        role, name = spec["role"], spec["function"]
        if role in ("pre", "post") or (role == "fused_part" and spec["part"] != 0):
            continue
        # Swift creates q=k=0, v[first]=1, v[last]=3 and opens only those two keys.
        # The expected output of the weightless attention function is 2 at every coordinate.
        if role == "attention":
            direct.append({"kind": "attention", "function": name})
        count = 2 if role == "packed" else 1
        cases.append({"kind": "route", "function": name, "sequences": sequences[:count], "offset": len(data)})
        for ref in references[:count]:
            data.extend(ref.astype("<f4", copy=False).tobytes())
    out = Path(out_dir)
    with TemporaryDirectory(prefix="_tmp_", dir=out) as tmp:
        stage = Path(tmp)
        (stage / "selftest.bin").write_bytes(data)
        (stage / "selftest.json").write_text(
            json.dumps({"format": PPLX_FORMAT, "cases": cases + direct}, indent=1) + "\n",
            encoding="utf-8",
        )
        for name in ("selftest.bin", "selftest.json"):
            (stage / name).replace(out / name)
    print(f"[selftest] {len(cases)} routes, {len(direct)} direct attention cases", flush=True)


def run_pplx(dist, binary, max_tokens):
    import subprocess
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from pplx_verify import swift_binary, swift_environment

    with TemporaryDirectory(prefix="pplx-warm-") as data_dir:
        result = subprocess.run(
            [swift_binary(binary), "models", "warm", "--model-dir", str(Path(dist).resolve()),
             "--data-dir", data_dir, "--max-tokens", str(max_tokens), "--format", "json"],
            env=swift_environment(), check=False, stdout=subprocess.PIPE, text=True,
        )
        print(result.stdout, end="")
        if result.returncode:
            return result.returncode
        report = json.loads(result.stdout)
        if report.get("max_tokens") != max_tokens or not report.get("selftest"):
            print(f"SELFTEST FAIL: expected a warm self-test at {max_tokens} tokens", file=sys.stderr)
            return 1
        return 0


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run",))
    ap.add_argument("dist")
    ap.add_argument("--functions", help="comma-separated functions to run (EmbeddingGemma 2)")
    ap.add_argument("--cpu", action="store_true", help="run the EmbeddingGemma 2 CPU control")
    ap.add_argument("--binary", help="anerun Swift executable (pplx)")
    ap.add_argument("--max-tokens", type=int, help="warm through this token cap (pplx; default 32768)")
    args = ap.parse_args()
    with open(os.path.join(args.dist, "selftest.json"), encoding="utf-8") as f:
        is_pplx = json.load(f).get("format") == PPLX_FORMAT
    if is_pplx:
        from pplx_verify import MAX_TOKENS

        if args.functions is not None or args.cpu:
            ap.error("pplx self-test uses the Swift host; choose a --max-tokens cap")
        cap = MAX_TOKENS if args.max_tokens is None else args.max_tokens
        if not 1 <= cap <= MAX_TOKENS:
            ap.error(f"--max-tokens must be between 1 and {MAX_TOKENS}")
        return run_pplx(args.dist, args.binary, cap)
    if args.binary is not None or args.max_tokens is not None:
        ap.error("--binary and --max-tokens apply to pplx")
    only = set(args.functions.split(",")) if args.functions is not None else None
    ok, res = run(args.dist, only, "cpu" if args.cpu else "ane")
    print(f"SELFTEST {'PASS' if ok else 'FAIL'} {sum(r['pass'] for r in res)}/{len(res)} "
          f"min_cos={min(r['min_cos'] for r in res):.6f} max_rel_err={max(r.get('rel_err', 0) for r in res):.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
