"""Negative control for the self-test: convert functions with the repeat_interleave K/V attention layout (known to
miscompile on the ANE for the global layers, which have one K/V head) and run their self-tests from selftest.json,
on the CPU and on CPU+ANE. The self-test is useful only if it fails winp2048_b512 on the ANE. The global layers of the
long (winl) functions use eg2.blocked_attention, which this script does not replace, so only winp2048_b512 still
puts the faulty layout into the global layers; the winl rebuilds change their sliding layers only. Existing
neg_*.mlmodelc in work_dir are reused, so use a fresh work_dir after the code changes.
Usage: selftest_negative.py <hf_dir> <dist> <work_dir> [function ...]   (default: winp2048_b512 winl4096_b512
winl8192_b512). Prints one NEG line per function and compute unit."""
import json
import os
import subprocess
import sys

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build  # noqa: E402
import eg2  # noqa: E402
import host  # noqa: E402
import selftest  # noqa: E402


def attend_repeat(q, k, v, bias, kvh):
    g = 4 // kvh
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + bias, dim=-1), v)


hf, dist, work = sys.argv[1:4]
names = sys.argv[4:] or ["winp2048_b512", "winl4096_b512", "winl8192_b512"]
os.makedirs(work, exist_ok=True)
eg2.attend_fold = attend_repeat          # band_attention / full_attention look it up at trace time
torch.set_num_threads(8)
W = eg2.Weights(hf)
spec = {s[0]: s for s in build.function_specs()}
meta = json.load(open(f"{dist}/selftest.json"))
blob = open(f"{dist}/selftest.bin", "rb").read()
get = lambda e: np.frombuffer(blob, "<f4", int(np.prod(e["shape"])), e["offset"]).reshape(e["shape"])  # noqa: E731
tab = host.TokenTable(f"{dist}/embeddings.bf16")
tests = {t["function"]: t for t in meta["tests"]}
for name in names:
    p = f"{work}/neg_{name}.mlpackage"
    if not os.path.exists(f"{work}/neg_{name}.mlmodelc"):
        build.convert_one(W, spec[name], p)
        subprocess.run(["xcrun", "coremlcompiler", "compile", p, work], check=True, capture_output=True)
    t = tests[name]
    S = selftest.function_size(name)
    if t["kind"] == "winl":
        feed, rows = host.long_feed(tab.rows(meta["doc_ids"][:t["doc_rows"]]), S), 1
    else:
        feed, rows = host.pack([tab.rows(q) for q in t["sequences"]], S), len(t["sequences"])
    exp = get(t["expected"]).reshape(rows, -1)
    tol = meta["tolerance"][t["kind"]]["min_cos"]
    for cu_name, cu in (("cpu", ct.ComputeUnit.CPU_ONLY), ("ane", ct.ComputeUnit.CPU_AND_NE)):
        m = ct.models.CompiledMLModel(f"{work}/neg_{name}.mlmodelc", compute_units=cu)
        out = np.asarray(m.predict(feed)["embedding"]).reshape(-1, 768)
        c = min(host.cos(out[k], exp[k]) for k in range(rows))
        print(f"NEG repeat-layout {name} {cu_name}: min_cos={c:.6f} pass={c >= tol}", flush=True)
        del m
