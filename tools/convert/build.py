#!/usr/bin/env python3
"""Build the ANEEmbed EmbeddingGemma 2 Core ML release package from the original HF weights.

    python -I build.py --hf-dir <google/embeddinggemma-2 snapshot> --out <dist>

    python -I build.py --model pplx-embed-v1-0.6b --hf-dir <pinned pplx snapshot> --out <dist>

fetch.py models downloads the requested model inputs at the pinned revisions (see sources.json and README.md).

Stages (default: all, in order; each is idempotent and skips finished work unless --force):
  convert   one single-function .mlpackage per function into <work>/fn/  (fp16 ML Program, macOS 15):
            embed_{64..1280} (full attention), winp{1536,2048}_b512 (windowed, packing),
            winl{3072,4096,6144,8192}_b512 (windowed, one long text per call)
  merge     multifunction merge -> <out>/EmbeddingGemma2Text.mlpackage (coremltools dedups identical weights)
  compile   xcrun coremlcompiler -> <out>/EmbeddingGemma2Text.mlmodelc
  assets    embeddings.bf16 (from model.safetensors), tokenizer.json, vision package + position table (original weights), LICENSE/NOTICE
  selftest  selftest.json + selftest.bin (fp32 torch expectations; see selftest.py)
  manifest  manifest.json (file sha256/sizes, signatures, host contract, token ids, provenance)

PPLX intermediates in <work>/fn are grouped by converter provenance. A merge-only run
uses the matching traces from an earlier convert stage.

No step uses the GPU; conversion and fp32 references run on the CPU. Requirements: requirements.txt.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fetch import load_sources, sha256

TEXT_PKG = "EmbeddingGemma2Text"
VISION_PKG = "EmbeddingGemma2Vision"
VISION_KEY = ".eg2-vision-key"


def function_specs():
    """(name, kind, args, inputs [(name, shape)], outputs [(name, dtype)])"""
    from eg2 import BAND, BUCKETS, WINDOW_LONG, WINDOW_PACK, k_for
    from host import function_name
    specs = []
    for S in BUCKETS:
        specs.append((function_name(S), "bucket", (S,),
                      [("inputs_embeds", (1, S, 512)), ("global_bias", (1, 1, S, S)), ("local_bias", (1, 1, S, S)),
                       ("pool", (k_for(S), S))], [("embedding", "fp32")]))
    for S in WINDOW_PACK:
        specs.append((function_name(S), "winp", (S,),
                      [("inputs_embeds", (1, S, 512)), ("band_bias", (1, 1, S, BAND)), ("global_bias", (1, 1, S, S)),
                       ("pool", (k_for(S), S))], [("embedding", "fp32")]))
    for S in WINDOW_LONG:
        specs.append((function_name(S, long=True), "winl", (S,),
                      [("inputs_embeds", (1, S, 512)), ("band_bias", (1, 1, S, BAND)), ("key_bias", (1, 1, 1, S)),
                       ("pool", (1, S))], [("embedding", "fp32")]))
    return specs


def make_module(W, kind, args):
    from eg2 import Bucket, Window
    return {"bucket": Bucket, "winp": Window, "winl": Window}[kind](W, *args).eval()


def convert_one(W, spec, path):
    import coremltools as ct
    import numpy as np
    import torch
    name, kind, args, ins, outs = spec
    mod = make_module(W, kind, args)
    ex = tuple(torch.zeros(*s) for _, s in ins)
    with torch.no_grad():
        tr = torch.jit.trace(mod, ex, check_trace=False)
    dt = {"fp32": np.float32, "fp16": np.float16}
    ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=s, dtype=np.float16) for n, s in ins],
                    outputs=[ct.TensorType(name=n, dtype=dt[d]) for n, d in outs],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE,
                    skip_model_load=True)
    ml.save(path)


def stage_convert(a):
    import torch
    from eg2 import Weights
    torch.set_num_threads(8)
    os.makedirs(f"{a.work}/fn", exist_ok=True)
    W = Weights(a.hf_dir)
    for spec in function_specs():
        p = f"{a.work}/fn/{spec[0]}.mlpackage"
        if os.path.exists(p) and not a.force:
            continue
        t = time.time()
        tmp = f"{a.work}/fn/_tmp_{spec[0]}.mlpackage"
        convert_one(W, spec, tmp)
        if os.path.exists(p):
            shutil.rmtree(p)
        os.replace(tmp, p)
        print(f"[convert] {spec[0]} {time.time() - t:.1f}s", flush=True)


def stage_merge(a):
    import coremltools as ct
    out = f"{a.out}/{TEXT_PKG}.mlpackage"
    if os.path.exists(out) and not a.force:
        return
    t = time.time()
    desc = ct.utils.MultiFunctionDescriptor()
    for spec in function_specs():
        desc.add_function(f"{a.work}/fn/{spec[0]}.mlpackage", src_function_name="main", target_function_name=spec[0])
    desc.default_function_name = "embed_512"
    tmp = f"{a.out}/_tmp_{TEXT_PKG}.mlpackage"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    ct.utils.save_multifunction(desc, tmp)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.replace(tmp, out)
    print(f"[merge] {out} {time.time() - t:.1f}s", flush=True)


def stage_compile(a):
    src = f"{a.out}/{TEXT_PKG}.mlpackage"
    dst = f"{a.out}/{TEXT_PKG}.mlmodelc"
    if os.path.exists(dst) and not a.force:
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    t = time.time()
    subprocess.run(["xcrun", "coremlcompiler", "compile", src, a.out], check=True, capture_output=True)
    print(f"[compile] {dst} {time.time() - t:.1f}s", flush=True)


def write_apache_license(path):
    """Apache License 2.0 full text (from the transformers wheel, which is Apache-2.0), header line stripped.
    selftest.py tokenizes this file, so its bytes are part of the self-test expectations."""
    import importlib.metadata as md
    src = next(f for f in md.files("transformers") if f.name == "LICENSE")
    text = src.read_text()
    text = text[text.index("Apache License"):]
    with open(path, "w") as f:
        f.write("                                 " + text)


def stage_assets(a):
    import importlib.metadata as md
    from eg2_vision import convert
    from pplx_build import compiled_complete, read_key, replace_package, select_compiler, write_key
    from safetensors import safe_open

    emb = f"{a.out}/embeddings.bf16"
    if not os.path.exists(emb) or a.force:
        t = safe_open(f"{a.hf_dir}/model.safetensors", "pt").get_tensor("language_model.embed_tokens.weight")
        import torch
        assert t.dtype == torch.bfloat16 and tuple(t.shape) == (262144, 512)
        t.contiguous().view(torch.int16).numpy().astype("<i2").tofile(emb + ".tmp")
        os.replace(emb + ".tmp", emb)
    shutil.copy2(f"{a.hf_dir}/tokenizer.json", f"{a.out}/tokenizer.json")

    out = Path(a.out)
    # A prebuilt vision package from an older release must never be silently
    # relabeled. Publish the marker last so an interrupted replacement retries.
    source = load_sources()["models"][0]["sha256"]["model.safetensors"]
    graph = [source, sha256(Path(HERE) / "eg2_vision.py")]
    graph.extend(md.version(p) for p in ("torch", "coremltools", "transformers", "numpy"))
    compiler, key = select_compiler(" ".join(graph))
    package = f"{VISION_PKG}.mlpackage"
    compiled = f"{VISION_PKG}.mlmodelc"
    marker = out / VISION_KEY
    reusable = (read_key(marker) == key and compiled_complete(out / compiled) and
                (out / package / "Manifest.json").is_file() and (out / "position_embeddings.f16").is_file())
    if a.force or not reusable:
        with tempfile.TemporaryDirectory(prefix="_tmp_vision_", dir=out) as tmp:
            stage = Path(tmp)
            convert(a.hf_dir, stage)
            subprocess.run([str(compiler), "compile", str(stage / package), str(stage)], check=True)
            if not compiled_complete(stage / compiled):
                raise RuntimeError(f"incomplete Core ML compiler output for {VISION_PKG}")
            write_key(stage, marker, "incomplete")
            for name in (package, compiled):
                replace_package(stage / name, out / name)
            (stage / "position_embeddings.f16").replace(out / "position_embeddings.f16")
            write_key(stage, marker, key)
    write_apache_license(f"{a.out}/LICENSE")
    shutil.copy2(os.path.join(HERE, "NOTICE"), f"{a.out}/NOTICE")
    print("[assets] done", flush=True)


def main():
    import pplx_build

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("embeddinggemma-2", pplx_build.MODEL, pplx_build.QWEN), default="embeddinggemma-2")
    ap.add_argument("--hf-dir", required=True, help="pinned weights and configuration for the selected model (fetch.py models)")
    ap.add_argument("--vision-dir", help="deprecated compatibility argument; vision is built from --hf-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="intermediate conversion cache (shared by sibling PPLX outputs)")
    ap.add_argument("--stages", default="convert,merge,compile,assets,selftest,manifest")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    stages = set(a.stages.split(","))
    if stages - set(pplx_build.STAGES):
        ap.error(f"expected stages from {', '.join(pplx_build.STAGES)}")
    if a.model in (pplx_build.MODEL, pplx_build.QWEN):
        if a.vision_dir is not None:
            ap.error("--vision-dir only applies to embeddinggemma-2")
        a.work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), "work-qwen" if a.model == pplx_build.QWEN else "work-pplx")
        pplx_build.run(a)
        return
    if a.vision_dir is not None:
        print("[assets] --vision-dir is deprecated; building from the raw Google weights", flush=True)
    a.work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), "work")
    os.makedirs(a.out, exist_ok=True)
    source = load_sources()["models"][0]
    required = set()
    if stages & {"convert", "assets", "selftest"}:
        required.update(("model.safetensors", "config.json"))
    if "assets" in stages:
        required.add("tokenizer.json")
    if "manifest" in stages:
        required.add("config_sentence_transformers.json")
    for rel in sorted(required):
        want = source["sha256"][rel]
        if sha256(Path(a.hf_dir) / rel) != want:
            raise SystemExit(f"{rel}: expected pinned SHA-256 {want}")
    print(f"[source] {source['id']} revision {source['revision']}", flush=True)
    with pplx_build.build_lock(Path(a.out) / ".eg2-output.lock"):
        build_eg2(a, source)


def build_eg2(a, source):
    stages = a.stages.split(",")
    if "convert" in stages:
        stage_convert(a)
    if "merge" in stages:
        stage_merge(a)
    if "compile" in stages:
        stage_compile(a)
    if "assets" in stages:
        stage_assets(a)
    if "selftest" in stages:
        import selftest
        selftest.generate(a.hf_dir, a.out)
    if "manifest" in stages:
        import manifest
        from pplx_build import read_key
        if read_key(Path(a.out) / VISION_KEY) == "incomplete":
            raise SystemExit("vision replacement was interrupted; rerun --stages assets")
        manifest.write(a, source["revision"])


if __name__ == "__main__":
    main()
