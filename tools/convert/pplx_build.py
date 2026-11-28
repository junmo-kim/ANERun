"""Build the fixed PPLXPackageContract.swift package ABI from the pinned source."""
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from fetch import check, load_sources, sha256

MODEL = "pplx-embed-v1-0.6b"
QWEN = "qwen3-embedding-0.6b"
TEXT_PACKAGE = "PPLXText"
STAGES = ("convert", "merge", "compile", "assets", "selftest", "manifest")
MERGE_KEY = ".pplx-merge-key"
COMPILE_KEY = ".pplx-compile-key"
MERGE_INCOMPLETE = "incomplete"


def check_source(hf_dir, source_key="pplx"):
    source = load_sources()[source_key]
    for rel, digest in source["sha256"].items():
        if not check(os.path.join(hf_dir, rel), digest):
            raise SystemExit(f"expected {source['id']} at {source['revision']} in {hf_dir}")
    metadata = Path(hf_dir) / ".cache/huggingface/download/model.safetensors.metadata"
    if metadata.is_file() and metadata.read_text().splitlines()[:1] != [source["revision"]]:
        raise SystemExit(f"{metadata} does not record source revision {source['revision']}")
    return source


def tensor(name, shape, dtype="fp16"):
    return {"name": name, "shape": list(shape), "dtype": dtype}


def function_specs(*, causal=False):
    def spec(role, name, inputs, outputs, **route):
        package = TEXT_PACKAGE if not causal and role in ("packed", "fused", "fused_part") else name
        return dict(role=role, package=package, function=name, inputs=inputs, outputs=outputs, **route)

    for length in ((64,) if causal else (64, 512, 1024, 2048, 4096)):
        packed = length <= 1024
        slots = max(8, length // 32) if packed else 1
        role, prefix = ("packed", "embed") if packed else ("fused", "fused")
        yield spec(role, f"{prefix}_{length}", [
            tensor("inputs_embeds", (1, length, 1024)),
            tensor("bias", (1, 1, length if packed else 1, length)),
            tensor("pool", (slots, length)),
        ], [tensor("pooled", (slots, 1024), "fp32")], length=length)
    for part in range(0 if causal else 4):
        hidden = (1, 8192, 1024)
        inputs = [tensor("inputs_embeds" if part == 0 else "hidden", hidden), tensor("bias", (1, 1, 1, 8192))]
        if part == 3:
            inputs.append(tensor("pool", (1, 8192)))
        output = tensor("pooled", (1, 1024), "fp32") if part == 3 else tensor("hidden_out", hidden)
        yield spec("fused_part", f"fused_8192_part{part}", inputs, [output], length=8192, part=part)
    hidden, query = (1, 2048, 1024), (1, 2048, 2048)
    for layer in range(28):
        xs = tensor("xs", hidden)
        yield spec("pre", f"pre{layer}", [xs], [
            tensor("q", query), tensor("k", hidden), tensor("v", hidden),
        ], layer=layer)
        yield spec("post", f"post{layer}", [xs, tensor("o", query)], [
            tensor("a", hidden), tensor("d", hidden),
        ], layer=layer)
    query = (1, 16, 2048, 128)
    for length in ((2048, 8192, 32768) if causal else (16384, 32768)):
        keys = (1, 8, length, 128)
        yield spec("attention", f"attn_S{length}_Q2048", [
            tensor("q", query), tensor("k", keys), tensor("v", keys), tensor("bias", (1, 1, 1, length)),
        ], [tensor("o", query)], length=length)


def package_names(*, causal=False):
    specs = function_specs(causal=causal)
    return list(dict.fromkeys(spec["package"] for spec in specs))


def trace_key(source, specs):
    here = Path(__file__).resolve().parent
    provenance = {
        "source": source,
        "functions": specs,
        "scripts": {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                    for name in ("build.py", "pplx_build.py", "pplx.py", "manifest.py", "requirements.txt")},
        "dependencies": {name: version(name) for name in ("coremltools", "torch", "numpy", "safetensors", "protobuf")},
        "python": sys.version,
        "machine": platform.machine(),
    }
    return hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()


def select_compiler(graph_key):
    try:
        result = subprocess.run(["xcrun", "--find", "coremlcompiler"], check=True, capture_output=True, text=True)
        selected = Path(result.stdout.strip())
        if not selected.is_absolute():
            raise ValueError("xcrun did not return an absolute compiler path")
        compiler = selected.resolve(strict=True)
        xcode = next((parent for parent in selected.parents if parent.name == "Contents"), None)
        metadata = xcode / "version.plist" if xcode is not None else None
        provenance = {
            "graph": graph_key,
            "compiler": str(compiler),
            "compiler_sha256": sha256(compiler),
            "xcode_sha256": sha256(metadata) if metadata is not None and metadata.is_file() else None,
            "host": platform.version(),
        }
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot identify the selected coremlcompiler: {error}") from error
    return compiler, hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()


@contextmanager
def build_lock(path):
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_key(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, UnicodeError):
        return None


def write_key(staging, path, key):
    pending = staging / path.name
    pending.write_text(key + "\n", encoding="utf-8")
    pending.replace(path)


def module(weights, spec, *, causal=False):
    import pplx
    role = spec["role"]
    if role == "packed":
        return pplx.Bucket(weights, spec["length"], mode="softmax", stages=() if causal else pplx.PPLX_SCALES)
    if role == "fused":
        return pplx.Bucket(weights, spec["length"], qblock=512, mode="softmax")
    if role == "fused_part":
        start = spec["part"] * 7
        return pplx.Chunk(weights, spec["length"], start, start + 7, qblock=256, mode="mmsum")
    if role == "attention":
        return pplx.CausalAttention() if causal else pplx.Attention()
    layer = spec["layer"]
    if role == "pre":
        return pplx.Pre(weights.layers[layer])
    return pplx.Post(weights.layers[layer], 1.0 if causal else pplx.alphas(len(weights.layers))[layer])


def convert_one(weights, spec, path, *, causal=False):
    import coremltools as ct
    import numpy as np
    import torch
    from manifest import check_pplx_spec

    inputs = spec["inputs"]
    dtype = {"fp16": np.float16, "fp32": np.float32}
    with torch.no_grad():
        traced = torch.jit.trace(module(weights, spec, causal=causal).eval(),
                                 tuple(torch.zeros(t["shape"]) for t in inputs), check_trace=False)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name=t["name"], shape=t["shape"], dtype=dtype[t["dtype"]]) for t in inputs],
        outputs=[ct.TensorType(name=t["name"], dtype=dtype[t["dtype"]]) for t in spec["outputs"]],
        convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    check_pplx_spec(model.get_spec(), [spec], merged=False)
    model.save(str(path))


def replace_package(staged, target):
    if target.exists():
        shutil.rmtree(target)
    staged.rename(target)


def stage_convert(a, fn_dir, specs):
    import torch
    from pplx import Weights

    torch.set_num_threads(8)
    weights = None
    for spec in specs:
        name = spec["function"]
        path = fn_dir / f"{name}.mlpackage"
        if path.is_dir() and not a.force:
            continue
        print(f"[convert] starting {name}", flush=True)
        start = time.monotonic()
        if spec["role"] == "attention":
            weights = None
        elif weights is None:
            weights = Weights(a.hf_dir)
        with tempfile.TemporaryDirectory(prefix="_tmp_", dir=fn_dir) as tmp:
            staged = Path(tmp) / path.name
            convert_one(weights, spec, staged, causal=a.model == QWEN)
            replace_package(staged, path)
        print(f"[convert] {name} {time.monotonic() - start:.1f}s", flush=True)


def stage_merge(a, fn_dir, specs, key):
    out = Path(a.out)
    reusable = not a.force and read_key(out / MERGE_KEY) == key
    packages = {spec["package"] for spec in specs}
    pending = {name for name in packages if not reusable or not (out / f"{name}.mlpackage").is_dir()}
    if not pending:
        return
    for spec in specs:
        if spec["package"] in pending and not (fn_dir / f"{spec['function']}.mlpackage").is_dir():
            raise SystemExit(f"no matching PPLX trace for {spec['function']} in {fn_dir}; run --stages convert first")

    import coremltools as ct
    from manifest import check_pplx_spec

    text = [s for s in specs if s["package"] == TEXT_PACKAGE]
    path = out / f"{TEXT_PACKAGE}.mlpackage"
    with tempfile.TemporaryDirectory(prefix="_tmp_", dir=out) as tmp:
        stage = Path(tmp)
        write_key(stage, out / MERGE_KEY, MERGE_INCOMPLETE)
        (out / COMPILE_KEY).unlink(missing_ok=True)
        if TEXT_PACKAGE in pending:
            print(f"[merge] starting {path.name}", flush=True)
            desc = ct.utils.MultiFunctionDescriptor()
            for spec in text:
                name = spec["function"]
                desc.add_function(str(fn_dir / f"{name}.mlpackage"), "main", name)
            desc.default_function_name = "embed_64"
            staged = stage / path.name
            ct.utils.save_multifunction(desc, str(staged))
            check_pplx_spec(ct.utils.load_spec(str(staged)), text, merged=True)
            replace_package(staged, path)
            print(f"[merge] {path}", flush=True)
        for spec in specs:
            if spec["package"] == TEXT_PACKAGE or spec["package"] not in pending:
                continue
            target = out / f"{spec['package']}.mlpackage"
            staged = stage / target.name
            shutil.copytree(fn_dir / target.name, staged)
            replace_package(staged, target)
        write_key(stage, out / MERGE_KEY, key)


def compiled_complete(directory):
    required = ("model.mil", "coremldata.bin", "weights/weight.bin")
    return all(path.is_file() and path.stat().st_size > 0 for path in (directory / file for file in required))


def require_compiled(directory):
    if not compiled_complete(directory):
        raise SystemExit(f"incomplete compiled PPLX package {directory.name}; rerun --stages compile")


def stage_compile(a, compiler, key):
    out = Path(a.out)
    reusable = not a.force and read_key(out / COMPILE_KEY) == key
    names = package_names(causal=a.model == QWEN)
    pending = [name for name in names if not reusable or not compiled_complete(out / f"{name}.mlmodelc")]
    if not pending:
        return
    with tempfile.TemporaryDirectory(prefix="_tmp_", dir=out) as tmp:
        (out / COMPILE_KEY).unlink(missing_ok=True)
        for name in pending:
            target = out / f"{name}.mlmodelc"
            start = time.monotonic()
            print(f"[compile] starting {name}", flush=True)
            subprocess.run([str(compiler), "compile", str(out / f"{name}.mlpackage"), tmp], check=True)
            compiled = Path(tmp) / target.name
            require_compiled(compiled)
            replace_package(compiled, target)
            print(f"[compile] {name} {time.monotonic() - start:.1f}s", flush=True)
        write_key(Path(tmp), out / COMPILE_KEY, key)


def stage_assets(a):
    import numpy as np
    import torch
    from safetensors import safe_open

    causal = a.model == QWEN
    src, out = Path(a.hf_dir), Path(a.out)
    with tempfile.TemporaryDirectory(prefix="_tmp_", dir=out) as tmp:
        stage = Path(tmp)
        with safe_open(str(src / "model.safetensors"), framework="pt") as f:
            prefix = "model." if "model.norm.weight" in f.keys() else ""
            for name, key, shape in (
                ("embeddings.f32", "embed_tokens.weight", (151669 if causal else 151936, 1024)),
                ("final_norm.f32", "norm.weight", (1024,)),
            ):
                t = f.get_tensor(prefix + key)
                if t.dtype != (torch.bfloat16 if causal else torch.float32) or tuple(t.shape) != shape:
                    raise ValueError(f"{key} has an unexpected dtype or shape; expected shape {shape}")
                t.float().numpy().astype(np.dtype("<f4"), copy=False).tofile(stage / name)
                del t
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            shutil.copyfile(src / name, stage / name)
        license_file = "QWEN-LICENSE" if causal else "PPLX-LICENSE"
        attribution = f"{QWEN} by Qwen (Apache-2.0; see LICENSE)." if causal else f"{MODEL} by Perplexity AI (MIT; see LICENSE)."
        shutil.copyfile(Path(__file__).with_name(license_file), stage / "LICENSE")
        source = load_sources()["qwen" if causal else "pplx"]
        (stage / "NOTICE").write_text(
            f"{attribution}\n"
            f"Source: https://huggingface.co/{source['id']}/tree/{source['revision']}\n"
            "ANEEmbed converts the model into fp16 Core ML ML Programs with fused and decomposed attention.\n"
            "The float32 embedding table and final RMSNorm weights, tokenizer and configuration are retained.\n",
            encoding="utf-8",
        )
        for path in stage.iterdir():
            path.replace(out / path.name)
    print("[assets] wrote float32 table, final norm, tokenizer, config and license", flush=True)


def run(a):
    causal = a.model == QWEN
    source = check_source(a.hf_dir, "qwen" if causal else "pplx")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    requested = set(a.stages.split(","))
    with build_lock(out / ".pplx-output.lock"):
        merged = read_key(out / MERGE_KEY)
        has_graphs = merged is not None or any(path.suffix in (".mlpackage", ".mlmodelc") for path in out.iterdir())
        updates_assets = bool(requested & {"assets", "selftest"})
        needs_graphs = bool(requested & {"compile", "manifest"}) or (has_graphs and updates_assets)
        if needs_graphs or requested & {"convert", "merge"}:
            specs = list(function_specs(causal=causal))
            graph_key = trace_key(source, specs)
        if needs_graphs and "merge" not in requested:
            if merged == MERGE_INCOMPLETE:
                raise SystemExit("PPLX merge was interrupted; rerun --stages merge first")
            if merged != graph_key:
                raise SystemExit("PPLX graph provenance is missing or stale; run --stages convert,merge or use a fresh --out")

        if requested & {"convert", "merge"}:
            fn_dir = Path(a.work) / "fn" / graph_key
            fn_dir.mkdir(parents=True, exist_ok=True)
            with build_lock(fn_dir / ".lock"):
                if "convert" in requested:
                    stage_convert(a, fn_dir, specs)
                if "merge" in requested:
                    stage_merge(a, fn_dir, specs, graph_key)

        if "compile" in requested:
            compiler, compile_key = select_compiler(graph_key)
            stage_compile(a, compiler, compile_key)
        elif "manifest" in requested:
            recorded_key = read_key(out / COMPILE_KEY)
            compiled = [out / f"{name}.mlmodelc" for name in package_names(causal=causal)]
            if recorded_key is not None or any(path.exists() for path in compiled):
                _, compile_key = select_compiler(graph_key)
                if recorded_key != compile_key:
                    raise SystemExit("compiled PPLX packages are stale; rerun --stages compile before writing the manifest")
                for directory in compiled:
                    require_compiled(directory)
        if "assets" in requested:
            stage_assets(a)
        if "selftest" in requested:
            if causal:
                from qwen_reference import generate
            else:
                from selftest import generate_pplx as generate
            generate(a.hf_dir, out)
        if "manifest" in requested:
            from manifest import write_pplx
            write_pplx(out, source)
