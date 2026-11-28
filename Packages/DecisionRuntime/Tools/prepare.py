"""Build a self-contained ANE decision bundle for one target chip. Requires Xcode 27."""
import argparse
import json
from pathlib import Path
import plistlib
import shutil
import struct
import subprocess
import tempfile
import time


def tokenize_tables(source, target):
    tokenizer = json.loads(source.read_text())
    bpe = tokenizer["model"]
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    remap = dict(zip(visible, visible))
    remap.update((b, 256+i) for i, b in enumerate(n for n in range(256) if n not in remap))
    vocab = bpe["vocab"]
    with (target/"bpe.bin").open("wb") as out:
        out.write(struct.pack("<256I", *(vocab[chr(remap[b])] for b in range(256))))
        for merge in bpe["merges"]:
            a, b = merge.split(" ") if isinstance(merge, str) else merge
            out.write(struct.pack("<III", vocab[a], vocab[b], vocab[a+b]))
    split = tokenizer["pre_tokenizer"]["pretokenizers"][0]
    special = {t["content"]: t["id"] for t in tokenizer["added_tokens"]}
    (target/"tokens.json").write_text(json.dumps({"special": special, "pattern": split["pattern"]["Regex"]}))


class PartialPlacement(ValueError):
    pass


def verified(path, digest, expected_functions=None):
    if json.loads((path/"metadata.json").read_text())["sourceHash"].lower() != digest:
        raise ValueError(f"Unexpected source hash for {path.name}")
    manifest, = path.rglob("manifest.plist")
    with manifest.open("rb") as f:
        versions = plistlib.load(f)["Package Version"]
    if not versions:
        raise ValueError("No compiled package version")
    names = set()
    for version in versions.values():
        modules = version["Optimized Modules"]
        if not modules:
            raise ValueError("No compiled modules")
        for m in modules.values():
            entries = m["Entry Function Attributes"]
            names.update(entries)
            if not entries or not all("mps.fullyPlacedOnANE" in attrs for attrs in entries.values()):
                raise PartialPlacement(f"Partial ANE placement in {path.name}")

    if expected_functions is not None and (
            any(not any(name == f or name.startswith(f + "_") for name in names) for f in expected_functions)
            or any(not any(name == f or name.startswith(f + "_") for f in expected_functions) for name in names)):
        raise ValueError(f"Unexpected entry points in {path.name}")


DEFAULT_COMPILE_ATTEMPTS = 8


def compile_stage(asset, path, arch, digest, max_attempts=DEFAULT_COMPILE_ATTEMPTS, functions=None):
    # The compiler can produce a partial specialization with a zero exit status.
    # A fresh compilation of the same graph can produce complete ANE placement.
    for attempt in range(1, max_attempts + 1):
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            candidate = Path(temp).resolve() / "stage.aimodelc"
            subprocess.run(["xcrun", "coreai-build", "compile", asset.resolve(), "--architecture", arch,
                            "--preferred-compute", "neural-engine", "--platform", "macOS",
                            "--min-deployment-version", "27.0", "--output", candidate], check=True)
            try:
                verified(candidate, digest, functions)
            except PartialPlacement as exc:
                if attempt == max_attempts:
                    raise PartialPlacement(f"Partial ANE placement in {asset.name} after {max_attempts} attempts") from exc
                print(f"Retrying partial ANE compilation of {asset.name}", flush=True)
                continue
            candidate.rename(path)
            return


def normalize_schedule(manifest):
    if "execution" not in manifest and manifest["source"]["repo"] == "fastino/GLiDE-2B":
        if manifest["attention_mode"] != "noncausal_full_attention":
            raise ValueError("Unsupported attention mask")
        count = len(manifest["assets"])
        manifest.update(execution=[dict(stages=[i], passes=1) for i in range(count)],
                        stage_kinds=["attention"] * count, block_mask=False)


def glide9_assets(manifest):
    """Validate the device-only ABI before compiling or opening any operator path."""
    lookup = manifest.get("embedding_lookup") or {}
    banks = lookup.get("assets", [])
    if (lookup.get("bank_size"), lookup.get("lane_width"), lookup.get("vocab_size"), len(banks)) != (4096, 16, 248320, 61):
        raise ValueError("Missing or incompatible GLiDE-9B embedding banks")
    layout = [(f"lookup{i}", f"lookup{i}", item) for i, item in enumerate(banks)]
    for field, filename, function in (("decision_head", "head", "readout"),
                                      ("residual_blend", "blend2", "blend2"),
                                      ("result_stats", "readoutStats", "readoutStats")):
        layout.append((filename, function, manifest.get(field) or {}))
    for filename, function, asset in layout:
        source_name = filename + ".aimodel"
        compiled_name = f"{filename}.{manifest.get('architecture')}.aimodelc"
        if asset.get("asset") not in (source_name, compiled_name) or asset.get("functions") != [function]:
            raise ValueError(f"Invalid GLiDE-9B asset for {function}")
    return [asset for _, _, asset in layout]


def bundle_child(root, name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Expected a bundle-relative filename")
    path = root / name
    if path.is_symlink():
        raise ValueError(f"Unexpected symbolic link at {name}")
    return path


def build(source, out, arch, max_attempts=DEFAULT_COMPILE_ATTEMPTS):
    if not arch.isascii() or not arch.isalnum():
        raise ValueError("Expected a chip architecture identifier")
    manifest = json.loads(bundle_child(source, "model.json").read_text())
    normalize_schedule(manifest)
    if manifest.get("architecture", arch) != arch:
        raise ValueError("Compiled bundle is for a different chip")
    glide9 = manifest["source"]["repo"] == "fastino/GLiDE-9B"
    elephant = manifest["source"]["repo"] == "baselabs/elephant-1"
    valen = manifest["source"]["repo"] == "Valen-Team/Valen-Decision-1.5-4B"
    extras = glide9_assets(manifest) if glide9 else []
    if valen:
        extras = [*manifest["embedding_lookup"]["assets"], *(manifest[k] for k in
                  ("role_pool", "role_mixer", "decision_head", "result_stats", "vision_position", "feature_scale"))]
        if manifest.get("vision_256") is not None:
            extras.append(manifest["vision_256"])
    if elephant:
        from elephant.pack import validate
        validate(manifest, candidate=True, portable="architecture" not in manifest)
        extras = [*manifest["residual_rounding"], *manifest["embedding_lookup"]["assets"],
                  *(manifest[key] for key in ("decision_head", "vision_position", "feature_scale",
                                              "result_stats", "embedding_unscale"))]
    for item in manifest["assets"] + extras + ([dict(asset=manifest["vision_asset"],
            functions=manifest.get("vision_functions"), source_hash=manifest["vision_source_hash"])] if manifest.get("vision_asset") else []):
        name = item["asset"]
        asset = bundle_child(source, name)
        if asset.suffix == ".aimodel":
            path = out/f"{asset.stem}.{arch}.aimodelc"
            if (asset / "main.hash").read_bytes().hex() != item["source_hash"]:
                raise ValueError(f"Unexpected source hash for {name}")
            start = time.time()
            compile_stage(asset, path, arch, item["source_hash"], max_attempts, item["functions"] if glide9 or elephant or valen else None)
            print(f"compile {name} {start:.6f} {time.time():.6f}", flush=True)
        elif asset.suffix == ".aimodelc":
            path = out/name
            subprocess.run(["cp", "-cR", asset, path], check=True)
            verified(path, item["source_hash"], item["functions"] if glide9 or elephant or valen else None)
        else:
            raise ValueError(f"Unknown asset {name}")
        if name == manifest.get("vision_asset"):
            manifest["vision_asset"] = path.name
        item["asset"] = path.name
    if glide9:
        table = bundle_child(source, "rotary.fp16")
        if table.stat().st_size != manifest["context"] * manifest["rotary_dim"] * 2:
            raise ValueError("Invalid GLiDE-9B rotary table")
        shutil.copyfile(table, out / table.name)
    if elephant or valen:
        limit = 4096 if valen else 2048
        for name, size in [(manifest["rotary_tables"]["text"], limit * 32 * 4),
                           (manifest["rotary_tables"]["vision"], limit * (16 if valen else 18) * 4),
                           (manifest["vision_fractions"], 511 * 512)]:
            table = bundle_child(source, name)
            if table.stat().st_size != size:
                raise ValueError(f"Unexpected constant table size for {name}")
            shutil.copyfile(table, out / name)
    ck = bundle_child(source, "checkpoint")
    for name in ("model.safetensors", "config.json", "LICENSE", "NOTICE", "licenses",
                 "decision_config.json", "readout.safetensors"):
        if (glide9 or elephant or valen) and name in ("model.safetensors", "readout.safetensors"):
            continue
        if (ck/name).exists():
            subprocess.run(["cp", "-cR", ck/name, out/name], check=True)
    tokenize_tables(ck/"tokenizer.json", out)
    manifest["architecture"] = arch
    (out/"model.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--architecture", required=True)
    p.add_argument("--compile-attempts", type=int, default=DEFAULT_COMPILE_ATTEMPTS,
                   help="Maximum compilations per graph after partial ANE placement (default: %(default)s)")
    args = p.parse_args()
    if args.compile_attempts < 1:
        p.error("--compile-attempts must be positive")
    if args.output.exists():
        p.error("Choose a new output directory")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staging = Path(tmp)/"bundle"
        staging.mkdir()
        build(args.source, staging, args.architecture, args.compile_attempts)
        staging.rename(args.output)
