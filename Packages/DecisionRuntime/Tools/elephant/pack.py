"""Build the qualified Elephant h17d device profile from the audited device bundle."""
import argparse
import json
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT.parent
REPO = "baselabs/elephant-1"
REVISION = "b3f8a93462fb1051a9de031e94a7ff380e254072"
BANK, HIDDEN, LANES, VOCAB = 4096, 4096, 16, 248320
BANK_COUNT = (VOCAB + BANK - 1) // BANK
ASSET_FIELDS = ("result_stats", "decision_head", "vision_position", "feature_scale")
SIDECARS = ("NOTICE", "bpe.bin", "config.json", "decision_config.json", "tokens.json")
FLAGS = {"mps.fullyPlacedOnANE", "mps.noGPUActivity"}


def leaf(name, suffix=""):
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name or not name.endswith(suffix):
        raise ValueError(f"Expected product-relative filename, got {name!r}")
    return name


def retained(manifest):
    vision = dict(asset=manifest["vision_asset"], functions=manifest["vision_functions"],
                  source_hash=manifest["vision_source_hash"])
    return [*manifest["assets"], vision, *manifest["residual_rounding"],
            *(manifest[key] for key in ASSET_FIELDS)]


def assets(manifest):
    return [*retained(manifest), *manifest["embedding_lookup"]["assets"], manifest["embedding_unscale"]]


def validate(manifest, candidate=False, portable=False):
    source = manifest["source"]
    banks = manifest["embedding_lookup"]
    if source["repo"] != REPO or source["revision"] != REVISION or (not portable and manifest["architecture"] != "h17d"):
        raise ValueError("Expected the pinned Elephant-1 revision for h17d")
    if (banks["bank_size"], banks["lane_width"], banks["vocab_size"], len(banks["assets"])) != (BANK, LANES, VOCAB, BANK_COUNT):
        raise ValueError("Unexpected embedding bank ABI")
    rows = assets(manifest) if candidate else [*retained(manifest), *banks["assets"]]
    if len(manifest["assets"]) != 24 or manifest["stage_kinds"] != ["linear", "linear", "attention"] * 8:
        raise ValueError("Unexpected Elephant backbone schedule")
    functions = [(row, f"lookup{i}") for i, row in enumerate(banks["assets"])]
    functions.extend((manifest[k], logical) for k, logical in (
        ("decision_head", "readout"), ("vision_position", "visionpos"),
        ("feature_scale", "featurescale"), ("result_stats", "stats")))
    if candidate:
        functions.append((manifest["embedding_unscale"], "unscale"))
    for row, logical in functions:
        if len(row["functions"]) != 1 or not re.fullmatch(logical + r"(?:_[0-9a-f]{64})?", row["functions"][0]):
            raise ValueError(f"Unexpected device entry for {logical}")
    if [row["functions"] for row in manifest["residual_rounding"]] != [["relusc"], ["two"], ["finite"]]:
        raise ValueError("Unexpected residual rounding functions")
    names = [leaf(row["asset"], ".aimodel" if portable else ".aimodelc") for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate asset path")
    for row in rows:
        fingerprint = row["source_hash"]
        if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError(f"Invalid source hash for {row['asset']}")
        if not row["functions"] or any(not isinstance(n, str) or not n for n in row["functions"]):
            raise ValueError(f"Invalid function list for {row['asset']}")
    if set(manifest["rotary_tables"]) != {"text", "vision"}:
        raise ValueError("Unexpected rotary tables")
    for name in manifest["rotary_tables"].values():
        leaf(name, ".fp16")
    if candidate:
        leaf(manifest["vision_fractions"], ".fp16")
    return rows


def audit(folder, manifest):
    rows = validate(manifest, candidate=True)
    keys = set()
    for row in rows:
        asset = folder / row["asset"]
        source = json.loads((asset/"metadata.json").read_text())["sourceHash"].lower()
        key = (asset/"main.hash").read_bytes()
        if source != row["source_hash"] or len(key) != 32 or key in keys:
            raise ValueError(f"Unexpected asset identity for {asset.name}")
        keys.add(key)
        placement, = asset.rglob("manifest.plist")
        versions = plistlib.loads(placement.read_bytes())["Package Version"]
        entries = {}
        for version in versions.values():
            for module in version["Optimized Modules"].values():
                functions = module["Entry Function Attributes"]
                if not functions or any(any(flag not in flags for flag in FLAGS) for flags in functions.values()):
                    raise ValueError(f"Incomplete ANE placement for {asset.name}")
                entries.update(functions)
        if len(entries) != len(row["functions"]) or any(
                not any(n == f or n.startswith(f + "_") for n in entries) for f in row["functions"]):
            raise ValueError(f"Missing compiled entries for {asset.name}")
    return dict(assets=len(rows), functions=sum(len(row["functions"]) for row in rows))


def build(baseline, target):
    gain = 256
    sys.path.insert(0, str(TOOLS))
    from prepare import compile_stage
    from elephant.export_embeddings import export

    manifest = json.loads((baseline/"model.json").read_text())
    validate(manifest)
    manifest["vision_fractions"] = "vision-fractions.fp16"
    for name, size in [(manifest["vision_fractions"], 511*512),
                       (manifest["rotary_tables"]["text"], 2048*32*4),
                       (manifest["rotary_tables"]["vision"], 2048*18*4)]:
        if (baseline/name).stat().st_size != size:
            raise ValueError(f"Incorrect byte count for {name}")
    if target.exists():
        raise FileExistsError(target)
    staged = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    product = staged/"product"
    product.mkdir()
    source = staged/"source"
    source.mkdir()
    reused = retained(manifest)
    reused = [row for row in reused if row != manifest["decision_head"]]
    notices = [name for name in ("LICENSE", "licenses") if (baseline/name).exists()]
    for name in [*SIDECARS, *notices, *manifest["rotary_tables"].values(), manifest["vision_fractions"],
                 *(row["asset"] for row in reused)]:
        src = baseline/name
        if src.is_symlink() or (src.is_dir() and any(p.is_symlink() for p in src.rglob("*"))):
            raise ValueError(f"Cannot package symlinks in {name}")
        subprocess.run(["cp", "-cR", src, product/name], check=True)

    def compile_export(name, graph):
        digest, entry = export(source, name, graph, gain)
        output = product/f"{name}-gain{gain}.h17d.aimodelc"
        compile_stage(source/f"{name}.aimodel", output, "h17d", digest, functions=[entry])
        return dict(asset=output.name, source_hash=digest, functions=[entry])

    from elephant.export_embeddings import graphs
    banks = []
    for name, graph in graphs(baseline/"model.safetensors", gain):
        row = compile_export(name, graph)
        if name == "unscale":
            manifest["embedding_unscale"] = row
        else:
            banks.append(row)
        print("compiled", name, flush=True)
    manifest["embedding_lookup"]["assets"] = banks
    from elephant.tail_stable_head import export_head
    head = export_head(baseline/"model.safetensors", source)
    source_path = source/head["asset"]
    compiled = product/"readout-tail-stable.h17d.aimodelc"
    compile_stage(source_path, compiled, "h17d", head["source_hash"], functions=head["functions"])
    manifest["decision_head"] = dict(head, asset=compiled.name)
    (product/"model.json").write_text(json.dumps(manifest, indent=2)+"\n")
    inventory = audit(product, manifest)
    product.rename(target)
    (staged/"inventory.json").write_text(json.dumps(inventory, indent=2)+"\n")
    print(json.dumps(dict(product=str(target), source=str(source), gain=gain, **inventory)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("target", type=Path, nargs="?")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        manifest_path = args.baseline if args.baseline.is_file() else args.baseline/"model.json"
        manifest = json.loads(manifest_path.read_text())
        print(len(validate(manifest, candidate="embedding_unscale" in manifest)))
    elif args.target is None:
        parser.error("target is required for a build")
    else:
        build(args.baseline.resolve(), args.target.absolute())


if __name__ == "__main__":
    main()
