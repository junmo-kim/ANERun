"""Assemble portable source bytes matching a prepared GLiDE-9B qualification manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from prepare import bundle_child, glide9_assets, tokenize_tables


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assemble(prepared, backbone, operator_dirs, output):
    manifest = json.loads((prepared / "model.json").read_bytes())
    baseline = json.loads((backbone / "model.json").read_bytes())
    if (manifest["source"] != baseline["source"] or manifest["source"] != {
            "repo": "fastino/GLiDE-9B", "revision": "ab54df5acba0eebf7df45560ffdaacd64e32df3b"}):
        raise ValueError("Expected pinned GLiDE-9B")
    arch = manifest["architecture"]
    assets = manifest["assets"] + glide9_assets(manifest)
    if len(assets) != 96 or len({x["asset"] for x in assets}) != len(assets):
        raise ValueError("Incomplete GLiDE-9B qualification manifest")
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        for item in assets:
            name = item["asset"]
            suffix = "." + arch + ".aimodelc"
            if not name.endswith(suffix):
                raise ValueError(f"Expected a prepared asset: {name}")
            source_name = name.removesuffix(suffix) + ".aimodel"
            for directory in [backbone, *operator_dirs]:
                asset = bundle_child(directory, source_name)
                if not asset.is_dir():
                    continue
                if (asset / "main.hash").read_bytes().hex() != item["source_hash"]:
                    continue
                if any(path.is_symlink() for path in asset.rglob("*")):
                    raise ValueError(f"Unexpected link in {source_name}")
                shutil.copytree(asset, stage / source_name)
                if digest(stage / source_name / "main.mlirb") != item["source_hash"]:
                    raise ValueError(f"Source bytes differ from the qualification manifest: {name}")
                item["asset"] = source_name
                break
            else:
                raise ValueError(f"Missing matching source for {name}")
        rotary = bundle_child(prepared, "rotary.fp16")
        if digest(rotary) != "17d78a433463890c4f14c3e356d944505eb75e05c34541e9698760de2d2ec899":
            raise ValueError("Unexpected GLiDE-9B rotary table")
        shutil.copyfile(rotary, stage / rotary.name)
        ck = stage / "checkpoint"
        ck.mkdir()
        for name in ("tokenizer.json", "config.json", "decision_config.json", "LICENSE", "NOTICE", "licenses"):
            source = bundle_child(backbone / "checkpoint", name)
            if source.is_dir():
                shutil.copytree(source, ck / name)
            else:
                shutil.copyfile(source, ck / name)
        tokens = Path(tmp) / "tokens"
        tokens.mkdir()
        tokenize_tables(ck / "tokenizer.json", tokens)
        for name in ("bpe.bin", "tokens.json"):
            if digest(tokens / name) != digest(bundle_child(prepared, name)):
                raise ValueError(f"Tokenizer differs from qualified bundle: {name}")
        for name in ("config.json", "decision_config.json"):
            if digest(ck / name) != digest(bundle_child(prepared, name)):
                raise ValueError(f"Configuration differs from qualified bundle: {name}")
        del manifest["architecture"]
        (stage / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("backbone", type=Path, help="Portable backbone with its original checkpoint metadata")
    parser.add_argument("output", type=Path)
    parser.add_argument("--operator-dir", action="append", type=Path, required=True,
                        help="Directory containing retained .aimodel sources (repeat as necessary)")
    args = parser.parse_args()
    assemble(args.prepared, args.backbone, args.operator_dir, args.output)
