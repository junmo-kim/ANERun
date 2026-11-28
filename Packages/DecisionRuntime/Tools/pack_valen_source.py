"""Assemble a source-only Valen bundle from matched backbone, tokenizer and device graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from prepare import bundle_child, tokenize_tables
from valen_identity import PINS, recipe_and_names


def pack(manifest_path, sources, operators, sidecars, output, identities=None):
    m = json.loads(manifest_path.read_bytes())
    if (m["source"] != dict(repo="Valen-Team/Valen-Decision-1.5-4B", revision="9b0cacc976d7fb72f252ee1adb1564f5f15e462c")
            or len(m["assets"]) != 32 or "architecture" in m):
        raise ValueError("Expected the pinned portable Valen backbone")
    if output.exists():
        raise FileExistsError(output)
    fields = ("role_mixer", "role_pool", "decision_head", "result_stats", "vision_position",
              "feature_scale", "embedding_lookup", "vision_fractions")
    index = operators / "operators.json"
    records = json.loads(index.read_bytes()) if index.is_file() else m
    ops = {field: records[field] for field in fields}
    extra = [ops[name] for name in ("role_pool", "role_mixer", "decision_head", "result_stats", "vision_position", "feature_scale")]
    extra.extend(ops["embedding_lookup"]["assets"])
    if len(extra) != 67:
        raise ValueError("Expected all 67 Valen device assets")
    if "vision_256" in records:
        ops["vision_256"] = records["vision_256"]
        extra.append(ops["vision_256"])
    vision = dict(asset=m["vision_asset"], source_hash=m["vision_source_hash"], functions=m["vision_functions"])
    origins = {}
    if identities is not None:
        pins = json.loads(PINS.read_bytes())["assets"]
        for row in [*m["assets"], vision]:
            stem = Path(row["asset"]).stem
            if stem not in pins:
                continue
            pin = pins[stem]
            if pin["original_source_sha256"] != row["source_hash"]:
                raise ValueError(f"Unexpected original source for {stem}")
            _, mapping = recipe_and_names(row, pin, m["source"]["revision"])
            row.update(source_hash=pin["alternate_source_sha256"], functions=list(mapping.values()))
            origins[row["asset"]] = [identities / stem]
        if len(origins) != len(pins):
            raise ValueError("Incomplete pinned Valen identity set")
        m.update(vision_source_hash=vision["source_hash"], vision_functions=vision["functions"])
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        for record, roots in [(a, origins.get(a["asset"], sources)) for a in [*m["assets"], vision]] + [(a, [operators]) for a in extra]:
            name = record["asset"]
            if not name.endswith(".aimodel"):
                raise ValueError(f"Expected a portable source for {name}")
            for root in roots:
                asset = bundle_child(root, name)
                if not asset.is_dir() or (asset/"main.hash").read_bytes().hex() != record["source_hash"]:
                    continue
                with (asset/"main.mlirb").open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != record["source_hash"]:
                        raise ValueError(f"Changed source payload in {name}")
                if any(p.is_symlink() for p in asset.rglob("*")) or (stage/name).exists():
                    raise ValueError(f"Duplicate or linked source in {name}")
                subprocess.run(["cp", "-cR", asset, stage/name], check=True)
                break
            else:
                raise ValueError(f"Missing matched source for {name}")
        if (sidecars/"tokenizer.json").is_file():
            tokenize_tables(bundle_child(sidecars, "tokenizer.json"), stage)
        else:
            for name in ("bpe.bin", "tokens.json"):
                subprocess.run(["cp", "-c", bundle_child(sidecars, name), stage/name], check=True)
        for name in ("LICENSE", "NOTICE", "config.json"):
            subprocess.run(["cp", "-c", bundle_child(sidecars, name), stage/name], check=True)
        subprocess.run(["cp", "-c", operators/"vision-fractions.fp16", stage/"vision-fractions.fp16"], check=True)
        subprocess.run(["swift", Path(__file__).parent/"elephant/rotary.swift", stage, "valen-4b"], check=True)
        m.update(ops, rotary_tables=dict(text="text-rotary.fp16", vision="vision-rotary.fp16"))
        (stage/"model.json").write_text(json.dumps(m, indent=2)+"\n")
        stage.rename(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("operators", type=Path)
    p.add_argument("sidecars", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source-dir", type=Path, action="append", required=True)
    p.add_argument("--identity-dir", type=Path, help="Directory prepared by valen_identity.py")
    args = p.parse_args()
    pack(args.manifest, args.source_dir, args.operators, args.sidecars, args.output, args.identity_dir)
