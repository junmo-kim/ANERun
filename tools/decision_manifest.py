#!/usr/bin/env python3
"""Index a clean, qualified decision bundle before uploading its contents to an HF revision root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--portable", action="store_true")
    parser.add_argument("--architecture", action="append", default=[])
    args = parser.parse_args()
    bundle = args.bundle
    inventory = bundle / "decision-manifest.json"
    files = []
    empty_directories = []
    seen = set()
    for path in sorted(bundle.rglob("*")):
        relative = path.relative_to(bundle).as_posix()
        folded = relative.casefold()
        if path.is_symlink() or folded in seen:
            parser.error(f"symlink or case collision: {relative}")
        seen.add(folded)
        if path == inventory and path.is_file():
            continue
        if (any(part.startswith(".") for part in Path(relative).parts)
                or "\\" in relative or any(ord(char) < 32 or ord(char) == 127 for char in relative)
                or relative.split("/")[0].casefold() in {
                    inventory.name, "manifest.json"
                }):
            parser.error(f"unexpected entry in packed bundle: {relative}")
        if path.is_dir():
            if not any(path.iterdir()):
                empty_directories.append(relative)
            continue
        if not path.is_file():
            parser.error(f"expected regular file: {relative}")
        files.append(dict(path=relative, size=path.stat().st_size, sha256=digest(path)))
    descriptor = json.loads((bundle / "model.json").read_bytes())
    architecture = descriptor.get("architecture")
    if args.portable:
        if architecture is not None or not args.architecture or any(
                not a.isascii() or not a.isalnum() for a in args.architecture):
            parser.error("portable model.json omits architecture; supply each qualified --architecture")
        if (bundle / "checkpoint").exists() or any(p.suffix == ".aimodelc" for p in bundle.iterdir()):
            parser.error("portable bundle requires final-form sidecars and .aimodel sources")
        manifest = dict(format="portable-v1", model=args.model, architectures=sorted(set(args.architecture)), files=files)
    elif args.architecture:
        parser.error("--architecture requires --portable")
    elif not isinstance(architecture, str) or not architecture:
        parser.error("packed model.json requires architecture")
    else:
        manifest = dict(model=args.model, architecture=architecture, files=files)
    if empty_directories:
        manifest["directories"] = empty_directories
    data = (json.dumps(manifest,
                       sort_keys=True, indent=2) + "\n").encode()
    if len(data) > 4 << 20:
        parser.error("inventory exceeds downloader limit (4 MiB)")
    with tempfile.NamedTemporaryFile(dir=bundle, delete=False) as output:
        output.write(data)
    os.replace(output.name, inventory)
    print(hashlib.sha256(data).hexdigest(), inventory)


if __name__ == "__main__":
    main()
