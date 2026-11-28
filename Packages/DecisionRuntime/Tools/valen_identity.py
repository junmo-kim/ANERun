"""Reproduce the verified, symbol-only Valen Core AI source variants."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

PINS = Path(__file__).with_name("valen_identity_pins.json")
RESOURCE_DEFINITION = re.compile(rb'^\s*(resource_\d+): ("0x[0-9A-F]+"),?\s*$')
RESOURCE_REFERENCE = re.compile(rb"dense_resource<(resource_\d+)>")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_source(path, expected):
    actual = digest(path / "main.mlirb")
    if actual != expected or (path / "main.hash").read_bytes().hex() != expected:
        raise ValueError(f"Unexpected source hash for {path}. Expected {expected}, got {actual}.")


def canonical_ir(program):
    module = program._module._mlir_module
    definitions, ir = {}, []
    with tempfile.TemporaryFile() as assembly:
        module.operation.print(file=assembly, binary=True, enable_debug_info=False,
                               print_generic_op_form=True)
        assembly.seek(0)
        for line in assembly:
            match = RESOURCE_DEFINITION.match(line)
            if match:
                name, payload = match.groups()
                definitions[name] = hashlib.sha256(payload).hexdigest().encode()
            else:
                ir.append(line)

    referenced = set()

    def substitute(match):
        name = match[1]
        referenced.add(name)
        return b"dense_resource<sha256_" + definitions[name] + b">"

    graph = hashlib.sha256()
    for line in ir:
        graph.update(RESOURCE_REFERENCE.sub(substitute, line))
    if not definitions or referenced != definitions.keys():
        raise ValueError("Unexpected Core AI dense-resource layout")
    for payload_hash in sorted(definitions.values()):
        graph.update(payload_hash)
    return graph.hexdigest(), len(definitions)


def public_entries(program):
    module = program._module._mlir_module
    return sorted(op.attributes["sym_name"].value for op in module.body.operations
                  if op.operation.name == "coreai.graph" and "priv" not in op.attributes)


def rename_entries(program, mapping):
    from coreai._compiler._mlir_libs._mlir.ir import SymbolTable

    module = program._module._mlir_module
    if public_entries(program) != sorted(mapping):
        raise ValueError("Core AI source entry points differ from the manifest")
    with module.context:
        for old, new in mapping.items():
            graph = program._get_graph(old)
            SymbolTable.replace_all_symbol_uses(old, new, module.operation)
            SymbolTable.set_symbol_name(graph, new)
    if public_entries(program) != sorted(mapping.values()):
        raise ValueError("Core AI public entry name did not change as requested")


def recipe_and_names(item, pins, revision):
    recipe = {
        "version": 1, "recipe": "rename Core AI public graph symbol only",
        "profile": "valen-4b", "publisher_revision": revision,
        "canonical_ir_sha256": pins["canonical_ir_sha256"],
        "original_source_sha256": pins["original_source_sha256"], "minimum_os": "27.0",
    }
    names = {}
    # The h16g compiler's vision symbol limit requires the shorter pinned identity.
    hex_length = pins.get("identity_hex_length", 64)
    for entry in item["functions"]:
        payload = json.dumps({**recipe, "entry": entry}, sort_keys=True, separators=(",", ":")).encode()
        suffix = hashlib.sha256(payload).hexdigest()[:hex_length]
        names[entry] = f"{entry}_{suffix}"
    return recipe, names


def generate(original, output, mapping, expected_ir, alternate_hash):
    from coreai.authoring.asset import AIModelAsset

    program = AIModelAsset.load(original).program
    if canonical_ir(program) != expected_ir:
        raise ValueError(f"Pinned IR or tensor payload mismatch for {original}")
    rename_entries(program, mapping)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / output.name
        stage.mkdir()
        program._save_bytecode(stage / "main.mlirb")
        actual_hash = digest(stage / "main.mlirb")
        if actual_hash != alternate_hash:
            raise ValueError(f"Serialized source hash differs from the verified alternate: {actual_hash}")
        (stage / "main.hash").write_bytes(bytes.fromhex(actual_hash))
        shutil.copyfile(original / "metadata.json", stage / "metadata.json")
        restored = AIModelAsset.load(stage).program
        rename_entries(restored, {new: old for old, new in mapping.items()})
        if canonical_ir(restored) != expected_ir:
            raise ValueError(f"Round-trip IR or tensor payload mismatch for {original}")
        stage.rename(output)


def run(source, destination, selection, verify_only):
    pins = json.loads(PINS.read_text())
    manifest_path = source / "model.json"
    if digest(manifest_path) != pins["source_manifest_sha256"]:
        raise ValueError(f"Unrecognized pinned Valen source manifest: {manifest_path}")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination trees must be disjoint")
    manifest = json.loads(manifest_path.read_text())
    items = {Path(row["asset"]).stem: row for row in manifest["assets"]}
    items["vision"] = dict(asset=manifest["vision_asset"], functions=manifest["vision_functions"],
                           source_hash=manifest["vision_source_hash"])
    for key in selection or pins["assets"]:
        item, pin = items[key], pins["assets"][key]
        original = source / item["asset"]
        output = destination / key / item["asset"]
        original_hash, alternate_hash = pin["original_source_sha256"], pin["alternate_source_sha256"]
        if item["source_hash"] != original_hash:
            raise ValueError(f"Pinned source identity differs for {key}")
        verify_source(original, original_hash)
        recipe, names = recipe_and_names(item, pin, pins["publisher_revision"])
        if output.exists() or verify_only:
            verify_source(output, alternate_hash)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            generate(original, output, names, (pin["canonical_ir_sha256"], pin["dense_tensor_resources"]),
                     alternate_hash)
        record = {
            "asset": item["asset"], "functions": list(names.values()), "source_hash": alternate_hash,
            "source_path": str(output), "original_source_path": str(original), "entry_mapping": names,
            "recipe": recipe, "dense_tensor_resources": pin["dense_tensor_resources"],
        }
        if not verify_only:
            report = output.parent / "item.json"
            content = json.dumps(record, indent=2) + "\n"
            if report.exists() and report.read_text() != content:
                raise ValueError(f"Refusing to overwrite a different record: {report}")
            if not report.exists():
                report.write_text(content)
        print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Pinned portable Valen source bundle")
    parser.add_argument("--destination", required=True, type=Path, help="Separate output directory")
    parser.add_argument("--asset", action="append", choices=json.loads(PINS.read_text())["assets"],
                        help="Select a stage or vision. Default: all pinned alternates.")
    parser.add_argument("--verify-only", action="store_true", help="Check existing alternate bytes without writing")
    args = parser.parse_args()
    run(args.source_root.resolve(), args.destination.resolve(), args.asset, args.verify_only)


if __name__ == "__main__":
    main()
