"""Exercise the complete Elephant portable-to-prepared manifest without an ANE."""
import json
from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest.mock import patch

from elephant.pack import assets, validate
from prepare import build, verified


def row(name, function=None):
    return dict(asset=name+".aimodel", source_hash="12"*32, functions=[function or name])


def manifest():
    return dict(source=dict(repo="baselabs/elephant-1", revision="b3f8a93462fb1051a9de031e94a7ff380e254072"),
                assets=[row(f"stage{i}", f"scan{i}") for i in range(24)],
                stage_kinds=["linear", "linear", "attention"]*8, execution=[],
                vision_asset="vision.aimodel", vision_functions=["p1024"], vision_source_hash="12"*32,
                residual_rounding=[row(n) for n in ("relusc", "two", "finite")],
                embedding_lookup=dict(bank_size=4096, lane_width=16, vocab_size=248320,
                                      assets=[row(f"lookup{i}") for i in range(61)]),
                result_stats=row("stats"), decision_head=row("readout", "readout_"+"ab"*32),
                embedding_unscale=row("unscale"), vision_position=row("visionpos"),
                feature_scale=row("featurescale"), vision_fractions="fractions.fp16",
                rotary_tables=dict(text="text.fp16", vision="vision.fp16"))


class ElephantDeviceTests(unittest.TestCase):
    def test_pack_rejects_reordered_banks_and_escaping_tables(self):
        m = manifest()
        banks = m["embedding_lookup"]["assets"]
        banks[2], banks[3] = banks[3], banks[2]
        with self.assertRaisesRegex(ValueError, "lookup2"):
            validate(m, candidate=True, portable=True)
        m = manifest()
        m["rotary_tables"]["vision"] = "../vision.fp16"
        with self.assertRaisesRegex(ValueError, "relative"):
            validate(m, candidate=True, portable=True)

    def test_portable_bundle_packs_all_94_assets_and_tables(self):
        m = manifest()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root/"out"
            out.mkdir()
            (root/"model.json").write_text(json.dumps(m))
            for item in assets(m):
                d = root/item["asset"]
                d.mkdir()
                (d/"main.hash").write_bytes(bytes.fromhex(item["source_hash"]))
            ck = root/"checkpoint"
            ck.mkdir()
            (ck/"model.safetensors").touch()
            for name, size in [("fractions.fp16", 511*512), ("text.fp16", 2048*32*4), ("vision.fp16", 2048*18*4)]:
                (root/name).write_bytes(bytes(size))
            def compile_one(source, target, arch, digest, attempts, functions):
                self.assertTrue(functions)
                target.mkdir()
            with patch("prepare.compile_stage", side_effect=compile_one) as compiler, patch("prepare.tokenize_tables"):
                build(root, out, "h17d")
            self.assertEqual(compiler.call_count, 94)
            prepared = json.loads((out/"model.json").read_text())
            self.assertTrue(all((out/r["asset"]).is_dir() for r in validate(prepared, candidate=True)))
            self.assertFalse((out/"model.safetensors").exists())
            self.assertEqual(len(list(out.glob("*.fp16"))), 3)

    def test_placement_checks_the_entire_bound_function_name(self):
        entry = "scan0_" + "ab"*32
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/"metadata.json").write_text(json.dumps({"sourceHash": "12"*32}))
            attrs = {entry+"_0": ["mps.fullyPlacedOnANE"]}
            module = {"Entry Function Attributes": attrs}
            data = {"Package Version": {"v": {"Optimized Modules": {"m": module}}}}
            (path/"manifest.plist").write_bytes(plistlib.dumps(data))
            verified(path, "12"*32, [entry])
            with self.assertRaisesRegex(ValueError, "entry points"):
                verified(path, "12"*32, ["scan0_"+"cd"*32])
