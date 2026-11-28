"""Exercise publication of an AOT stage after the compiler returns a placement report."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

from prepare import PartialPlacement, build, compile_stage


class CompilationTests(unittest.TestCase):
    def compile(self, reports):
        with TemporaryDirectory() as directory:
            dest = Path(directory) / "out.aimodelc"
            attempts = []
            def compiler(args, check):
                path = Path(args[-1])
                path.mkdir()
                attempts.append(path)
            with patch("prepare.subprocess.run", side_effect=compiler), patch("prepare.verified", side_effect=reports):
                try:
                    compile_stage(Path(directory) / "in.aimodel", dest, "h17d", "digest", max_attempts=3)
                finally:
                    self.assertEqual(len(attempts), len(set(attempts)))
                    self.assertTrue(all(not p.exists() for p in attempts))
                    self.assertEqual(dest.exists(), reports[-1] is None)
            return len(attempts)

    def test_retries_only_partial_placement_and_publishes_verified_stage(self):
        self.assertEqual(self.compile([PartialPlacement(), None]), 2)
        with self.assertRaisesRegex(PartialPlacement, "after 3 attempts"):
            self.compile([PartialPlacement()] * 3)
        with self.assertRaisesRegex(ValueError, "source hash"):
            self.compile([ValueError("source hash")])

    def test_valen_optional_vision_asset_is_compiled_and_relocated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(source=dict(repo="Valen-Team/Valen-Decision-1.5-4B"), assets=[],
                            embedding_lookup=dict(assets=[]), rotary_tables=dict(text="text", vision="vision"),
                            vision_fractions="fractions")
            for key in ("role_pool", "role_mixer", "decision_head", "result_stats", "vision_position", "feature_scale", "vision_256"):
                asset = root / (key + ".aimodel")
                asset.mkdir()
                (asset / "main.hash").write_bytes(bytes(32))
                manifest[key] = dict(asset=asset.name, functions=["p256" if key == "vision_256" else key],
                                     source_hash="00" * 32)
            for name, size in (("text", 4096*32*4), ("vision", 4096*16*4), ("fractions", 511*512)):
                (root / name).write_bytes(bytes(size))
            for tier in (True, False):
                if not tier:
                    del manifest["vision_256"]
                out = root / str(tier)
                out.mkdir()
                (root / "model.json").write_text(json.dumps(manifest))
                with patch("prepare.compile_stage") as compile_graph, patch("prepare.tokenize_tables"):
                    build(root, out, "h16g")
                result = json.loads((out / "model.json").read_text())
                compiled = [c.args for c in compile_graph.call_args_list if c.args[0].stem == "vision_256"]
                self.assertEqual(len(compiled), int(tier))
                self.assertEqual("vision_256" in result, tier)
                if tier:
                    self.assertEqual(result["vision_256"]["asset"], compiled[0][1].name)
                    self.assertEqual(compiled[0][1].name, "vision_256.h16g.aimodelc")
                    self.assertEqual(compiled[0][-1], ["p256"])
