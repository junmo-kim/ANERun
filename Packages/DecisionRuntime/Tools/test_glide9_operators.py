import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare import build, bundle_child, glide9_assets


def asset(name, function=None):
    return dict(asset=name + ".aimodel", functions=[function or name], source_hash="12" * 32)


def manifest():
    return dict(source={"repo": "fastino/GLiDE-9B"}, assets=[asset("stage0", "scan0")],
                context=128, rotary_dim=64, decision_head=asset("head", "readout"),
                result_stats=asset("readoutStats"), residual_blend=asset("blend2"),
                embedding_lookup=dict(bank_size=4096, lane_width=16, vocab_size=248320,
                                      assets=[asset(f"lookup{i}") for i in range(61)]))


class DeviceBundleTests(unittest.TestCase):
    def test_incomplete_and_escaping_operator_names_fail(self):
        for field in ("decision_head", "residual_blend", "result_stats", "embedding_lookup"):
            model = manifest()
            del model[field]
            with self.assertRaises(ValueError):
                glide9_assets(model)
        model = manifest()
        banks = model["embedding_lookup"]["assets"]
        banks[0], banks[1] = banks[1], banks[0]
        with self.assertRaises(ValueError):
            glide9_assets(model)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "linked").symlink_to(root.parent)
            for name in ("../escape", "/outside", "..", "linked"):
                with self.assertRaises(ValueError):
                    bundle_child(root, name)

    def test_pack_compiles_every_operator_and_omits_host_weights(self):
        def compile_fixture(source, dest, arch, digest, attempts, functions):
            self.assertEqual(digest, (source / "main.hash").read_bytes().hex())
            self.assertEqual(len(functions), 1)
            dest.mkdir()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            model = manifest()
            (root / "model.json").write_text(json.dumps(model))
            for item in model["assets"] + glide9_assets(model):
                graph = root / item["asset"]
                graph.mkdir()
                (graph / "main.hash").write_bytes(bytes.fromhex(item["source_hash"]))
            (root / "rotary.fp16").write_bytes(bytes(128 * 64 * 2))
            ck = root / "checkpoint"
            ck.mkdir()
            for name in ("model.safetensors", "readout.safetensors"):
                (ck / name).touch()
            with patch("prepare.compile_stage", side_effect=compile_fixture), patch("prepare.tokenize_tables"):
                build(root, output, "h17d")
            packed = json.loads((output / "model.json").read_text())
            self.assertEqual(len(glide9_assets(packed)), 64)
            for item in packed["assets"] + glide9_assets(packed):
                self.assertTrue((output / item["asset"]).is_dir())
                self.assertTrue(item["asset"].endswith(".h17d.aimodelc"))
            self.assertEqual((output / "rotary.fp16").read_bytes(), (root / "rotary.fp16").read_bytes())
            self.assertFalse(any(output.glob("*.safetensors")))

class RetainedSourceTests(unittest.TestCase):
    def test_assembly_checks_payload_and_is_atomic(self):
        import hashlib
        import subprocess
        from pack_glide9_source import assemble

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared, backbone, extras = [root / n for n in ("prepared", "backbone", "extras")]
            for p in (prepared, backbone, extras):
                p.mkdir()
            model = manifest()
            model["source"]["revision"] = "ab54df5acba0eebf7df45560ffdaacd64e32df3b"
            model["context"] = 4096
            model["assets"] = [asset(f"stage{i}", f"scan{i}") for i in range(32)]
            (backbone / "model.json").write_text(json.dumps(model))
            for item in model["assets"] + glide9_assets(model):
                path = (backbone if item in model["assets"] else extras) / item["asset"]
                path.mkdir()
                data = path.name.encode()
                (path / "main.mlirb").write_bytes(data)
                hash_bytes = hashlib.sha256(data).digest()
                (path / "main.hash").write_bytes(hash_bytes)
                item.update(source_hash=hash_bytes.hex(), asset=path.stem + ".h17d.aimodelc")
            model["architecture"] = "h17d"
            (prepared / "model.json").write_text(json.dumps(model))
            subprocess.run(["swift", Path(__file__).with_name("glide9_rotary.swift"),
                            prepared / "rotary.fp16", "4096"], check=True)
            ck = backbone / "checkpoint"
            ck.mkdir()
            (ck / "licenses").mkdir()
            for name in ("tokenizer.json", "config.json", "decision_config.json", "LICENSE", "NOTICE"):
                (ck / name).write_bytes(b"{}")
            for name in ("config.json", "decision_config.json", "bpe.bin", "tokens.json"):
                (prepared / name).write_bytes(b"{}")
            def tables(source, directory):
                for name in ("tokens.json", "bpe.bin"):
                    (directory / name).write_bytes(b"{}")
            with patch("pack_glide9_source.tokenize_tables", side_effect=tables):
                assemble(prepared, backbone, [extras], root / "packed")
                portable = json.loads((root / "packed/model.json").read_text())
                self.assertEqual(len(list((root / "packed").glob("*.aimodel"))), 96)
                self.assertNotIn("architecture", portable)
                self.assertEqual(len(glide9_assets(portable)), 64)
                (extras / "lookup3.aimodel/main.mlirb").write_bytes(b"different")
                with self.assertRaisesRegex(ValueError, "Source bytes"):
                    assemble(prepared, backbone, [extras], root / "rejected")
                self.assertFalse((root / "rejected").exists())
