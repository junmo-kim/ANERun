"""CPU tests for graph layout, raw slot selection and recurrent stage grouping."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from decision_graph import CheckpointReader
from decision_vision import VisionGraph, vision_inputs
from export_base import PROFILES, copy_notices, elephant_slots, execution_schedule, export, runtime_weights


class BaseExportTests(unittest.TestCase):
    def test_valen_profile_selects_qualified_graph_math(self):
        cfg = SimpleNamespace(
            layer_types=("linear_attention", "full_attention"),
            hidden=64, rotary_dim=16, heads=4, kv_heads=2, head_dim=16,
        )
        backbone = SimpleNamespace(cfg=cfg, reader=None)
        def emit_runtime_weights(profile, source, reader, dest):
            dest.mkdir()
            (dest / "model.safetensors").touch()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("export_base.verify_text_checkpoint"),
            patch("export_base.load_weights", return_value=backbone) as load,
            patch("export_base.export_stage", return_value={}),
            patch("export_base.export_vision", return_value={}),
            patch("export_base.runtime_weights", side_effect=emit_runtime_weights),
            patch("glide9_operators.export_operators", return_value={}),
            patch("elephant.operators.export_operators", return_value={}),
            patch("valen_operators.export", return_value={}),
        ):
            source = Path(tmp)
            (source / "config.json").write_text('{"vision_config": {}}')
            (source / "decision_config.json").write_text("{}")
            for profile in PROFILES:
                with self.subTest(profile=profile):
                    export(profile, source, source / profile, [256])
                    load.assert_called_with(source, scaled_recurrence=True,
                                            valen_recurrence=profile == "valen-4b")

    def test_glide_midloop_includes_linear_tail(self):
        ranges = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 17),
                  (18, 19), (20, 23), (24, 27), (28, 31)]
        loop = {"blocks": [[12, 17], [18, 23]], "passes": 2}
        result = execution_schedule(ranges, loop)
        self.assertEqual([x["stages"] for x in result], [[0], [1], [2], [3, 4], [5, 6], [7], [8]])
        self.assertEqual([x["passes"] for x in result], [1, 1, 1, 2, 2, 1, 1])
        with self.assertRaisesRegex(ValueError, "inside an exported stage"):
            execution_schedule(ranges, {"blocks": [[12, 16]], "passes": 2})

    def test_glide_data_only_snapshot_keeps_upstream_licenses(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            for size in ("2B", "9B"):
                with self.subTest(size=size):
                    dest = source / size
                    dest.mkdir()
                    copy_notices("glide-" + size.lower(), source, dest)
                    upstream = Path(__file__).with_name("base_notices") / f"GLiDE-{size}.txt"
                    self.assertTrue((dest / "NOTICE").read_bytes().startswith(upstream.read_bytes()))
                    self.assertIn("Fastino Community License", (dest / "LICENSE").read_text())
                    license_file = dest / "licenses" / f"Qwen3.5-{size}-LICENSE"
                    self.assertIn("Apache License", license_file.read_text())

    def test_elephant_raw_rows_and_notices(self):
        labels = [chr(i) for i in range(65, 91)]
        alphabet = list(labels)
        labels += [a + b for a in alphabet for b in alphabet if a + b != "BQ"]
        vocab = {label: 750 - i for i, label in enumerate(labels)}
        codes, ids = elephant_slots(vocab)
        self.assertEqual(codes, labels[:256])
        self.assertEqual(ids, list(range(750, 494, -1)))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            dest = source / "runtime"
            head = torch.arange(1502, dtype=torch.float32).reshape(-1, 2)
            embed = "model.language_model.embed_tokens.weight"
            pos = "model.visual.pos_embed.weight"
            save_file({"lm_head.weight": head, embed: head.clone(), pos: head[:4].clone()},
                      source / "model.safetensors")
            (source / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}}))
            (source / "config.json").write_text("{}")
            (source / "NOTICE").write_bytes(b"upstream\r\n")
            (source / "LICENSE").write_bytes(b"original license\n")
            runtime_weights("elephant-1", source, CheckpointReader(source), dest)
            stored = load_file(dest / "model.safetensors")
            self.assertEqual(set(stored), {embed, pos, "slots.weight"})
            torch.testing.assert_close(stored["slots.weight"], head[ids])
            self.assertEqual(json.loads((dest / "decision_config.json").read_text())["codes"], codes)
            self.assertTrue((dest / "NOTICE").read_bytes().startswith(b"upstream\r\n"))
            self.assertEqual((dest / "LICENSE").read_bytes(), b"original license\n")

    @unittest.skipUnless(os.environ.get("DECISION_VISION_CHECKPOINT"), "set DECISION_VISION_CHECKPOINT to a raw snapshot")
    @torch.inference_mode()
    def test_real_vision_against_transformers(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        source = Path(os.environ["DECISION_VISION_CHECKPOINT"])
        reader = CheckpointReader(source)
        config = json.loads((source / "config.json").read_text())
        cfg = config["vision_config"]
        prefix = ("backbone." if config.get("qwen_execution") == "shared_state" else "model.") + "visual."
        hf_config = Qwen3_5VisionConfig(**cfg)
        hf_config._attn_implementation = "eager"
        reference = Qwen3_5VisionModel(hf_config).float().eval()
        reference.load_state_dict({k: reader.get_tensor(prefix + k).float() for k in reference.state_dict()},
                                  assign=True)
        # A rectangular grid exercises position interpolation and both RoPE axes.
        gh, gw = 8, 12
        n = gh * gw
        rng = torch.Generator().manual_seed(123)
        px = torch.rand((n, 3, 1, 16, 16), generator=rng).expand(-1, -1, 2, -1, -1).reshape(n, 1536) * 2 - 1
        expected = reference(px, torch.tensor([[1, gh, gw]])).pooler_output
        del reference
        c = cfg["hidden_size"]
        order = torch.arange(n).reshape(gh//2, 2, gw//2, 2).permute(0, 2, 1, 3).flatten()
        positions = reader.get_tensor(prefix + "pos_embed.weight").float().T.reshape(1, c, 48, 48)
        pos = F.interpolate(positions, (gh, gw), mode="bilinear", align_corners=True).reshape(1, c, 1, n)
        quarter = c // cfg["num_heads"] // 4
        freq = 10000.0 ** (-torch.arange(quarter)[:, None] / quarter)
        angles = torch.cat([freq * (order // gw), freq * (order % gw)]).reshape(1, 1, 2 * quarter, n)
        graph = VisionGraph(reader, prefix, cfg).float().eval()
        actual = graph(px.T.reshape(1, 1536, 1, n), angles.cos(), angles.sin(),
                       torch.zeros(1, 1, 1, n), pos[:, :, :, order])[0, :, 0, :].T
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(((actual - expected).norm() / expected.norm()).item(), 1e-4)

    def test_patch_attention_and_merger_against_token_major_math(self):
        # Two heads, distinct positions and four padded keys exercise the head,
        # token, rotary and 2x2 merge axes independently of the traced layout.
        rng = torch.Generator().manual_seed(310)
        def rand(*shape):
            return torch.randn(shape, generator=rng) * 0.2

        c, inner, h, n = 8, 12, 2, 12
        cfg = dict(hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[], spatial_merge_size=2,
                   hidden_size=c, num_heads=h, depth=1, intermediate_size=inner, out_hidden_size=6,
                   in_channels=3, temporal_patch_size=2, patch_size=2)
        patch_width = 24
        data = {"patch_embed.proj.weight": rand(c, 3, 2, 2, 2), "patch_embed.proj.bias": rand(c)}
        for key, i, o in [("blocks.0.attn.qkv", c, 3*c), ("blocks.0.attn.proj", c, c),
                          ("blocks.0.mlp.linear_fc1", c, inner), ("blocks.0.mlp.linear_fc2", inner, c),
                          ("merger.linear_fc1", 4*c, 4*c), ("merger.linear_fc2", 4*c, 6)]:
            data[key + ".weight"], data[key + ".bias"] = rand(o, i), rand(o)
        for key in ["blocks.0.norm1", "blocks.0.norm2", "merger.norm"]:
            data[key + ".weight"], data[key + ".bias"] = 1 + rand(c), rand(c)
        # The source graph quantizes checkpoint parameters to FP16 for export.
        data = {k: v.half().float() for k, v in data.items()}

        def linear(x, name):
            return F.linear(x, data[name + ".weight"].flatten(1), data[name + ".bias"])

        def norm(x, name):
            return F.layer_norm(x, (c,), data[name + ".weight"], data[name + ".bias"], 1e-6)

        with tempfile.TemporaryDirectory() as tmp:
            save_file(data, Path(tmp) / "model.safetensors")
            graph = VisionGraph(CheckpointReader(tmp), "", cfg).float().eval()
        patches, pos = rand(n, patch_width), rand(n, c)
        freq = rand(n, c//h//2)
        mask = torch.zeros(n)
        mask[-4:] = -30000
        x = linear(patches, "patch_embed.proj") + pos
        q, k, v = linear(norm(x, "blocks.0.norm1"), "blocks.0.attn.qkv").view(n, 3, h, c//h).unbind(1)
        def rotate(t):
            pair = torch.cat((-t[:, :, c//h//2:], t[:, :, :c//h//2]), -1)
            a = freq.repeat(1, 2).unsqueeze(1)
            return t * a.cos() + pair * a.sin()
        attn = F.scaled_dot_product_attention(rotate(q).transpose(0, 1), rotate(k).transpose(0, 1),
                                              v.transpose(0, 1), attn_mask=mask.expand(n, n))
        x = x + linear(attn.transpose(0, 1).reshape(n, c), "blocks.0.attn.proj")
        mid = F.gelu(linear(norm(x, "blocks.0.norm2"), "blocks.0.mlp.linear_fc1"), approximate="tanh")
        x = x + linear(mid, "blocks.0.mlp.linear_fc2")
        merged = norm(x, "merger.norm").reshape(n//4, 4*c)
        expected = linear(F.gelu(linear(merged, "merger.linear_fc1")), "merger.linear_fc2")
        shapes = vision_inputs(cfg, n)
        args = [a.T.reshape(s) for a, s in zip((patches, freq.cos(), freq.sin(), mask[:, None], pos),
                                               shapes.values())]
        actual = graph(*args)[0, :, 0, :].T
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        restored = torch.export.export(graph, tuple(args)).module()
        torch.testing.assert_close(restored(*args)[0, :, 0, :].T, expected, rtol=1e-5, atol=2e-6)
        with self.assertRaisesRegex(ValueError, "Incomplete spatial merge group"):
            graph(*(arg[..., :-1] for arg in args))


if __name__ == "__main__":
    unittest.main()
