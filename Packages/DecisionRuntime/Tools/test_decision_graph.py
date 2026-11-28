"""Two-chunk CPU parity against the installed Transformers Qwen3.5 model."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from decision_graph import CHUNK, AttendStage, ScanStage, load_weights, stage_ranges
from text_norm import BoundedRMSNorm


class TextStageTest(unittest.TestCase):
    def test_first_layer_scale_does_not_follow_stage_boundaries(self):
        cfg = SimpleNamespace(layer_types=("linear_attention",) * 3)
        backbone = SimpleNamespace(cfg=cfg, valen_recurrence=True)

        def layer(*args, **kwargs):
            node = torch.nn.Identity()
            node.state_channels = 1
            return node

        with patch("decision_graph.DeltaLayer", side_effect=layer) as delta:
            ScanStage(backbone, 0, 1)
            ScanStage(backbone, 2, 2)
        self.assertEqual(
            [(call.args[1], call.kwargs["recurrence_scale"]) for call in delta.call_args_list],
            [("layers.0.", 256), ("layers.1.", 64), ("layers.2.", 64)],
        )

    def test_fp16_norm_range(self):
        # Uniform near-limit input and a single large channel have opposite
        # effects on the mean absolute scale. Check both along with tiny rows.
        x = torch.zeros(1, 4096, 1, 4, dtype=torch.float16)
        x[..., 0] = 65504
        x[:, 17, :, 1] = -65504
        x[..., 2] = 0.0002
        expected = x.float() * torch.rsqrt(x.float().square().mean(1, keepdim=True) + 1e-6)
        actual = BoundedRMSNorm(4096, 1e-6).half()(x)
        torch.testing.assert_close(actual.float(), expected, rtol=0.003, atol=1e-6)

    def test_weighted_head_norm_with_scaled_epsilon(self):
        # Each (head, token) owns a reduction, including at state boundaries.
        dim = 128
        rng = torch.Generator().manual_seed(72)
        rows = torch.randn(1, 3, 4, dim, generator=rng).half()
        rows[:, :, 0] = 0
        rows[:, :, 1] *= 0.0005
        rows[:, :, 2] *= 1000
        module = BoundedRMSNorm(dim, 1e-6 / dim, axis=-1, input_scale=2).half()
        module.weight.copy_(torch.linspace(-1, 1, dim))
        gold = torch.nn.functional.rms_norm(
            rows.double(), (dim,), module.weight.double(), eps=4e-6 / dim)
        exported = torch.export.export(module, (rows,)).module()
        torch.testing.assert_close(exported(rows).double(), gold, rtol=0.003, atol=0.0005)

    @torch.inference_mode()
    def test_cached_chunks(self):
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng(devices=[]):
            torch.manual_seed(314)
            cfg = Qwen3_5TextConfig(
                hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, layer_types=["linear_attention", "full_attention"],
                linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
                linear_num_key_heads=2, linear_num_value_heads=4, vocab_size=100,
                rope_parameters=dict(rope_theta=10_000_000, rope_type="default", partial_rotary_factor=0.5,
                                     mrope_section=[1, 1, 2], mrope_interleaved=True),
            )
            cfg._attn_implementation = "eager"
            model = Qwen3_5TextModel(cfg).eval()
            model.layers[0].linear_attn.A_log.uniform_(-1, 1)
            model.layers[0].linear_attn.dt_bias.uniform_(-3, 0)
            cfg.save_pretrained(tmp)
            save_file({"model." + k: v for k, v in model.state_dict().items()},
                      Path(tmp) / "model.safetensors")
            length = 2 * CHUNK
            ids = torch.randint(0, cfg.vocab_size, (1, length))
            expected = model(input_ids=ids, use_cache=False).last_hidden_state
            embedding = model.embed_tokens(ids).transpose(1, 2).unsqueeze(2) * 64
            angles = torch.arange(length)[:, None] * 10_000_000 ** (-torch.arange(0, 8, 2) / 8)
            mask = torch.ones(length, length).triu(1) * -30000
            for dtype, scaled, atol in [(torch.float32, False, 2e-5), (torch.float16, False, 0.012),
                                        (torch.float32, True, 2e-5), (torch.float16, True, 0.012)]:
                with self.subTest(dtype=dtype, scaled=scaled):
                    backbone = load_weights(tmp, dtype=dtype, scaled_recurrence=scaled)
                    self.assertEqual(stage_ranges(backbone.cfg), [(0, 0), (1, 1)])
                    self.assertEqual(stage_ranges(backbone.cfg, pair_before_attention=True), [(0, 1)])
                    scan = ScanStage(backbone, 0, 0)
                    query = ScanStage(backbone, 1, 1)
                    attend = AttendStage(backbone, 1, length)
                    state = torch.zeros(1, scan.channels, 1, CHUNK, dtype=dtype)
                    rows = []
                    for i in range(0, length, CHUNK):
                        positions = angles[i:i + CHUNK].T[None, None]
                        hidden, state = scan(embedding[..., i:i + CHUNK].to(dtype), state)
                        empty = torch.zeros(1, query.channels, 1, CHUNK, dtype=dtype)
                        *row, _ = query(hidden, positions.cos().to(dtype), positions.sin().to(dtype), empty)
                        rows.append(row)
                    keys = torch.cat([r[3] for r in rows], dim=2)
                    values = torch.cat([r[4] for r in rows], dim=2)
                    chunks = [attend(*row[:3], keys, values, local_mask[None, None].to(dtype))
                              for row, local_mask in zip(rows, mask.split(CHUNK))]
                    actual = torch.cat(chunks, dim=3).squeeze(2).transpose(1, 2)
                    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=atol)


if __name__ == "__main__":
    unittest.main()
