"""Regression checks for the FP16 formulas used in the PPLX graphs."""

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pplx import attend_fold, rms_safe


class PPLXMathTests(unittest.TestCase):
    def test_key_block_averages_fit_in_float16(self):
        q = torch.zeros(1, 2, 2, 4, dtype=torch.float16)
        k = torch.zeros(1, 1, 32768, 4, dtype=torch.float16)
        v = torch.full_like(k, 4)
        bias = torch.zeros(1, 1, 1, 32768, dtype=torch.float16)
        bias[..., 16384:] = -10000
        block_mean = attend_fold(q, k, v, bias, kvh=1, mode="mmsum", key_block=4096)
        fused_mean = attend_fold(q, k, v, bias, kvh=1, mode="mmsum")
        self.assertTrue(torch.equal(block_mean, torch.full_like(q, 4)))
        self.assertTrue(torch.equal(fused_mean, torch.full_like(q, 4)))

    def test_causal_key_blocks_use_global_visibility_for_each_query_head(self):
        generator = torch.Generator().manual_seed(17)
        q = torch.randn(1, 4, 3, 8, generator=generator)
        k = torch.randn(1, 2, 12, 8, generator=generator)
        v = torch.randn(1, 2, 12, 8, generator=generator)
        row = torch.arange(3).reshape(1, 1, 3, 1)
        # Four keys precede this tile, three belong to it, the rest are masked.
        relative = torch.tensor([-1] * 4 + [0, 1, 2] + [3] * 5).reshape(1, 1, 1, 12)
        actual = attend_fold(q, k, v, relative, kvh=2, mode="mmsum", key_block=4, query_rows=row)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1),
            attn_mask=relative <= row, scale=1.0,
        )
        torch.testing.assert_close(actual, expected)

    def test_rms_epsilon_retains_tiny_magnitudes(self):
        x = torch.tensor([[0, 0], [1e-4, 1e-4], [3e-4, 3e-4]], dtype=torch.float16)
        source = x.float()
        reference = source / (source.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        torch.testing.assert_close(rms_safe(x).float(), reference, atol=0.001, rtol=0)


if __name__ == "__main__":
    unittest.main()
