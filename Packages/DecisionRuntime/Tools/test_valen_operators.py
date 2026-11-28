"""Valen's device inputs preserve role spans and candidate order."""
import unittest

import torch

from valen_operators import Readout, RolePool


class ValenOperatorTests(unittest.TestCase):
    def test_role_crosses_chunk_boundary(self):
        # Nonzero padding and out-of-span values must contribute zero.
        rows = torch.arange(256, dtype=torch.float32).reshape(1, 1, 1, 256)
        pos = torch.arange(256)
        mask = ((pos >= 122) & (pos < 141)).reshape(1, 1, 1, 256)
        pool = RolePool()
        value = torch.zeros(1, 1, 1, 1)
        for chunk, valid in zip(rows.split(128, -1), mask.float().split(128, -1)):
            value = pool(chunk, valid, torch.tensor(19.), value)
        torch.testing.assert_close(value.flatten(), rows[..., 122:141].mean().reshape(1))

    def test_masked_candidates_preserve_order(self):
        logits = torch.full((1, 256, 1, 1), 10., dtype=torch.float16)
        logits[0, :3, 0, 0] = torch.tensor([-1., 2., 1.])
        mask = torch.full_like(logits, -30000.)
        mask[:, :3] = 0
        probabilities = Readout()(logits, mask).flatten()
        torch.testing.assert_close(probabilities[:3], torch.softmax(logits.flatten()[:3], 0))
        self.assertEqual(probabilities[3:].count_nonzero().item(), 0)
