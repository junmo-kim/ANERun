"""Weighted RMS normalization for channel and attention-head tensors."""

import math

import torch
from torch import nn


class BoundedRMSNorm(nn.Module):
    def __init__(self, dim, eps, axis=1, input_scale=1.0):
        super().__init__()
        self.axis = axis
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.epsilon_root = math.sqrt(eps) * input_scale
        self.dot_scale = 8 / math.sqrt(dim)

    def forward(self, value):
        row = value.movedim(self.axis, -1)
        # A positive scale cancels from x / sqrt(mean(x²) + eps).
        # Bound the scaled input by 16 even for a row of maximum FP16 values.
        scale = (row.abs().amax(-1, keepdim=True) / 16).clamp_min(self.epsilon_root)
        scaled = row / scale
        # The squared length of [8*x/sqrt(D), 8*sqrt(eps)] in scaled
        # coordinates is 64*(mean(x²) + eps)/scale². The factor 8 keeps
        # small contributions representable. The exact sum is at most 64*(16²+1).
        vector = torch.cat((scaled * self.dot_scale, (8 * self.epsilon_root) / scale), dim=-1)
        vectors = vector.reshape(1, -1, 1, vector.shape[-1])
        energy = (vectors @ vectors.transpose(2, 3)).reshape_as(scale)
        result = scaled * (8 * torch.rsqrt(energy)) * self.weight
        return result.movedim(-1, self.axis)
