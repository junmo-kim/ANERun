"""Qwen3.5 image transformer for the fixed patch ABI in Vision.swift."""

import torch
from torch import nn
from torch.nn import functional as F


def projection(reader, prefix, inputs, outputs):
    # A 3-D patch kernel acts on the already flattened (C,T,H,W) patch.
    layer = nn.Conv2d(inputs, outputs, 1, device="meta")
    weight = reader.get_tensor(prefix + ".weight")
    bias = reader.get_tensor(prefix + ".bias")
    if weight.shape[0] != outputs or weight[0].numel() != inputs or bias.shape != (outputs,):
        raise ValueError(f"Invalid projection shape at {prefix}")
    layer.load_state_dict({"weight": weight.reshape(outputs, inputs, 1, 1).half(),
                           "bias": bias.half()}, assign=True)
    return layer


class MergeProjection(nn.Module):
    def __init__(self, reader, prefix, channels):
        super().__init__()
        width = 4 * channels
        matrix = reader.get_tensor(prefix + ".weight")
        bias = reader.get_tensor(prefix + ".bias")
        if matrix.shape != (width, width) or bias.shape != (width,):
            raise ValueError(f"Invalid merger projection at {prefix}")
        # Checkpoint column offset*C + c acts on channel c of Swift token
        # 4*group + offset. Each offset contributes to the same output row.
        self.taps = nn.ModuleList()
        for offset, block in enumerate(matrix.split(channels, dim=1)):
            has_bias = offset == 0
            tap = nn.Conv2d(channels, width, 1, bias=has_bias, device="meta")
            params = {"weight": block.contiguous().half()[:, :, None, None]}
            if has_bias:
                params["bias"] = bias.half()
            tap.load_state_dict(params, assign=True)
            self.taps.append(tap)

    def forward(self, tokens):
        return sum(tap(tokens[..., offset::4]) for offset, tap in enumerate(self.taps))


class ChannelNorm(nn.Module):
    def __init__(self, reader, prefix, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6, device="meta")
        self.norm.load_state_dict({k: reader.get_tensor(prefix + "." + k).half()
                                   for k in ("weight", "bias")}, assign=True)

    def forward(self, x):
        return self.norm(x.transpose(1, 3)).transpose(1, 3)


class VisionBlock(nn.Module):
    def __init__(self, reader, prefix, cfg):
        super().__init__()
        self.heads = cfg["num_heads"]
        channels = cfg["hidden_size"]
        inner = cfg["intermediate_size"]
        self.head_width = channels // self.heads
        self.norm1 = ChannelNorm(reader, prefix + ".norm1", channels)
        self.norm2 = ChannelNorm(reader, prefix + ".norm2", channels)
        self.qkv = projection(reader, prefix + ".attn.qkv", channels, 3 * channels)
        self.proj = projection(reader, prefix + ".attn.proj", channels, channels)
        self.up = projection(reader, prefix + ".mlp.linear_fc1", channels, inner)
        self.down = projection(reader, prefix + ".mlp.linear_fc2", inner, channels)

    @staticmethod
    def rotary(x, cos, sin):
        a, b = x.chunk(2, dim=2)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=2)

    def forward(self, x, cos, sin, mask):
        n = x.shape[-1]
        q, k, v = (t.reshape(1, self.heads, self.head_width, n)
                   for t in self.qkv(self.norm1(x)).chunk(3, dim=1))
        q = self.rotary(q, cos, sin) * self.head_width ** -0.5
        k = self.rotary(k, cos, sin)
        scores = q.transpose(2, 3) @ k
        weights = torch.softmax(scores + mask, dim=-1)
        context = (v @ weights.transpose(2, 3)).reshape_as(x)
        x = x + self.proj(context)
        return x + self.down(F.gelu(self.up(self.norm2(x)), approximate="tanh"))


class VisionGraph(nn.Module):
    def __init__(self, reader, prefix, cfg):
        super().__init__()
        if (cfg["hidden_act"] != "gelu_pytorch_tanh" or cfg["deepstack_visual_indexes"]
                or cfg["spatial_merge_size"] != 2):
            raise ValueError("Unsupported vision encoder")
        channels = cfg["hidden_size"]
        patch_width = cfg["in_channels"] * cfg["temporal_patch_size"] * cfg["patch_size"] ** 2
        self.patch = projection(reader, prefix + "patch_embed.proj", patch_width, channels)
        self.blocks = nn.ModuleList(VisionBlock(reader, prefix + f"blocks.{i}", cfg)
                                    for i in range(cfg["depth"]))
        self.norm = ChannelNorm(reader, prefix + "merger.norm", channels)
        self.up = MergeProjection(reader, prefix + "merger.linear_fc1", channels)
        self.down = projection(reader, prefix + "merger.linear_fc2", 4 * channels, cfg["out_hidden_size"])

    def forward(self, pixel_values, cos, sin, attn_mask, pos_embeds):
        if pixel_values.shape[-1] % 4:
            raise ValueError("Incomplete spatial merge group")
        x = self.patch(pixel_values) + pos_embeds
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
        return self.down(F.gelu(self.up(self.norm(x))))


def vision_inputs(cfg, capacity=1024):
    half_head = cfg["hidden_size"] // (2 * cfg["num_heads"])
    patch_width = cfg["in_channels"] * cfg["temporal_patch_size"] * cfg["patch_size"] ** 2
    return {"pixel_values": (1, patch_width, 1, capacity), "cos": (1, 1, half_head, capacity),
            "sin": (1, 1, half_head, capacity), "attn_mask": (1, 1, 1, capacity),
            "pos_embeds": (1, cfg["hidden_size"], 1, capacity)}
