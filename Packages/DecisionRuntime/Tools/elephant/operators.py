"""Export the Elephant device ABI from the pinned publisher tensors."""
import hashlib
from pathlib import Path
import struct
import subprocess

import torch
from safetensors import safe_open

from base_coreai import add_entry, save_asset
from .export_embeddings import export, graphs
from .pack import BANK, HIDDEN, LANES, REPO, VOCAB
from .tail_stable_head import export_head


class StepScale(torch.nn.Module):
    def forward(self, x):
        magnitude = x.abs()
        step = torch.full_like(magnitude, 2.0**-14)
        for exponent in range(-13, 16):
            threshold = 2.0**exponent
            indicator = (magnitude.clamp(max=threshold) / threshold - 1) * 2048 + 1
            step = step + indicator.clamp(0, 1) * (threshold / 2)
        return torch.where(x < 0, -step, step)


class Two(torch.nn.Module):
    def forward(self, x, s):
        mantissa = x / s
        return mantissa + 8, mantissa + 16


class Finite(torch.nn.Module):
    def forward(self, x, s, b, c):
        first = (b - 8) * s
        second = (c - 16) * s
        tie = (x - first).abs() == s.abs() / 256
        rounded = torch.where(tie, second, first)
        return torch.where(x.abs() >= 65408, x, rounded.clamp(-65504., 65504.))


class VisionPosition(torch.nn.Module):
    def __init__(self, weight, channels=1152):
        super().__init__()
        if tuple(weight.shape) != (2304, channels):
            raise ValueError("Unexpected vision position table")
        self.register_buffer("weight", weight.T.half().contiguous()[:, :, None, None])
        self.register_buffer("axis", torch.arange(48, dtype=torch.float16).reshape(1, 48, 1, 1))

    def forward(self, row, column, fy, fx):
        wy = torch.relu(1 - ((row - self.axis) + fy).abs()).reshape(1, 48, 1, 16)
        wx = torch.relu(1 - ((column - self.axis) + fx).abs()).reshape(1, 1, 48, 16)
        return torch.nn.functional.conv2d((wy * wx).reshape(1, 2304, 1, 16), self.weight) * 1


class Stats(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("indices", torch.arange(256, dtype=torch.float16).reshape(1, 256, 1, 1))

    def forward(self, p, top_index, choice_scale, choice_bias, mad_scale):
        expected = (p * self.indices).sum(1, keepdim=True)
        choice = torch.relu(p.amax(1, keepdim=True) * choice_scale + choice_bias)
        error = (p * (self.indices - top_index).abs()).sum(1, keepdim=True)
        return torch.cat((expected, choice, torch.relu(1 - error * mad_scale)), dim=1)


class FeatureScale(torch.nn.Module):
    def forward(self, features):
        return features * 64


def export_auxiliaries(weights, output):
    from coreai_torch import TorchConverter

    def graph(name, module, shapes, outputs, author=REPO):
        converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
        entry = add_entry(converter, name, module, shapes, outputs, bind_weights=author == REPO)
        path = output / (name + ".aimodel")
        return dict(asset=path.name, source_hash=save_asset(converter, path, author, "MIT"),
                    functions=[entry])

    hidden = (1, HIDDEN, 1, 128)
    rounding = [graph("relusc", StepScale(), {"x": hidden}, ["s"], "fixture/sc"),
                graph("two", Two(), dict.fromkeys(("x", "s"), hidden), ["b", "c"], "fixture/even"),
                graph("finite", Finite(), dict.fromkeys(("x", "s", "b", "c"), hidden), ["y"], "fixture/even")]
    with safe_open(weights, framework="pt", device="cpu") as f:
        position = VisionPosition(f.get_tensor("model.visual.pos_embed.weight"))
    vision = graph("visionpos", position, dict.fromkeys(("row", "column", "fy", "fx"), (1, 1, 1, 16)), ["pos"])
    stats = graph("stats", Stats(), dict(p=(1, 256, 1, 1), **dict.fromkeys(
        ("top_index", "choice_scale", "choice_bias", "mad_scale"), (1, 1, 1, 1))), ["statistics"])
    scale = graph("featurescale", FeatureScale(), {"features": (1, HIDDEN, 1, 256)}, ["scaled"])
    return dict(residual_rounding=rounding, vision_position=vision, result_stats=stats, feature_scale=scale,
                decision_head=export_head(weights, output))


def export_operators(weights, output):
    result = export_auxiliaries(weights, output)
    banks = []
    for name, module in graphs(weights, 256):
        digest, entry = export(output, name, module, 256)
        row = dict(asset=name + ".aimodel", source_hash=digest, functions=[entry])
        if name == "unscale":
            unscale = row
        else:
            banks.append(row)
    subprocess.run(["swift", Path(__file__).with_name("rotary.swift"), output], check=True)
    fractions = b"".join(struct.pack("<e", n / d) for d in range(1, 512) for n in range(d))
    if hashlib.sha256(fractions).hexdigest() != "b7a89fd57e618c6bda78f6d9492c7a3d68e9ab60ba44c5a41fd278e3a6881eb7":
        raise ValueError("Unexpected FP16 vision fraction table")
    (output/"vision-fractions.fp16").write_bytes(fractions)
    return dict(result, embedding_lookup=dict(bank_size=BANK, lane_width=LANES, vocab_size=VOCAB, assets=banks),
                embedding_unscale=unscale,
                rotary_tables=dict(text="text-rotary.fp16", vision="vision-rotary.fp16"),
                vision_fractions="vision-fractions.fp16")
