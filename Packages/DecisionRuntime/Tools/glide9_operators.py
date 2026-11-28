"""GLiDE-9B device functions accompanying the 32 recurrent backbone stages."""
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file

from base_coreai import add_entry, save_asset
from decision_graph import CHUNK

HIDDEN = 4096
VOCAB = 248320
BANK_SIZE = 4096
LANES = 16
SLOTS = 255


class Selector(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.register_buffer("weights", rows.T.half().contiguous()[:, :, None, None])

    def forward(self, one_hot):
        return torch.nn.functional.conv2d(one_hot, self.weights) * 64


class Blend(torch.nn.Module):
    def forward(self, previous, current):
        return previous * 0.5 + current * 0.5


class Readout(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("kernel", weight.half().reshape(SLOTS, HIDDEN, 1, 1))

    def forward(self, hidden, mask):
        scores = torch.nn.functional.conv2d(hidden, self.kernel)
        return torch.softmax(scores / 1.0 + mask, dim=1)


class Stats(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("indices", torch.arange(SLOTS).half().reshape(1, SLOTS, 1, 1))

    def forward(self, p, exclude):
        top = p.amax(1, keepdim=True)
        second = (p + exclude).amax(1, keepdim=True)
        binary = (p[:, :1] * 2 - 1).abs()
        expected = (p * self.indices).sum(1, keepdim=True)
        return torch.cat([top - second, binary, expected], dim=1)


def export_function(output, filename, name, module, inputs, result):
    from coreai_torch import TorchConverter

    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    add_entry(converter, name, module, inputs, [result])
    path = output / (filename + ".aimodel")
    digest = save_asset(converter, path, "fastino/GLiDE-9B", "Fastino Community License")
    return dict(asset=path.name, functions=[name], source_hash=digest)


def export_operators(checkpoint, reader, output, context):
    if json.loads((checkpoint / "decision_config.json").read_text())["temperature"] != 1:
        raise ValueError("GLiDE-9B readout requires the pinned unit temperature")
    weight = load_file(checkpoint / "readout.safetensors")["weight"]
    table = reader.get_tensor("model.language_model.embed_tokens.weight")
    if (weight.dtype != torch.bfloat16 or list(weight.shape) != [SLOTS, HIDDEN]
            or table.dtype != torch.bfloat16 or list(table.shape) != [VOCAB, HIDDEN]):
        raise ValueError("Unexpected GLiDE-9B readout or embedding table")

    banks = []
    for i, rows in enumerate(table.split(BANK_SIZE)):
        name = f"lookup{i}"
        banks.append(export_function(output, name, name, Selector(rows),
                                     {"one_hot": (1, len(rows), 1, LANES)}, "hidden_out"))
    hidden = (1, HIDDEN, 1, CHUNK)
    slots = (1, SLOTS, 1, 1)
    blend = export_function(output, "blend2", "blend2", Blend(),
                            {"previous": hidden, "current": hidden}, "hidden_out")
    head = export_function(output, "head", "readout", Readout(weight),
                           {"hidden": (1, HIDDEN, 1, 1), "mask": slots}, "probabilities")
    stats = export_function(output, "readoutStats", "readoutStats", Stats(),
                            {"p": slots, "exclude": slots}, "statistics")
    subprocess.run(["swift", Path(__file__).with_name("glide9_rotary.swift"),
                    output / "rotary.fp16", str(context)], check=True)
    return dict(embedding_lookup=dict(bank_size=BANK_SIZE, lane_width=LANES,
                                     vocab_size=VOCAB, assets=banks),
                residual_blend=blend, decision_head=head, result_stats=stats)
