"""Export Valen's FP16 device inputs, role pooling, Mixer and calibrated readout."""
import argparse
import hashlib
import shutil
import tempfile
import json
from pathlib import Path
import struct
import subprocess

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from base_coreai import add_entry, save_asset
from elephant.operators import FeatureScale, Stats, VisionPosition
from elephant.export_embeddings import Lookup

REPO = "Valen-Team/Valen-Decision-1.5-4B"
HIDDEN, VOCAB, BANK, LANES = 2560, 248320, 4096, 16


class RolePool(nn.Module):
    def forward(self, hidden, valid, length, previous):
        return previous + (hidden * (valid / length.clamp(min=1))).sum(-1, keepdim=True)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_norm = nn.LayerNorm(256)
        self.channel_norm = nn.LayerNorm(256)
        self.token_mlp = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 4))
        self.channel_mlp = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, rows):
        rows = rows + self.token_mlp(self.token_norm(rows).transpose(1, 2)).transpose(1, 2)
        return rows + self.channel_mlp(self.channel_norm(rows))


class Mixer(nn.Module):
    def __init__(self, tensors):
        super().__init__()
        self.input_norm = nn.LayerNorm(HIDDEN)
        self.projection = nn.Linear(HIDDEN, 256)
        self.roles = nn.Parameter(torch.zeros(1, 4, 256))
        self.blocks = nn.Sequential(Block(), Block())
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1))
        self.load_state_dict(tensors)
        self.half()

    def forward(self, pooled):
        rows = pooled.reshape(1, HIDDEN, 4).transpose(1, 2)
        rows = self.projection(self.input_norm(rows)) + self.roles
        return self.output(self.blocks(rows)[:, 2, :]).reshape(1, 1, 1, 1)


class Readout(nn.Module):
    def forward(self, logits, mask):
        return torch.softmax(logits + mask, dim=1)


class ScaledVision(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, pixel_values, cos, sin, attn_mask, pos_embeds):
        return self.base(pixel_values, cos, sin, attn_mask, pos_embeds) * 64


def export_small_vision(checkpoint, output):
    from coreai_torch import TorchConverter
    from checkpoint_integrity import verify_text_checkpoint
    from decision_graph import CheckpointReader
    from decision_vision import VisionGraph, vision_inputs

    verify_text_checkpoint(checkpoint, REPO, "9b0cacc976d7fb72f252ee1adb1564f5f15e462c")
    cfg = json.loads((checkpoint / "config.json").read_text())["vision_config"]
    graph = ScaledVision(VisionGraph(CheckpointReader(checkpoint), "backbone.visual.", cfg))
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    entry = add_entry(converter, "p256", graph, vision_inputs(cfg, 256), ["features"], bind_weights=True)
    path = output / "vision256.aimodel"
    # Long exported vision symbols exceed the h16g compiler's layer-name buffer.
    from coreai.authoring.asset import AIModelAsset
    from valen_identity import rename_entries

    short = entry[:len("p256_") + 8]
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        donor = Path(tmp) / path.name
        save_asset(converter, donor, REPO, "Apache-2.0")
        program = AIModelAsset.load(donor).program
        rename_entries(program, {entry: short})
        stage = Path(tmp) / "short.aimodel"
        stage.mkdir()
        program._save_bytecode(stage / "main.mlirb")
        with (stage / "main.mlirb").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").digest()
        (stage / "main.hash").write_bytes(digest)
        shutil.copyfile(donor / "metadata.json", stage / "metadata.json")
        stage.rename(path)
    return dict(asset=path.name, source_hash=digest.hex(), functions=[short])


def export(weights, output):
    from coreai_torch import TorchConverter
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)

    def graph(name, module, shapes, result):
        converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
        entry = add_entry(converter, name, module, shapes, [result], bind_weights=True)
        path = output / (name + ".aimodel")
        digest = save_asset(converter, path, REPO, "Apache-2.0")
        item = dict(asset=path.name, source_hash=digest, functions=[entry])
        print(name, digest, flush=True)
        return item

    scalar = (1, 1, 1, 1)
    slots = (1, 256, 1, 1)
    with safe_open(weights, framework="pt", device="cpu") as f:
        head = {k.removeprefix("head."): f.get_tensor(k) for k in f.keys() if k.startswith("head.")}
        result = dict(role_mixer=graph("valenMixer", Mixer(head), {"pooled": (1, HIDDEN, 1, 4)}, "logit"),
                      role_pool=graph("valenPool", RolePool(), dict(hidden=(1, HIDDEN, 1, 128),
                          valid=(1, 1, 1, 128), length=scalar, previous=(1, HIDDEN, 1, 1)), "pooled"),
                      decision_head=graph("readout", Readout(), dict(logits=slots, mask=slots), "probabilities"))
        result["result_stats"] = graph("stats", Stats(), dict(p=slots, **dict.fromkeys(
            ("top_index", "choice_scale", "choice_bias", "mad_scale"), scalar)), "statistics")
        position = VisionPosition(f.get_tensor("backbone.visual.pos_embed.weight"), channels=1024)
        result["vision_position"] = graph("visionpos", position,
            dict.fromkeys(("row", "column", "fy", "fx"), (1, 1, 1, LANES)), "pos")
        result["feature_scale"] = graph("featurescale", FeatureScale(), {"features": (1, HIDDEN, 1, 256)}, "scaled")
        table = f.get_slice("backbone.language_model.embed_tokens.weight")
        if table.get_shape() != [VOCAB, HIDDEN]:
            raise ValueError("Unexpected Valen embedding shape")
        banks = [graph(f"lookup{i}", Lookup(table[start:start+BANK], 1),
                       {"one_hot": (1, min(BANK, VOCAB-start), 1, LANES)}, "hidden_out")
                 for i, start in enumerate(range(0, VOCAB, BANK))]
    result["embedding_lookup"] = dict(bank_size=BANK, lane_width=LANES, vocab_size=VOCAB, assets=banks)
    result["vision_fractions"] = "vision-fractions.fp16"
    (output/result["vision_fractions"]).write_bytes(
        b"".join(struct.pack("<e", n/d) for d in range(1, 512) for n in range(d)))
    subprocess.run(["swift", Path(__file__).parent/"elephant/rotary.swift", output, "valen-4b"], check=True)
    result["rotary_tables"] = dict(text="text-rotary.fp16", vision="vision-rotary.fp16")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--vision-checkpoint", type=Path, help="Include a scaled 256-patch vision graph from the pinned raw checkpoint")
    args = p.parse_args()
    records = export(args.weights, args.output)
    if args.vision_checkpoint:
        records["vision_256"] = export_small_vision(args.vision_checkpoint, args.output)
    (args.output/"operators.json").write_text(json.dumps(records, indent=2)+"\n")
