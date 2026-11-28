"""Export a pinned text decision checkpoint for the Swift scan/attend ABI."""
import argparse
import json
from pathlib import Path
import shutil
import string
import tempfile

import torch
from safetensors.torch import save_file

import decision_graph as graph
from base_coreai import export_stage
from checkpoint_integrity import verify_text_checkpoint
from decision_graph import CHUNK

PROFILES = {
    "knowline-9b-gen1": ("PelaAI/KnowLine-9B-Gen1", "325378d128e9f5eb5088196668f5874c5682b6c0"),
    "jeffan-9b-1211": ("tattrongvu/Jeffan-9B-1211", "862d1a7a8b64a8d708ddb0b0cd775a740f01841a"),
    "k-decision-9b": ("ONYULTEA/K-Decision-9B", "29e7d96e37de0c29b498d390dfb38d1e38c17339"),
}


def symbol_rows(profile, checkpoint):
    if profile == "k-decision-9b":
        meta = json.loads((checkpoint / "kdecision_meta.json").read_text())
        codes = list(string.ascii_uppercase)
        if (meta["marker"], meta["marker_id"]) != ("□", 169260):
            raise ValueError("Unsupported K-Decision marker")
        return codes, [meta["sym2id"][code] for code in codes]
    if profile == "knowline-9b-gen1":
        from transformers import AutoTokenizer
        from knowline_reference import codes_and_ids
        calibration = json.loads((checkpoint / "CALIBRATION.json").read_text())
        if calibration["readout_temperature"] != 1.18:
            raise ValueError("Unsupported KnowLine calibration")
        return codes_and_ids(AutoTokenizer.from_pretrained(checkpoint, local_files_only=True))
    if profile == "jeffan-9b-1211":
        from transformers import AutoTokenizer
        from jeffan_reference import codes_and_ids
        return codes_and_ids(AutoTokenizer.from_pretrained(checkpoint, local_files_only=True))
    raise ValueError(profile)


def write_bundle(profile, checkpoint, output, tiers):
    repo, revision = PROFILES[profile]
    runtime = output / "checkpoint"
    runtime.mkdir()
    reader = graph.CheckpointReader(checkpoint)
    embedding = "model.language_model.embed_tokens.weight"
    save_file({embedding: reader.get_tensor(embedding)}, runtime / "model.safetensors")
    codes, ids = symbol_rows(profile, checkpoint)
    save_file({"weight": reader.get_tensor("lm_head.weight")[ids].float()}, runtime / "readout.safetensors")
    (runtime / "decision_config.json").write_text(json.dumps({"codes": codes, "temperature": 1.0}))
    for name in ("config.json", "tokenizer.json"):
        shutil.copyfile(checkpoint / name, runtime / name)
    license_path = checkpoint / "LICENSE"
    shutil.copyfile(license_path if license_path.exists() else
                    Path(__file__).resolve().parents[1] / "Licenses/Apache-2.0.txt", runtime / "LICENSE")
    notice = (checkpoint / "NOTICE").read_text() if (checkpoint / "NOTICE").is_file() else ""
    (runtime / "NOTICE").write_text(f"{notice}\nConverted from {repo}@{revision}. Apache-2.0.\n")
    backbone = graph.load_weights(checkpoint)
    cfg = backbone.cfg
    ranges = graph.stage_ranges(cfg)
    execution = [dict(stages=[i], passes=1) for i in range(len(ranges))]
    causal = profile != "jeffan-9b-1211"
    if not causal:
        dm = json.loads((checkpoint / "dm_config.json").read_text())
        if (dm["attention"], dm["layout"], dm["readout"], dm["reads"]) != ("noncausal", "state_first", "code", 1):
            raise ValueError("Unsupported Jeffan renderer")
        loop = dm["loop"]
        if loop != {"blocks": "4-7,8-11,12-15", "passes": 2}:
            raise ValueError("Unsupported Jeffan recurrent schedule")
        blocks = [list(map(int, block.split("-"))) for block in loop["blocks"].split(",")]
        execution = graph.execution_schedule(ranges, {"blocks": blocks, "passes": loop["passes"]})
    stages = [export_stage(backbone, start, end, i, tiers, output, block_mask=causal,
                           repo=repo, license_name="Apache-2.0")
              for i, (start, end) in enumerate(ranges)]
    manifest = dict(source=dict(repo=repo, revision=revision), assets=stages,
                    execution=execution,
                    stage_kinds=["attention" if cfg.layer_types[end] == "full_attention" else "linear"
                                 for _, end in ranges], block_mask=causal, width=CHUNK,
                    tiers=tiers, context=tiers[-1], hidden=cfg.hidden, heads=cfg.heads,
                    kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, rotary_dim=cfg.rotary_dim)
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")


@torch.inference_mode()
def export(profile, checkpoint, output, tiers):
    if not tiers or sorted(set(tiers)) != tiers or any(x <= 0 or x % CHUNK or x > 16384 for x in tiers):
        raise ValueError("tiers must be increasing multiples of the chunk width up to 16384")
    if output.exists():
        raise FileExistsError(output)
    verify_text_checkpoint(checkpoint, *PROFILES[profile])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        write_bundle(profile, checkpoint, staging, tiers)
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tiers", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    args = parser.parse_args()
    export(args.profile, args.checkpoint, args.output, args.tiers)
