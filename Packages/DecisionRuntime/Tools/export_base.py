"""Export a raw Valen, GLiDE or Elephant checkpoint to a DecisionRuntime source bundle."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import string
import tempfile

import torch
from safetensors.torch import save_file

from base_coreai import export_stage, export_vision
from checkpoint_integrity import verify_text_checkpoint
from decision_graph import CHUNK, execution_schedule, load_weights, stage_ranges


PROFILES = {
    "valen-4b": ("Valen-Team/Valen-Decision-1.5-4B", "9b0cacc976d7fb72f252ee1adb1564f5f15e462c", "Apache-2.0"),
    "glide-2b": ("fastino/GLiDE-2B", "00f122c120cdeed82284106cb77d9bab5a50ad16", "Fastino Community License"),
    "glide-9b": ("fastino/GLiDE-9B", "ab54df5acba0eebf7df45560ffdaacd64e32df3b", "Fastino Community License"),
    "elephant-1": ("baselabs/elephant-1", "b3f8a93462fb1051a9de031e94a7ff380e254072", "MIT"),
}


def elephant_slots(vocab):
    # The slot alphabet consists of the first 256 single-token uppercase labels
    # in length-first, then lexicographic order (A..Z, AA..JU).
    labels = []
    for width in (1, 2):
        for chars in itertools.product(string.ascii_uppercase, repeat=width):
            label = "".join(chars)
            if label in vocab:
                labels.append(label)
                if len(labels) == 256:
                    return labels, [vocab[label] for label in labels]
    raise ValueError("Elephant tokenizer has fewer than 256 slot labels")


def runtime_weights(profile, source, reader, dest):
    dest.mkdir()
    prefix = "backbone." if profile == "valen-4b" else "model."
    names = [prefix + "language_model.embed_tokens.weight"]
    if profile in ("valen-4b", "elephant-1"):
        names.append(prefix + "visual.pos_embed.weight")
    if profile == "valen-4b":
        names.extend(sorted(k for k in reader.files if k.startswith("head.")))
    tensors = {} if profile == "glide-9b" else {k: reader.get_tensor(k) for k in names}
    if profile == "elephant-1":
        vocab = json.loads((source / "tokenizer.json").read_text())["model"]["vocab"]
        codes, ids = elephant_slots(vocab)
        tensors["slots.weight"] = reader.get_tensor("lm_head.weight")[ids].contiguous()
        (dest / "decision_config.json").write_text(json.dumps({"codes": codes}))
    if profile.startswith("glide-"):
        shutil.copyfile(source / "decision_config.json", dest / "decision_config.json")
        if profile != "glide-9b":
            shutil.copyfile(source / "readout.safetensors", dest / "readout.safetensors")
    if tensors:
        save_file(tensors, dest / "model.safetensors")
    for name in ("config.json", "tokenizer.json"):
        shutil.copyfile(source / name, dest / name)
    copy_notices(profile, source, dest)


def copy_notices(profile, source, dest):
    repo, revision, license_name = PROFILES[profile]
    for name in ("LICENSE", "NOTICE", "licenses"):
        path = source / name
        if path.is_dir():
            shutil.copytree(path, dest / name)
        elif path.is_file():
            shutil.copyfile(path, dest / name)
    # A data-only Hub snapshot may omit the accompanying license materials.
    # These local copies match the originals shipped with the pinned models.
    licenses = Path(__file__).resolve().parents[1] / "Licenses"
    if profile == "valen-4b" and not (dest / "LICENSE").exists():
        shutil.copyfile(licenses / "Apache-2.0.txt", dest / "LICENSE")
    if profile.startswith("glide-"):
        model = repo.split("/")[1]
        if not (dest / "LICENSE").exists():
            shutil.copyfile(licenses / (model + ".txt"), dest / "LICENSE")
        if not (dest / "NOTICE").exists():
            shutil.copyfile(Path(__file__).with_name("base_notices") / (model + ".txt"), dest / "NOTICE")
        qwen = dest / "licenses" / ("Qwen3.5-" + model.split("-")[1] + "-LICENSE")
        if not qwen.exists():
            qwen.parent.mkdir(exist_ok=True)
            shutil.copyfile(licenses / "Apache-2.0.txt", qwen)
    # The pinned Elephant snapshot advertises MIT but contains no LICENSE file.
    # Preserve its historical attribution even when the snapshot includes a notice.
    with (dest / "NOTICE").open("ab") as f:
        publisher = "Base Labs at Baseten, Elephant-1" if profile == "elephant-1" else repo
        f.write(f"\n{publisher} ({license_name}).\nhttps://huggingface.co/{repo}/tree/{revision}\n".encode())


@torch.inference_mode()
def export(profile, checkpoint, output, tiers):
    repo, revision, license_name = PROFILES[profile]
    if not tiers or tiers != sorted(set(tiers)) or any(n <= 0 or n % CHUNK or n > 16384 for n in tiers):
        raise ValueError("tiers must be increasing multiples of the chunk width up to 16384")
    if profile == "elephant-1" and tiers[-1] > 1024:
        raise ValueError("Elephant supports up to 1024 tokens")
    if output.exists():
        raise FileExistsError(output)
    verify_text_checkpoint(checkpoint, repo, revision)
    # Valen layer zero uses larger internal delta units. Packed state and
    # residual I/O retain the existing scale, as do all later recurrence layers.
    backbone = load_weights(
        checkpoint,
        scaled_recurrence=profile in ("valen-4b", "glide-2b", "glide-9b", "elephant-1"),
        valen_recurrence=profile == "valen-4b",
    )
    cfg = backbone.cfg
    ranges = stage_ranges(cfg, pair_before_attention=profile == "elephant-1")
    loop = None
    if profile.startswith("glide-"):
        loop = json.loads((checkpoint / "decision_config.json").read_text()).get("midloop")
    execution = execution_schedule(ranges, loop)
    visual = profile in ("valen-4b", "elephant-1")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        # These physical stages collided with GLiDE's native keys on h17d.
        # Bind lifted tensor bytes only at the entries changed in the qualified product.
        bound_stages = {0, 1, 4, 6, 9, 10, 12, 13, 16, 18, 21, 22} if profile == "elephant-1" else set()
        assets = [export_stage(backbone, start, end, i, tiers, stage, block_mask=visual,
                               repo=repo, license_name=license_name, bind_weights=i in bound_stages)
                  for i, (start, end) in enumerate(ranges)]
        bundle = dict(source=dict(repo=repo, revision=revision), assets=assets, context=tiers[-1],
                      tiers=tiers, width=CHUNK, hidden=cfg.hidden, rotary_dim=cfg.rotary_dim,
                      heads=cfg.heads, kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, block_mask=visual,
                      prompts=[], execution=execution,
                      stage_kinds=["attention" if cfg.layer_types[end] == "full_attention" else "linear"
                                   for _, end in ranges])
        if visual:
            prefix = "backbone." if profile == "valen-4b" else "model."
            vision_cfg = json.loads((checkpoint / "config.json").read_text())["vision_config"]
            bundle.update(export_vision(backbone.reader, prefix, vision_cfg, stage, repo, license_name))
        if profile == "glide-9b":
            from glide9_operators import export_operators
            bundle.update(export_operators(checkpoint, backbone.reader, stage, tiers[-1]))
        runtime_weights(profile, checkpoint, backbone.reader, stage / "checkpoint")
        if profile in ("elephant-1", "valen-4b"):
            if profile == "valen-4b":
                from valen_operators import export as export_operators
            else:
                from elephant.operators import export_operators
            weights = stage / "checkpoint" / "model.safetensors"
            bundle.update(export_operators(weights, stage))
            weights.unlink()  # Export-time tensors are fully represented in the device graphs.
        (stage / "model.json").write_text(json.dumps(bundle, indent=2) + "\n")
        stage.rename(output)


def main(profile=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if profile is None:
        parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tiers", type=int, nargs="+")
    args = parser.parse_args()
    profile = profile or args.profile
    defaults = [256, 512, 1024] if profile == "elephant-1" else [256, 512, 1024, 2048, 4096]
    export(profile, args.checkpoint, args.output, args.tiers or defaults)


if __name__ == "__main__":
    main()
