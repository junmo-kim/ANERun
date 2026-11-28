"""Export the qualified majority-complement FP16 slot head."""
import hashlib

import torch

from .pack import REPO


class TailStableHead(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight.half().contiguous()[:, :, None, None])

    def forward(self, hidden, mask):
        logits = torch.nn.functional.conv2d(hidden, self.weight) + mask
        peak = logits.amax(dim=1, keepdim=True)
        max_slots = (logits == peak).to(logits.dtype)
        other = 1 - max_slots
        relative = (logits - peak).exp()
        ties = max_slots.sum(dim=1, keepdim=True) - 1
        tail = (relative * other).sum(dim=1, keepdim=True) + ties
        inverse = (1 + tail).reciprocal()
        # The complement retains precision for a strict majority. Ties and
        # smaller peaks use the direct quotient.
        complement_slots = max_slots * (tail < 1).to(logits.dtype)
        peak_probability = 1 - tail * inverse
        return complement_slots * peak_probability + (1 - complement_slots) * relative * inverse


def export_head(weights, output):
    from safetensors import safe_open
    from coreai_torch import TorchConverter, get_decomp_table
    from base_coreai import save_asset

    with safe_open(weights, framework="pt", device="cpu") as f:
        weight = f.get_tensor("slots.weight")
    if list(weight.shape) != [256, 4096]:
        raise ValueError("Unexpected slot weights")
    model = TailStableHead(weight)
    digest = hashlib.sha256(b"elephant-tail-stable-head-v2")
    digest.update(memoryview(model.weight.reshape(-1).view(torch.uint8).numpy()))
    entry = f"readout_{digest.hexdigest()}"
    program = torch.export.export(model.eval(), (torch.zeros(1, 4096, 1, 1, dtype=torch.float16),
                                                 torch.zeros(1, 256, 1, 1, dtype=torch.float16)))
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    converter.add_exported_program(program.run_decompositions(get_decomp_table()),
                                   input_names=["hidden", "mask"], output_names=["probabilities"],
                                   entrypoint_name=entry)
    source = output / "readout.aimodel"
    source_hash = save_asset(converter, source, REPO, "MIT")
    return dict(asset=source.name, source_hash=source_hash, functions=[entry])
