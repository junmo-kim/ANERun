"""Export real-weight one-hot lookup graphs and a separate ANE compensation graph."""
import hashlib
import json

import torch

from .pack import BANK, HIDDEN, LANES, REPO, VOCAB


class Lookup(torch.nn.Module):
    def __init__(self, rows, gain):
        super().__init__()
        scaled = (rows.T.float() * (64 * gain)).half().contiguous()
        if not torch.isfinite(scaled).all():
            raise ValueError("Embedding gain exceeds the FP16 range")
        self.register_buffer("weight", scaled[:, :, None, None])

    def forward(self, one_hot):
        return torch.nn.functional.conv2d(one_hot, self.weight)


class Unscale(torch.nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.register_buffer("reciprocal", torch.tensor(1/gain, dtype=torch.float16))

    def forward(self, features):
        return features * self.reciprocal


def graphs(weights, gain):
    from safetensors import safe_open

    torch.set_num_threads(1)
    yield "unscale", Unscale(gain)
    with safe_open(weights, framework="pt", device="cpu") as file:
        rows = file.get_slice("model.language_model.embed_tokens.weight")
        if rows.get_shape() != [VOCAB, HIDDEN]:
            raise ValueError("Unexpected Elephant embedding tensor shape")
        for i, start in enumerate(range(0, VOCAB, BANK)):
            yield f"lookup{i}", Lookup(rows[start:start+BANK], gain)


def export(destination, name, graph, gain):
    from coreai_torch import TorchConverter, get_decomp_table
    from base_coreai import save_asset

    lookup = isinstance(graph, Lookup)
    operand = "one_hot" if lookup else "features"
    result = "hidden_out" if lookup else "scaled"
    shape = (1, graph.weight.shape[1] if lookup else HIDDEN, 1, LANES)
    program = torch.export.export(graph.eval(), (torch.zeros(shape, dtype=torch.float16),))
    program = program.run_decompositions(get_decomp_table())
    signature = hashlib.sha256(json.dumps(["elephant-embedding", name, shape, gain]).encode())
    for key, tensor in sorted(graph.state_dict().items()):
        signature.update(json.dumps([key, str(tensor.dtype), list(tensor.shape)]).encode())
        signature.update(memoryview(tensor.contiguous().reshape(-1).view(torch.uint8).numpy()))
    entry = f"{name}_{signature.hexdigest()}"
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    converter.add_exported_program(program, input_names=[operand], output_names=[result], entrypoint_name=entry)
    digest = save_asset(converter, destination/f"{name}.aimodel", REPO, "MIT")
    return digest, entry
