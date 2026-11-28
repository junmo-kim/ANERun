"""Trace float16 Swift ABI entry points and write Core AI source assets."""

import hashlib
import json

import torch

from decision_graph import CHUNK, AttendStage, ScanStage


def add_entry(converter, name, module, shapes, outputs, *, bind_weights=False):
    from coreai_torch import get_decomp_table

    args = tuple(torch.zeros(shape, dtype=torch.float16) for shape in shapes.values())
    program = torch.export.export(module.eval(), args).run_decompositions(get_decomp_table())
    if bind_weights:
        # Core AI's compiled cache key can omit external tensor bytes. Include the
        # lifted weights in the public function name before offline compilation.
        digest = hashlib.sha256()
        for spec in program.graph_signature.input_specs:
            tensors = program.state_dict if spec.target in program.state_dict else program.constants
            tensor = tensors.get(spec.target)
            if isinstance(tensor, torch.Tensor):
                data = tensor.detach().cpu().contiguous()
                header = json.dumps([spec.target, str(data.dtype), list(data.shape)]).encode()
                digest.update(len(header).to_bytes(8, "big"))
                digest.update(header)
                digest.update(memoryview(data.reshape(-1).view(torch.uint8).numpy()))
        name = f"{name}_{digest.hexdigest()}"
    converter.add_exported_program(program, input_names=list(shapes), output_names=outputs,
                                   entrypoint_name=name)
    return name


def save_asset(converter, path, repo, license_name):
    from coreai.runtime import AIModelAssetMetadata

    metadata = AIModelAssetMetadata()
    metadata.author = repo.split("/")[0]
    metadata.model_description = repo
    metadata.license = license_name
    converter.to_coreai().save_asset(path, metadata=metadata)
    digest = (path / "main.hash").read_bytes()
    if len(digest) != 32:
        raise ValueError(f"Unexpected Core AI digest in {path}")
    return digest.hex()


def export_vision(reader, prefix, cfg, output, repo, license_name):
    from coreai_torch import TorchConverter
    from decision_vision import VisionGraph, vision_inputs

    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    model = VisionGraph(reader, prefix + "visual.", cfg)
    name = add_entry(converter, "p1024", model, vision_inputs(cfg), ["features"])
    path = output / "vision.aimodel"
    digest = save_asset(converter, path, repo, license_name)
    return dict(vision_asset=path.name, vision_functions=[name], vision_source_hash=digest)


def export_stage(backbone, start, end, index, tiers, output, *, block_mask, repo, license_name, bind_weights=False):
    from coreai_torch import TorchConverter

    cfg = backbone.cfg
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    scan = ScanStage(backbone, start, end)
    attends = cfg.layer_types[end] == "full_attention"
    hidden = (1, cfg.hidden, 1, CHUNK)
    rotary = (1, 1, cfg.rotary_dim // 2, CHUNK)
    scan_inputs = {"hidden_in": hidden}
    scan_outputs = ["hidden_out"]
    if attends:
        scan_inputs.update(cos=rotary, sin=rotary)
        scan_outputs.extend(["query", "gate", "key", "value"])
    scan_inputs["packed_state"] = (1, scan.channels, 1, CHUNK)
    names = [add_entry(converter, f"scan{index}", scan, scan_inputs, scan_outputs + ["packed_state_out"],
                       bind_weights=bind_weights)]
    if attends:
        q = (1, cfg.heads, CHUNK, cfg.head_dim)
        for cap in tiers:
            name = f"attend{index}c{cap}"
            kv = (1, cfg.kv_heads, cap, cfg.head_dim)
            inputs = dict(hidden_in=hidden, query=q, gate=q, key=kv, value=kv,
                          mask=(1, 1, CHUNK if block_mask else 1, cap))
            names.append(add_entry(converter, name, AttendStage(backbone, end, cap), inputs, ["hidden_out"],
                                   bind_weights=bind_weights))
    path = output / f"stage{index}.aimodel"
    digest = save_asset(converter, path, repo, license_name)
    print(f"Exported {path.name} (layers {start}..{end})", flush=True)
    return dict(asset=path.name, functions=names, source_hash=digest)
