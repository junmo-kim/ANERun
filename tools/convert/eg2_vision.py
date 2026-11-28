"""Convert the installed Transformers vision layers to the fixed Swift patch/pool ABI."""
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import EmbeddingGemma2Model

BUDGETS = (70, 140, 280)
PACKAGE = "EmbeddingGemma2Vision"
CONVERTER = "tools/convert/eg2_vision.py"


def rotate_heads(heads, cos, sin):
    # Each spatial axis rotates 16 real/imaginary channel pairs per head.
    xr, xi, yr, yi = heads.split(16, dim=-1)
    return heads * cos + torch.cat((-xi, xr, -yi, yr), dim=-1) * sin


def norm(x, module):
    # Bounding the input before squaring keeps the RMS reduction finite in fp16.
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    y = x / scale
    variance = y.square().mean(dim=-1, keepdim=True) + (module.eps / scale) / scale
    y = y * torch.rsqrt(variance)
    return y * module.weight if module.with_scale else y


GELU_C = 0.7978845608028654  # sqrt(2 / pi)


def mlp(module, x):
    """The tower's gated MLP with GELU (tanh approximation) written out. Core ML's gelu op on the Neural Engine
    loses about ten times more precision in fp16 than the explicit tanh form (layer 9: relative error 0.0144 vs
    0.0012)."""
    gate = module.gate_proj(x)
    act = gate * (0.5 + 0.5 * torch.tanh(gate * (GELU_C + GELU_C * 0.044715 * gate * gate)))
    return module.down_proj(act * module.up_proj(x))


class VisionGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        tower = model.vision_tower
        self.patch_projection = tower.patch_embedder.input_proj
        self.encoder = tower.encoder
        self.projection = model.embed_vision.embedding_projection
        self.eps = model.config.vision_config.rms_norm_eps
        self.width = model.config.vision_config.hidden_size
        if model.config.vision_config.standardize:
            raise ValueError("expected the pinned EG2 vision tower without standardization")

    def forward(self, patches, positions, pool, valid, position_embeddings):
        x = self.patch_projection(2 * patches - 1) + position_embeddings
        cos, sin = (t[:, None] for t in self.encoder.rotary_emb(x, positions))
        key_bias = (valid[:, None, None, :] - 1) * 10000
        for layer in self.encoder.layers:
            attn = layer.self_attn
            h = norm(x, layer.input_layernorm)
            q = norm(attn.q_proj(h).reshape(1, -1, 12, 64).transpose(1, 2), attn.q_norm)
            k = norm(attn.k_proj(h).reshape(1, -1, 12, 64).transpose(1, 2), attn.k_norm)
            v = norm(attn.v_proj(h).reshape(1, -1, 12, 64).transpose(1, 2), attn.v_norm)
            q, k = rotate_heads(q, cos, sin), rotate_heads(k, cos, sin)
            weights = torch.softmax(q @ k.transpose(-1, -2) + key_bias, dim=-1)
            mix = (weights @ v).transpose(1, 2).reshape(1, -1, self.width)
            x = x + norm(attn.o_proj(mix), layer.post_attention_layernorm)
            x = x + norm(mlp(layer.mlp, norm(x, layer.pre_feedforward_layernorm)), layer.post_feedforward_layernorm)
        # Swift supplies one 3x3 average per soft token. The tower's sqrt(width)
        # factor before RMSNorm is equivalent to dividing epsilon by width.
        pooled = pool @ x
        scale = pooled.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        unit = pooled / scale
        # Padded pool rows are zero. Give them a finite denominator in fp16.
        nonempty = pool.amax(dim=-1) > 0
        variance = unit.square().mean(dim=-1, keepdim=True) + (self.eps / self.width) / scale.square()
        return self.projection(unit * torch.rsqrt(torch.where(nonempty[:, None], variance, 1.0)))


def convert(hf_dir, stage):
    """Create the package and little-endian position table in a fresh staging directory."""
    stage = Path(stage)
    torch.set_num_threads(8)
    model = EmbeddingGemma2Model.from_pretrained(
        hf_dir, local_files_only=True, use_safetensors=True, dtype=torch.float32, attn_implementation="eager",
    ).eval()
    table = model.vision_tower.patch_embedder.position_embedding_table.detach()
    table[:, :1024].numpy().astype("<f2").tofile(stage / "position_embeddings.f16")
    graph = VisionGraph(model).eval()
    del model, table
    functions = ct.utils.MultiFunctionDescriptor()
    for budget in BUDGETS:
        patches = budget * 9
        shapes = {"patches": (1, patches, 768), "positions": (1, patches, 2), "pool": (budget, patches),
                  "valid": (1, patches), "position_embeddings": (1, patches, 768)}
        with torch.no_grad():
            traced = torch.jit.trace(graph, tuple(torch.zeros(shape) for shape in shapes.values()), check_trace=False)
        ml = ct.convert(traced, inputs=[ct.TensorType(name=k, shape=s, dtype=np.float16) for k, s in shapes.items()],
                        outputs=[ct.TensorType(name="image_tokens", dtype=np.float16)], convert_to="mlprogram",
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15,
                        compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        target = str(stage / f"vision_{budget}.mlpackage")
        ml.save(target)
        functions.add_function(target, "main", f"vision_{budget}")
    functions.default_function_name = "vision_280"
    merged = str(stage / "merged.mlpackage")
    ct.utils.save_multifunction(functions, merged)
    ml = ct.models.MLModel(merged, compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    ml.user_defined_metadata["anerun.converter"] = CONVERTER
    ml.save(str(stage / f"{PACKAGE}.mlpackage"))
