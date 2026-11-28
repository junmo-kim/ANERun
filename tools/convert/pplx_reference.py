"""Pinned Hugging Face fp32 references for package canaries and evaluation."""
import json
from pathlib import Path

import numpy as np


def tokenizer(hf_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(hf_dir) / "tokenizer.json"))


def means(hf_dir, sequences):
    import torch
    from transformers import Qwen3Config, Qwen3Model

    torch.set_num_threads(8)
    spec = json.loads((Path(hf_dir) / "config.json").read_text())
    if spec.get("use_bidirectional_attention") is not True or set(spec["layer_types"]) != {"full_attention"}:
        raise ValueError("expected the bidirectional full-attention PPLX checkpoint")
    config = Qwen3Config(**{key: spec[key] for key in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "layer_types", "rope_parameters", "rms_norm_eps",
        "max_position_embeddings", "hidden_act", "attention_bias", "attention_dropout",
    )}, use_cache=False)
    model = Qwen3Model.from_pretrained(
        hf_dir, config=config, local_files_only=True, use_safetensors=True, dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    with torch.inference_mode():
        for sequence in sequences:
            ids = torch.tensor([sequence], dtype=torch.long)
            # The sequence is unpadded. Bypass Qwen's causal mask builder and
            # explicitly select bidirectional SDPA at every decoder layer.
            hidden = model(input_ids=ids, attention_mask={"full_attention": None}, is_causal=False).last_hidden_state
            yield hidden.mean(dim=1)[0].numpy()


def quantize(mean):
    import torch

    values = torch.as_tensor(np.asarray(mean, dtype=np.float32))
    return torch.clamp(torch.round(torch.tanh(values) * 127), -128, 127).numpy()
