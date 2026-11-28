"""Independent CPU/FP32 references from the pinned Hugging Face Qwen3 model.

Generate builder self-tests with the default CLI. Add --parity for standalone
text/ID fixtures and --long to compute 2060-token and exact 32768-token vectors as well.
"""
import argparse
from functools import partial
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

MAX_TOKENS = 32768
EOS = 151643
DIM = 1024
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QUERY = "서울역 근처 조용한 카페"
ROUTES = ("attn_S2048_Q2048", "attn_S8192_Q2048", "attn_S32768_Q2048")
SHORT_CASES = {
    "english": "The capital of China is Beijing.",
    "korean": "서울에서 부산까지 기차로 얼마나 걸리나요?",
    "code": "def clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n",
    "unicode": "Cafe\u0301 ☕️ | 北京 | السَّلَامُ | 👩🏽‍💻\n\t日本語",
    "whitespace": "  \n\n\t  ",
    "query": f"Instruct: {INSTRUCTION}\nQuery:{QUERY}",
}
LONG_LINE = (
    "{i:05d} | 서울에서 부산까지 | 北京与東京 | Cafe\u0301 ☕️ | العربية | 👩🏽‍💻\n"
    "    results[{i}] = min(max(value, 0), 4096)  # bounded result\n"
)


def tokenizer(hf_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(hf_dir) / "tokenizer.json"))


def encode(tok, text):
    ids = tok.encode(text, add_special_tokens=True).ids
    if ids[-1:] != [EOS]:
        raise ValueError("Hugging Face tokenizer did not append the model EOS")
    return ids


def validate_length(ids):
    if not 1 <= len(ids) <= MAX_TOKENS:
        raise ValueError(f"expected 1..{MAX_TOKENS} tokens including EOS, got {len(ids)}")


def model_for(hf_dir):
    import torch
    from transformers import Qwen3Model

    return Qwen3Model.from_pretrained(
        hf_dir, local_files_only=True,
        dtype=torch.float32, use_safetensors=True, attn_implementation="eager",
    ).eval()


def chunked_attention(module, query, key, value, attention_mask, *, chunk_size,
                      scaling=None, **kwargs):
    """HF attention interface for one unpadded, contiguous, causal sequence."""
    import torch
    from torch.nn.functional import scaled_dot_product_attention
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    if attention_mask is not None or query.shape[0] != 1 or query.shape[-2] != key.shape[-2]:
        raise ValueError("chunked attention requires one unpadded sequence without cache")
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    length = query.shape[-2]
    positions = torch.arange(length, device=query.device)
    output = torch.empty_like(query)
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        # An SDPA causal flag alone aligns to the upper left of the rectangle.
        # Use global positions for later query blocks and omit all future keys.
        mask = positions[:stop] <= positions[start:stop, None]
        output[:, :, start:stop] = scaled_dot_product_attention(
            query[:, :, start:stop], key[:, :, :stop], value[:, :, :stop],
            attn_mask=mask, dropout_p=0.0, scale=scaling,
        )
    return output.transpose(1, 2).contiguous(), None


def eos_hidden(model, ids, *, chunk_size=None):
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    validate_length(ids)
    if ids[-1] != EOS:
        raise ValueError("reference expects EOS as the last input token")
    if chunk_size is None and len(ids) > 2048:
        raise ValueError("long references require chunked attention")

    name = "anerun_qwen_chunked_reference"
    original = model.config._attn_implementation
    kwargs = {}
    if chunk_size is not None:
        if not 1 <= chunk_size <= 256:
            raise ValueError("expected a query chunk size in 1..256")
        ALL_ATTENTION_FUNCTIONS[name] = partial(chunked_attention, chunk_size=chunk_size)
        model.config._attn_implementation = name
        # Bypass HF's mask construction. The SDPA interface supplies causal
        # visibility for every chunk, including the last (possibly partial) one.
        kwargs["attention_mask"] = {"full_attention": None}
    try:
        with torch.inference_mode():
            inputs = torch.tensor([ids], dtype=torch.long, device="cpu")
            last = model(input_ids=inputs, use_cache=False, **kwargs).last_hidden_state[0, -1]
            return last.numpy().copy()
    finally:
        model.config._attn_implementation = original
        if chunk_size is not None:
            del ALL_ATTENTION_FUNCTIONS[name]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate(hf_dir, out):
    """Write selftest.json and packed raw little-endian f32 EOS vectors."""
    import numpy as np

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(hf_dir)
    model = model_for(hf_dir)
    inputs = {name: encode(tok, SHORT_CASES[name]) for name in ("english", "korean", "code")}
    blobs = {}
    for name, ids in inputs.items():
        hidden = eos_hidden(model, ids)
        if hidden.shape != (DIM,) or not np.isfinite(hidden).all():
            raise ValueError(f"invalid EOS vector: {name}")
        blobs[name] = hidden.astype("<f4").tobytes()
    routes = {"embed_64": ("english", "korean")}
    routes.update(dict.fromkeys(ROUTES, ("code",)))
    cases = []
    with (out / "selftest.bin").open("wb") as vectors:
        for function, names in routes.items():
            cases.append({
                "function": function, "kind": "route",
                "sequences": [inputs[name] for name in names], "offset": vectors.tell(),
            })
            for name in names:
                vectors.write(blobs[name])
    write_json(out / "selftest.json", {"format": "anerun-qwen-selftest/1", "cases": cases})


def boundary_text(tok, count=MAX_TOKENS):
    # Keep the multilingual/code body intact and fill the last few positions
    # with a token whose boundaries are checked with the actual HF tokenizer.
    lines = [LONG_LINE.format(i=i) for i in range(1000)]
    low, high = 0, len(lines)
    while low + 1 < high:
        mid = (low + high) // 2
        if len(encode(tok, "".join(lines[:mid]))) <= count:
            low = mid
        else:
            high = mid
    body = "".join(lines[:low])
    exact = body + " x" * (count - len(encode(tok, body)))
    if len(encode(tok, exact)) != count:
        raise ValueError(f"could not construct boundary fixture of {count} tokens")
    return exact


def generate_parity(hf_dir, out, *, long=False, chunk_size=128):
    import hashlib
    import importlib.metadata as metadata
    import numpy as np
    import torch

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(__file__).with_name("sources.json").read_text())["qwen"]
    for name, digest in sources["sha256"].items():
        with (Path(hf_dir) / name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != digest:
                raise ValueError(f"source digest mismatch: {name}")
    tok = tokenizer(hf_dir)
    model = model_for(hf_dir)
    cases = []
    max_error = 0.0

    def add_case(name, text, vectors, *, reject=False, chunked=False):
        nonlocal max_error
        ids = encode(tok, text)
        case = {"id": name, "text": text, "token_count": len(ids), "ids": ids}
        if name == "query":
            case["request"] = {"input": QUERY, "input_type": "query"}
        if reject:
            if len(ids) != MAX_TOKENS + 1:
                raise ValueError("overflow fixture must exceed the context limit by one token")
            try:
                validate_length(ids)
            except ValueError:
                case["expected_error"] = "context_length_exceeded"
            else:
                raise AssertionError("overflow fixture was accepted")
        else:
            start = time.monotonic()
            raw = eos_hidden(model, ids, chunk_size=chunk_size if chunked else None)
            norm = np.linalg.norm(raw)
            if raw.shape != (DIM,) or not np.isfinite(norm) or norm <= 0:
                raise ValueError(f"invalid EOS vector: {name}")
            if not chunked:
                alternate = eos_hidden(model, ids, chunk_size=min(7, chunk_size))
                error = float(np.max(np.abs(raw - alternate)))
                max_error = max(error, max_error)
                np.testing.assert_allclose(raw, alternate, rtol=1e-4, atol=5e-4)
            case["offset"] = vectors.tell()
            vectors.write(raw.astype("<f4").tobytes())
            vectors.write((raw / norm).astype("<f4").tobytes())
            print(f"{name}: {len(ids)} tokens, {time.monotonic() - start:.2f}s", flush=True)
        cases.append(case)

    exact = boundary_text(tok)
    with TemporaryDirectory(prefix=".qwen-reference-", dir=out) as staging:
        staging = Path(staging)
        binary = staging / "parity.bin"
        with binary.open("wb") as vectors:
            for name, text in SHORT_CASES.items():
                add_case(name, text, vectors)
            if long:
                add_case("tile-crossing-2060", boundary_text(tok, 2060), vectors, chunked=True)
                add_case("exact-32768", exact, vectors, chunked=True)
            add_case("overflow", exact + " x", vectors, reject=True)
        write_json(staging / "parity.json", {
            "format": "anerun-qwen-parity/1",
            "source": {key: sources[key] for key in ("id", "revision", "sha256")},
            "versions": {name: metadata.version(name) for name in ("torch", "transformers", "tokenizers", "numpy")},
            "threads": torch.get_num_threads(),
            "chunk_size": chunk_size,
            "dimension": DIM,
            "layout": "raw EOS then L2-normalized embedding, each 1024 little-endian float32",
            "short_eager_vs_chunked_max_abs": max_error,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "cases": cases,
        })
        for name in ("parity.bin", "parity.json"):
            (staging / name).replace(out / name)


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--parity", action="store_true", help="generate standalone API parity fixtures")
    parser.add_argument("--long", action="store_true", help="include the 2060- and 32768-token parity vectors")
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.chunk_size <= 256:
        parser.error("expected positive threads and chunk size in 1..256")
    torch.set_num_threads(args.threads)
    if args.parity or args.long:
        generate_parity(args.hf_dir, args.out, long=args.long, chunk_size=args.chunk_size)
    else:
        generate(args.hf_dir, args.out)


if __name__ == "__main__":
    main()
