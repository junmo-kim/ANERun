"""Score four Elephant regression requests with the standard Transformers Qwen3.5 model."""
import argparse
import base64
import json
from pathlib import Path
import string

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def requests(image):
    route = {"name": "team", "kind": "choice", "instructions": "Which team should handle this?",
             "candidates": [{"key": "billing", "description": "Charges, refunds, invoices"},
                            {"key": "tech", "description": "App or site faults"}]}
    return {
        "english": {"state": "I was charged twice. Please refund one of the charges.", "questions": [route,
                    {"name": "refund", "kind": "noul", "instructions": "Does the customer want a refund?", "candidates": []}]},
        "korean": {"state": "앱이 멈춰서 결제를 진행할 수 없습니다.", "questions": [route]},
        "json": {"state": '{"service":"checkout","status":"down","affected":"all customers"}',
                 "questions": [{"name": "urgency", "kind": "score", "instructions": "How urgent is this?",
                                "candidates": [{"key": "0", "description": "Can wait"}, {"key": "1", "description": "Today"},
                                               {"key": "2", "description": "Blocking every customer now"}]}]},
        "image": {"state": "Identify the shape in this picture.", "image": base64.b64encode(image.read_bytes()).decode(),
                  "questions": [{"name": "shape", "kind": "choice", "instructions": "Which shape is shown?",
                                "candidates": [{"key": "circle", "description": "A round disk"},
                                               {"key": "square", "description": "Four straight sides"}]}]},
    }


def prompt(state, q, image):
    start = ("<|im_start|>system\nApply the question to the state. Choose exactly one of the listed options. "
             "Respond with only its uppercase letter, with no explanation or reasoning.<|im_end|>\n"
             "<|im_start|>user\n<state>\n")
    if image:
        start += "<|vision_start|><|image_pad|><|vision_end|>\n"
    start += state + "\n</state>\n\n"
    kind = q["kind"]
    labels = [(c["key"], c["description"]) for c in q.get("candidates", [])]
    if kind == "noul":
        labels = [("yes", "the statement is true / the answer is yes"),
                  ("no", "the statement is false / the answer is no")]
    suffix = ("Rate the state: " if kind == "score" else "Question: ") + q["instructions"] + "\n\nOptions:\n"
    symbols = list(string.ascii_uppercase[:len(labels)])
    for code, (key, description) in zip(symbols, labels):
        suffix += f"{code}. " + ("level " if kind == "score" else "") + f"{key}: {description}\n"
    suffix += "\nAnswer with one letter: " + ", ".join(symbols) + ".<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return start, suffix, symbols, labels


def score(model, batch, prefix, slots):
    ids = batch["input_ids"]
    length = ids.shape[1]
    visible = torch.ones(length, length, dtype=torch.bool)
    visible[:prefix, prefix:] = False
    batch = dict(batch)
    if "image_grid_thw" in batch:
        positions, _ = model.model.get_rope_index(
            input_ids=ids, image_grid_thw=batch["image_grid_thw"],
            mm_token_type_ids=batch["mm_token_type_ids"], attention_mask=batch.get("attention_mask"))
        batch["position_ids"] = positions
    batch["attention_mask"] = {"full_attention": visible[None, None], "linear_attention": None}
    with torch.inference_mode():
        last = model.model(**batch, use_cache=False).last_hidden_state[0, -1]
        logits = model.lm_head.weight[slots].float() @ last.float()
        return torch.softmax(logits, dim=0).tolist()


def answer(kind, labels, values):
    if kind == "noul":
        return {"type": kind, "noul": values[0]}
    keys = [key for key, _ in labels]
    result = {"type": kind, "probabilities": dict(zip(keys, values))}
    n = len(values)
    top = values.index(max(values))
    if kind == "choice":
        result.update(choice=keys[top], confidence=(max(values) - 1/n) / (1 - 1/n))
    else:
        mad = sum(abs(i - (n-1)/2) for i in range(n)) / n
        result.update(score=sum(i*p for i, p in enumerate(values)), legend=dict(labels),
                      confidence=max(0, 1 - sum(abs(i-top)*p for i, p in enumerate(values)) / mad))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    tokenizer = processor.tokenizer
    if not args.tokens_only:
        model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.checkpoint, local_files_only=True, use_safetensors=True, output_loading_info=True,
            dtype=torch.bfloat16, attn_implementation="sdpa")
        if info["missing_keys"] or info.get("mismatched_keys"):
            raise ValueError(f"Incomplete checkpoint: {info}")
        model.eval()
    layouts = []
    for key, request in requests(args.image).items():
        (args.output / f"{key}.json").write_text(json.dumps(request, ensure_ascii=False))
        answers = {}
        for question in request["questions"]:
            image = "image" in request
            start, suffix, symbols, labels = prompt(request["state"], question, image)
            if image:
                with Image.open(args.image) as im:
                    batch = processor(text=[start + suffix], images=[im.convert("RGB")], return_tensors="pt")
            else:
                batch = tokenizer(start + suffix, add_special_tokens=False, return_tensors="pt")
            ids = batch["input_ids"][0].tolist()
            tail = tokenizer.encode(suffix, add_special_tokens=False)
            if ids[-len(tail):] != tail:
                raise ValueError("Unaligned question prefix")
            prefix = len(ids) - len(tail)
            layouts.append(dict(caseID=key, question=question["name"], ids=ids, prefix=prefix))
            if not args.tokens_only:
                slots = [tokenizer.convert_tokens_to_ids(letter) for letter in symbols]
                values = score(model, batch, prefix, slots)
                answers[question["name"]] = answer(question["kind"], labels, values)
        if not args.tokens_only:
            (args.output / f"{key}-reference.json").write_text(json.dumps({"answers": answers}, indent=2))
    (args.output / "token-layout.json").write_text(json.dumps(layouts))


if __name__ == "__main__":
    main()
