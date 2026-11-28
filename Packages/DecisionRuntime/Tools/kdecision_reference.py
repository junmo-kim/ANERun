"""CPU reference using the checkpoint symbol table and the Transformers causal LM."""
import argparse
import json
from pathlib import Path
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = ("당신은 한국어 의사결정 엔진이다. [상태]와 [질문]들을 읽고, 각 질문의 정답 심볼을 예측하라. "
               "질문 유형: choice(단일선택), noul(예/아니오). 설명 없이 각 필드의 심볼을 결정하며, 모든 필드는 한 번에 판단한다.")


def reference(checkpoint, fixtures, tokens_only):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    meta = json.loads((checkpoint / "kdecision_meta.json").read_text())
    marker, marker_id = meta["marker"], meta["marker_id"]
    if tokenizer.encode(marker, add_special_tokens=False) != [marker_id]:
        raise ValueError("Invalid K-Decision marker")
    model = None
    if not tokens_only:
        model, info = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True, trust_remote_code=False, use_safetensors=True,
            output_loading_info=True,
        )
        if info["missing_keys"]:
            raise ValueError(f"Missing HF weights: {info['missing_keys']}")
        model.eval()
    for fixture in fixtures:
        request = fixture["request"]
        state = request["state"].replace(marker, "▪").replace("@", "＠")
        lines = [INSTRUCTION, "", "[상태]", state, "", "[질문]"]
        fields, outputs = [], []
        for question in request["questions"]:
            name, kind = question["name"], question["kind"]
            if kind == "choice":
                choices = question["candidates"]
                if not 2 <= len(choices) <= 26:
                    raise ValueError("Expected 2..26 choices")
                symbols = string.ascii_uppercase[:len(choices)]
                keys = [c["key"] for c in choices]
                labels = [c["description"] or c["key"] for c in choices]
            elif kind == "noul":
                symbols, keys, labels = "YN", ["true", "false"], ["예", "아니오"]
            else:
                raise ValueError(f"Unexpected kind {kind}")
            options = ", ".join(f"{symbol}={label}" for symbol, label in zip(symbols, labels))
            lines.append(f"- {name} ({kind}) {question['instructions']} 옵션: {options}")
            fields.append(json.dumps(name, ensure_ascii=False) + ":" + marker)
            outputs.append((name, keys, [meta["sym2id"][symbol] for symbol in symbols]))
        lines.extend(["", "[결정] 각 필드 심볼을 JSON으로 답하라."])
        user = "\n".join(lines)
        text = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{{{','.join(fields)}}}<|im_end|>\n"
        ids = tokenizer.encode(text, add_special_tokens=False)
        positions = [i - 1 for i, token in enumerate(ids) if token == marker_id]
        if len(positions) != len(outputs) or any(pos < 0 for pos in positions):
            raise ValueError("Marker count mismatch")
        result = dict(id=fixture["id"], ids=ids, positions=positions, answers={})
        if model is not None:
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
            for pos, (name, keys, candidates) in zip(positions, outputs):
                p = torch.softmax(logits[pos, candidates], dim=-1).tolist()
                result["answers"][name] = dict(zip(keys, p))
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--tokens-only", action="store_true")
    args = parser.parse_args()
    reference(args.checkpoint, json.loads(args.fixtures.read_text()), args.tokens_only)
