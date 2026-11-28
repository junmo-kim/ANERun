"""Accuracy reference: the original EmbeddingGemma 2 in fp32 through Hugging Face transformers, on CPU.

Builds the evaluation cases from a public corpus and image folder and writes, for every input, the fp32
reference embedding. `compare.py` then sends exactly the same inputs to a running `aneembed serve` and
reports the cosine similarity per group.

Cases (texts carry the document prompt "title: none | text: " in the string itself, so both sides
tokenize the same characters; each text is cut by characters to the largest prefix whose token count,
<bos>/<eos> included, is at most the target):
  single_S        one text per request, S in 64...2048 (every packing size of the model package): targets S
                  and S/2 + 5 on six documents
  boundary        one text per request just over a packing size: 513, 769, 1025, 1281, 1537 tokens
  packed          requests of 10...40 short texts (6...200 tokens) from every document
  long_2049_4096  one text per request at 2049, 2600, 3072, 3073, 3300 and 4096 tokens
  long_4097_8192  one text per request at 4097, 5000, 6100, 6144, 6145, 7000 and 8192 tokens
  image_single    every image alone (official 280-token budget)
  image_batch     all images in requests of up to 8

Reference per text: ids = [<bos>] + tokenizer(text) + [<eos>], last_hidden_state mean-pooled over all
tokens, L2-normalized (this equals sentence-transformers `encode`, checked below). Images: the processor
with its default 280-token budget, the same pooling over <bos> <|image> soft tokens <image|> <eos>.

Usage (the tools/convert venv; CPU only). corpus.json and the image folder come from tools/convert
(fetch.py data, corpus.py, images.py), the model from `fetch.py models`:
    python -I scripts/accuracy/reference.py --hf-dir DIR --corpus corpus.json --images DIR --out reference.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

PROMPT = "title: none | text: "
BOS, EOS = 2, 1
DOCS = (0, 1, 2, 3, 5, 6)  # en, ko, ja, zh README; ko and ja constitution (tools/convert/corpus.py order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    from tokenizers import Tokenizer
    from transformers import EmbeddingGemma2Model, AutoProcessor

    tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))
    model = EmbeddingGemma2Model.from_pretrained(args.hf_dir, dtype=torch.float32, use_safetensors=True, local_files_only=True).eval()
    proc = AutoProcessor.from_pretrained(args.hf_dir, local_files_only=True, trust_remote_code=False)
    with open(args.corpus, "rb") as source:
        corpus_bytes = source.read()
    docs = json.loads(corpus_bytes)
    legacy = any(doc.get("name") == "hf-eg2-modeling-code" for doc in docs)
    versions = {doc.get("corpus_version", "aneembed-corpus/1" if legacy else "unversioned") for doc in docs}
    if len(versions) != 1:
        raise ValueError("accuracy inputs must come from a single corpus version")
    corpus_meta = {"version": versions.pop(), "sha256": hashlib.sha256(corpus_bytes).hexdigest()}

    def ids_of(text):
        return [BOS] + tok.encode(text, add_special_tokens=False).ids + [EOS]

    def cut(doc, target, start=0):
        """Largest character prefix of doc[start:] whose ids (with the prompt) are at most `target`."""
        body = docs[doc]["text"][start:]
        lo, hi = 1, min(len(body), target * 8)
        while len(ids_of(PROMPT + body[:hi])) <= target and hi < len(body):
            hi = min(len(body), hi * 2)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(ids_of(PROMPT + body[:mid])) <= target:
                lo = mid
            else:
                hi = mid - 1
        text = PROMPT + body[:lo]
        return text, len(ids_of(text))

    def embed_ids(ids):
        with torch.no_grad():
            h = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))
        e = h.last_hidden_state[0].mean(0)
        return (e / e.norm()).numpy()

    cases = []

    def text_case(group, texts):
        refs, counts = [], []
        for text in texts:
            ids = ids_of(text)
            counts.append(len(ids))
            refs.append(embed_ids(ids).round(7).tolist())
        cases.append({"group": group, "kind": "text", "texts": texts, "tokens": counts, "reference": refs})
        print(f"{group} tokens={counts if len(counts) < 6 else f'{len(counts)} texts'}", flush=True)

    started = time.time()
    for S in (64, 128, 256, 512, 768, 1024, 1280, 1536, 2048):
        for d in DOCS:
            for target in (S, S // 2 + 5):
                text, _ = cut(d, target, start=53 * d + target)
                text_case(f"single_{S}", [text])
    for target in (513, 769, 1025, 1281, 1537):
        for d in DOCS[:2]:
            text_case("boundary", [cut(d, target, start=31 * d)[0]])
    rng = np.random.default_rng(2048)
    for request in range(8):
        count = int(rng.integers(10, 41))
        texts = []
        for j in range(count):
            d = (request * 5 + j) % len(docs)
            texts.append(cut(d, int(rng.integers(6, 201)), start=int(rng.integers(0, 4000)))[0])
        text_case("packed", texts)
    long_docs = [i for i, doc in enumerate(docs) if len(ids_of(PROMPT + doc["text"])) >= 8192]
    j = 0
    for group, targets in (("long_2049_4096", (2049, 2600, 3072, 3073, 3300, 4096)),
                           ("long_4097_8192", (4097, 5000, 6100, 6144, 6145, 7000, 8192))):
        for target in targets:
            for r in range(2):
                d = long_docs[(j + 3 * r) % len(long_docs)]
                text_case(group, [cut(d, target)[0]])
            j += 1

    from PIL import Image

    names = sorted(os.listdir(args.images))
    image_refs = {}
    for name in names:
        image = Image.open(os.path.join(args.images, name))
        x = proc(images=[image], return_tensors="pt")
        with torch.no_grad():
            h = model(**{k: x[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_position_ids")})
        e = h.last_hidden_state[0].mean(0)
        image_refs[name] = ((e / e.norm()).numpy().round(7).tolist(), int(x["input_ids"].shape[1]))
        cases.append({"group": "image_single", "kind": "image", "images": [name],
                      "tokens": [image_refs[name][1]], "reference": [image_refs[name][0]]})
        print(f"image {name} {image.size} tokens={image_refs[name][1]}", flush=True)
    for start in range(0, len(names), 8):
        chunk = names[start:start + 8]
        cases.append({"group": "image_batch", "kind": "image", "images": chunk,
                      "tokens": [image_refs[n][1] for n in chunk], "reference": [image_refs[n][0] for n in chunk]})

    # The pipeline above equals sentence-transformers end to end (prompt handling, pooling, normalization).
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(args.hf_dir, device="cpu", local_files_only=True, trust_remote_code=False,
                             model_kwargs={"dtype": torch.float32, "use_safetensors": True})
    checks = []
    for case in cases[:12:3]:
        e = st.encode([case["texts"][0]], prompt="", normalize_embeddings=True)[0]
        checks.append(float(np.dot(e, case["reference"][0])))
    print(f"sentence-transformers cross-check min cos {min(checks):.7f}", flush=True)

    meta = {"format": "aneembed-accuracy/1", "corpus": corpus_meta, "prompt": PROMPT, "precision": "fp32", "device": "cpu",
            "torch": torch.__version__, "sentence_transformers_check_min_cos": min(checks),
            "seconds": round(time.time() - started, 1), "cases": cases}
    json.dump(meta, open(args.out, "w"))
    print(f"{len(cases)} cases -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
