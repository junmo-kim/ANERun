"""HF-anchored checks (fp32 transformers / sentence-transformers on CPU).

Usage: verify_hf.py anchor <dist> <corpus.json> --hf-dir <snapshot>   eg2.Reference vs HF EmbeddingGemma2Model
                                                                      (text, 64..8192 tokens)
       verify_hf.py images <dist> <img_dir> --hf-dir <snapshot>       ANE image path vs HF fp32 image embeddings
                                                                      (budgets 70/140/280)
corpus.json and the image folder come from corpus.py / images.py, the snapshot from fetch.py models.

Image path under test (host contract): HF processor pixels (Swift ports Gemma4ImageProcessor) -> vision_B on
CPU+ANE -> <bos><|image> tokens <image|><eos> -> embed_S / winp on CPU+ANE (single, and packed with other images/texts).
Reference: HF EmbeddingGemma2Model fp32 forward on the same processor outputs, mean pool over all tokens, L2."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("anchor", "images"))
ap.add_argument("dist")
ap.add_argument("data", help="corpus.json (anchor) or image folder (images)")
ap.add_argument("--hf-dir", required=True, help="google/embeddinggemma-2 snapshot")
args = ap.parse_args()
torch.set_num_threads(8)
mode, dist, HF = args.mode, args.dist, args.hf_dir
from transformers import EmbeddingGemma2Model, AutoProcessor  # noqa: E402

model = EmbeddingGemma2Model.from_pretrained(HF, dtype=torch.float32, use_safetensors=True, local_files_only=True).eval()
proc = AutoProcessor.from_pretrained(HF, local_files_only=True, trust_remote_code=False)
res = {}


def add(cat, c):
    res.setdefault(cat, []).append(c)


def summary():
    s = {k: {"n": len(v), "min": round(min(v), 6), "median": round(float(np.median(v)), 6)} for k, v in res.items()}
    print("SUMMARY", json.dumps(s), flush=True)


def hf_embed(inputs):
    with torch.no_grad():
        h = model(**inputs).last_hidden_state[0]
    m = inputs["attention_mask"][0].float()[:, None]
    e = (h * m).sum(0) / m.sum()
    return (e / e.norm()).numpy()


if mode == "anchor":
    from eg2 import Weights, ref_embed
    W = Weights(HF)
    tab = host.TokenTable(f"{dist}/embeddings.bf16")
    tok = host.Tokenizer(f"{dist}/tokenizer.json")
    docs = json.load(open(args.data))
    for d, n in ((0, 64), (1, 300), (2, 513), (3, 1500), (5, 2048), (6, 3000), (7, 5000), (8, 8192)):
        ids = tok.ids(docs[d]["text"])
        ids = ids[:n - 1] + [host.EOS]
        e_hf = hf_embed({"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})
        c = host.cos(e_hf, ref_embed(W, tab.rows(ids)))
        add("reference_vs_hf", c)
        print(f"anchor {docs[d]['name']} n={len(ids)} cos(Reference fp32, HF fp32)={c:.7f}", flush=True)
    # sentence-transformers end to end (tokenizer + prompt + pooling) for two short texts
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(HF, device="cpu", local_files_only=True, trust_remote_code=False,
                             model_kwargs={"dtype": torch.float32, "use_safetensors": True})
    for t in ("The quick brown fox jumps over the lazy dog.", "대한민국은 민주공화국이다. 대한민국의 주권은 국민에게 있다."):
        ids = tok.ids(t)
        e_st = st.encode([t], prompt=host.DOC_PREFIX, normalize_embeddings=True)[0]
        c = host.cos(e_st, ref_embed(W, tab.rows(ids)))
        add("reference_vs_sentence_transformers", c)
        print(f"anchor ST '{t[:20]}' n={len(ids)} cos={c:.7f}", flush=True)
    summary()

if mode == "images":
    import coremltools as ct
    from PIL import Image
    tab = host.TokenTable(f"{dist}/embeddings.bf16")
    tok = host.Tokenizer(f"{dist}/tokenizer.json")
    vis = host.Vision(f"{dist}/position_embeddings.f16")
    CU = ct.ComputeUnit.CPU_AND_NE
    vfn = {b: ct.models.CompiledMLModel(f"{dist}/EmbeddingGemma2Vision.mlmodelc", compute_units=CU, function_name=f"vision_{b}")
           for b in (70, 140, 280)}
    tfn = {}

    def bucket(S):
        if S not in tfn:
            tfn[S] = ct.models.CompiledMLModel(f"{dist}/EmbeddingGemma2Text.mlmodelc", compute_units=CU,
                                               function_name=host.function_name(S))
        return tfn[S]

    paths = sorted(glob.glob(f"{args.data}/*"))
    seqs, refs = {}, {}
    for p in paths:
        img = Image.open(p)
        name = os.path.basename(p)
        for b in (70, 140, 280):
            x = proc(images=[img], return_tensors="pt", images_kwargs={"max_soft_tokens": b})
            ref = hf_embed({k: x[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_position_ids")})
            with torch.no_grad():
                f_hf = model.get_image_features(pixel_values=x["pixel_values"], image_position_ids=x["image_position_ids"]).pooler_output[0].numpy()
            feed, ntok = vis.feed(x["pixel_values"][0].numpy(), x["image_position_ids"][0].numpy(), b)
            assert ntok == f_hf.shape[0] and x["input_ids"].shape[1] == ntok + 4
            t_ane = vfn[b].predict(feed)["image_tokens"][0, :ntok].astype(np.float32)
            cv = host.cos(t_ane, f_hf)
            add(f"vision_tokens_{b}", cv)
            rows = host.image_sequence(tab, t_ane)
            S = host.bucket_for(len(rows))
            e = bucket(S).predict(host.pack([rows], S))["embedding"][0]
            c = host.cos(e, ref)
            add(f"image_single_{b}", c)
            # text half only: HF fp32 vision tokens through the ANE text bucket
            e2 = bucket(S).predict(host.pack([host.image_sequence(tab, f_hf)], S))["embedding"][0]
            add(f"image_textpart_{b}", host.cos(e2, ref))
            seqs[(name, b)], refs[(name, b)] = rows, ref
            print(f"{name} {img.size} budget={b} tokens={ntok} S={S} cos(vision tokens)={cv:.6f} "
                  f"cos(image emb, HF fp32)={c:.6f} text-part={host.cos(e2, ref):.6f}", flush=True)
    names = sorted({k[0] for k in seqs})
    doc = tok.ids("Embedding images and texts together in one packed call: " * 4)
    trow = tab.rows(doc)
    t_ref = None
    for j in range(len(names)):
        a, b2, c3 = names[j], names[(j + 1) % len(names)], names[(j + 2) % len(names)]
        for S, items in ((1024, [(a, 280), (b2, 140), (c3, 70)]), (2048, [(a, 280), (b2, 280), (c3, 280), (a, 70)])):
            rows = [seqs[k] for k in items] + [trow]
            out = bucket(S).predict(host.pack(rows, S))["embedding"]
            cs = [host.cos(out[k], refs[it]) for k, it in enumerate(items)]
            for c in cs:
                add(f"image_packed_S{S}", c)
            print(f"packed S={S} {items} cos={[round(c, 6) for c in cs]}", flush=True)
    # cross-check: HF reference pipeline == sentence-transformers encode (default 280 budget)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(HF, device="cpu", local_files_only=True, trust_remote_code=False,
                             model_kwargs={"dtype": torch.float32, "use_safetensors": True})
    for p in paths[:2]:
        try:
            e_st = st.encode([Image.open(p)], normalize_embeddings=True)[0]
        except Exception as ex:  # noqa: BLE001
            print("ST image encode failed:", repr(ex)[:200], flush=True)
            continue
        c = host.cos(e_st, refs[(os.path.basename(p), 280)])
        add("hf_reference_vs_sentence_transformers_280", c)
        print(f"ST cross-check {os.path.basename(p)} cos(ST, HF reference)={c:.7f}", flush=True)
    summary()
