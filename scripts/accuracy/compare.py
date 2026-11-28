"""Accuracy of a running `aneembed serve` against the fp32 reference written by reference.py.

Sends every case through the HTTP API exactly as a client would (texts as strings, images as data URLs in
one request per case), then reports the cosine similarity to the reference per group (n, min, median, mean)
and whether usage.prompt_tokens equals the reference token count.

Standard library only:
    aneembed serve --max-tokens 8192 --port 18188 &
    python3 -I scripts/accuracy/compare.py --url http://127.0.0.1:18188 --reference reference.json --images DIR
"""
import argparse
import base64
import json
import math
import statistics
import sys
import time
import urllib.request


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18180")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--groups", default="", help="comma-separated subset of groups")
    parser.add_argument("--out", default="", help="write per-case results as JSON")
    args = parser.parse_args()
    reference = json.load(open(args.reference))
    only = set(filter(None, args.groups.split(",")))
    endpoint = args.url.rstrip("/") + "/v1/embeddings"
    groups, rows, token_mismatches = {}, [], 0
    started = time.time()
    for case in reference["cases"]:
        if only and case["group"] not in only:
            continue
        if case["kind"] == "text":
            body = {"input": case["texts"]}
        else:
            items = []
            for name in case["images"]:
                data = base64.b64encode(open(f"{args.images}/{name}", "rb").read()).decode()
                ext = name.rsplit(".", 1)[-1].lower()
                mime = "image/" + {"jpg": "jpeg", "tif": "tiff"}.get(ext, ext)
                items.append({"content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]})
            body = {"input": items}
        response = post(endpoint, body)
        vectors = [item["embedding"] for item in sorted(response["data"], key=lambda item: item["index"])]
        cosines = [cosine(v, r) for v, r in zip(vectors, case["reference"])]
        tokens_ok = response["usage"]["prompt_tokens"] == sum(case["tokens"])
        token_mismatches += not tokens_ok
        groups.setdefault(case["group"], []).extend(cosines)
        rows.append({"group": case["group"], "tokens": case["tokens"], "cos": cosines, "tokens_match": tokens_ok})
    summary = {}
    for group, values in groups.items():
        summary[group] = {"n": len(values), "min": round(min(values), 6), "median": round(statistics.median(values), 6),
                          "mean": round(statistics.fmean(values), 6)}
        print(f"{group:16s} n={len(values):4d} min={min(values):.6f} median={statistics.median(values):.6f} "
              f"mean={statistics.fmean(values):.6f}")
    print(f"token count mismatches: {token_mismatches} of {len(rows)} requests; {time.time() - started:.1f}s")
    if args.out:
        report = {"summary": summary, "token_mismatches": token_mismatches, "cases": rows}
        if "corpus" in reference:
            report["corpus"] = reference["corpus"]
        with open(args.out, "w") as file:
            json.dump(report, file)
    sys.exit(1 if token_mismatches else 0)


if __name__ == "__main__":
    main()
