"""Warm HTTP latency of a running `anerun serve` (EmbeddingGemma 2) for the README comparison rows. Usage:
    http_bench.py URL CORPUS IMAGE OUT.json [--rounds 5] [--per 20] [--skip-8192]
Each case: 3 discarded warm-up calls, then ROUNDS rounds of PER calls; reports the median of round medians and
the spread (min/max of round medians). Texts are cut to exactly N tokens (prompt, <bos>, <eos> included), checked
against usage.prompt_tokens."""
import json, sys, time, base64, statistics as st, urllib.request, threading
url, corpus, image, out = sys.argv[1:5]
args = sys.argv[5:]
ROUNDS = int(args[args.index("--rounds") + 1]) if "--rounds" in args else 5
PER = int(args[args.index("--per") + 1]) if "--per" in args else 20
def post(inp, timeout=600):
    r = urllib.request.Request(url + "/v1/embeddings", data=json.dumps({"model": "embeddinggemma-2", "input": inp}).encode(), headers={"Content-Type": "application/json"})
    s = time.perf_counter(); d = json.load(urllib.request.urlopen(r, timeout=timeout)); return time.perf_counter() - s, d["usage"]["prompt_tokens"]
docs = [d["text"] for d in json.load(open(corpus))]
text = "\n\n".join(docs)
def cut(n):
    lo, hi = 1, len(text)
    while lo < hi:                       # largest prefix with token count <= n
        mid = (lo + hi + 1) // 2; _, t = post("title: none | text: " + text[:mid])
        lo, hi = (mid, hi) if t <= n else (lo, mid - 1)
    s = "title: none | text: " + text[:lo]; _, t = post(s); return s, t
def measure(inp, per=PER, rounds=ROUNDS):
    for _ in range(3): post(inp)
    meds = []
    for _ in range(rounds):
        meds.append(st.median(post(inp)[0] for _ in range(per)))
    return {"median_ms": 1000 * st.median(meds), "round_min_ms": 1000 * min(meds), "round_max_ms": 1000 * max(meds)}
res = {}
sizes = [32, 512, 2048] + ([] if "--skip-8192" in args else [8192])
cuts = {n: cut(n) for n in sizes}
for n, (s, t) in cuts.items():
    res[f"{n} tokens"] = dict(measure(s, per=PER if n < 8192 else 5), tokens=t); print(n, res[f"{n} tokens"], flush=True)
img = {"content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(image, "rb").read()).decode()}}]}
res["image"] = dict(measure([img], per=10), tokens=post([img])[1]); print("image", res["image"], flush=True)
res["10 x 32"] = measure([cuts[32][0]] * 10); print("10x32", res["10 x 32"], flush=True)
res["10 x 2048"] = measure([cuts[2048][0]] * 10, per=3); print("10x2048", res["10 x 2048"], flush=True)
def clients(k=8, n=256):
    q = cuts[32][0]; done = []; lock = threading.Lock()
    def worker(m):
        for _ in range(m): post(q)
    for _ in range(3): post(q)
    rates = []
    for _ in range(ROUNDS):
        ts = [threading.Thread(target=worker, args=(n // k,)) for _ in range(k)]; s = time.perf_counter()
        [t.start() for t in ts]; [t.join() for t in ts]; rates.append(n / (time.perf_counter() - s))
    return {"req_per_s": st.median(rates), "round_min": min(rates), "round_max": max(rates)}
res["8 clients x 256 queries"] = clients(); print("clients", res["8 clients x 256 queries"], flush=True)
json.dump(res, open(out, "w"), indent=1)
