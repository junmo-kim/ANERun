"""Compare the local Swift HTTP API to saved HF Qwen CPU reference vectors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import load_sources
from pplx_verify import MIN_COSINE, swift_binary, swift_environment, wait_for_health
from qwen_reference import DIM, INSTRUCTION

MODEL = "qwen3-embedding-0.6b"


def verify(args):
    data = json.loads((args.reference / "parity.json").read_text())
    blob = (args.reference / "parity.bin").read_bytes()
    source = load_sources()["qwen"]
    if (data["format"] != "anerun-qwen-parity/1" or data["source"]["revision"] != source["revision"]
            or data["source"]["sha256"] != source["sha256"]
            or hashlib.sha256(blob).hexdigest() != data["binary_sha256"]):
        raise ValueError("reference identity or checksum mismatch")
    cases = data["cases"]
    if args.cap == 32768 and not any(c["token_count"] == 32768 and "offset" in c for c in cases):
        raise ValueError("full-cap verification requires the independent exact-32K reference")
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if manifest["model"] != MODEL or manifest["revision"] != source["revision"]:
        raise ValueError("package identity or revision mismatch")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(endpoint, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url + endpoint, data=body, headers={"Content-Type": "application/json"})
        with client.open(req, timeout=args.timeout) as response:
            return json.load(response)

    def embed(case, dimensions=DIM, *, truncate=True):
        payload = {"model": MODEL, "input": case["text"], "dimensions": dimensions, "truncate": truncate}
        if case["id"] == "query":
            prefix = f"Instruct: {INSTRUCTION}\nQuery:"
            if not case["text"].startswith(prefix):
                raise ValueError("query fixture is missing the pinned prefix")
            payload.update(input=case["text"][len(prefix):], input_type="query")
        return request("/v1/embeddings", payload)

    with TemporaryDirectory(prefix="anerun-qwen-verify-") as data_dir:
        started = time.monotonic()
        server = subprocess.Popen(
            [swift_binary(args.binary), "serve", "--model-dir", str(args.bundle.resolve()),
             "--data-dir", data_dir, "--host", "127.0.0.1", "--port", str(port), "--max-tokens", str(args.cap)],
            env=swift_environment(), stdout=sys.stderr, stderr=sys.stderr,
        )
        try:
            wait_for_health(server, client, url, args.timeout)
            model = request("/v1/models")["data"][0]
            print(json.dumps({"loaded_s": time.monotonic() - started, "pid": server.pid,
                              "context": model["max_context_length"]}), flush=True)
            if model["id"] != MODEL or model["max_context_length"] != args.cap or model["max_image_tokens"] != 0:
                raise ValueError("unexpected model capabilities")
            for case in cases:
                count = case["token_count"]
                ids = request("/tokenize", {"content": case["text"], "add_special": True, "parse_special": True})
                if ids["tokens"] != case["ids"] or len(ids["tokens"]) != count:
                    raise ValueError(f"{case['id']}: token mismatch")
                if count > args.cap:
                    try:
                        embed(case, truncate=False)
                    except urllib.error.HTTPError as error:
                        with error:
                            detail = json.load(error)
                        if detail["error"]["code"] != "context_length_exceeded":
                            raise ValueError(f"unexpected rejection: {detail}") from error
                        print(json.dumps({"id": case["id"], "rejected_tokens": count, "status": error.code}), flush=True)
                        continue
                    raise ValueError(f"{case['id']}: overflow accepted")
                for dim in ([DIM, 37] if case["id"] == "query" else [DIM]):
                    start = time.monotonic()
                    response = embed(case, dim)
                    elapsed = time.monotonic() - start
                    actual = np.asarray(response["data"][0]["embedding"], dtype=np.float64)
                    raw = np.frombuffer(blob, dtype="<f4", count=DIM, offset=case["offset"])[:dim].astype(np.float64)
                    if (response["usage"]["prompt_tokens"] != count or actual.shape != (dim,)
                            or not np.isfinite(actual).all() or not np.isfinite(raw).all()):
                        raise ValueError(f"{case['id']}: invalid output")
                    norm = float(np.linalg.norm(actual))
                    cosine = float((actual @ raw) / (norm * np.linalg.norm(raw)))
                    print(json.dumps({"id": case["id"], "tokens": count, "dimensions": dim, "seconds": elapsed,
                                      "norm": norm, "cosine": cosine, "embedding": actual.tolist()}), flush=True)
                    if not math.isclose(norm, 1, abs_tol=1e-4) or not cosine >= MIN_COSINE:
                        raise ValueError(f"{case['id']}: norm {norm}, cosine {cosine}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--binary")
    parser.add_argument("--cap", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if not 1 <= args.cap <= 32768 or not 0 < args.timeout < math.inf:
        parser.error("invalid cap or timeout")
    verify(args)
