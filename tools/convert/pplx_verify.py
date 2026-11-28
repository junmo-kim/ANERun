"""Use the Swift service to check pplx token counts and quantized embeddings."""
import json
import os
from pathlib import Path
import shutil
import socket
from statistics import median
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

from pplx_build import MODEL, check_source
from pplx_reference import means, quantize, tokenizer

MIN_COSINE = 0.995
MAX_TOKENS = 32768


def swift_binary(binary):
    if binary:
        return str(Path(binary).expanduser().resolve())
    repo = Path(__file__).resolve().parents[2]
    for configuration in ("release", "debug"):
        candidate = repo / ".build" / configuration / "anerun"
        if candidate.is_file():
            return str(candidate)
    if installed := shutil.which("anerun"):
        return installed
    raise SystemExit("build the Swift binary with swift build -c release, or pass --binary PATH")


def swift_environment():
    return {key: value for key, value in os.environ.items() if not key.startswith(("ANERUN_", "ANEEMBED_"))}


def load_rows(corpus, hf_dir):
    docs = json.loads(Path(corpus).read_text(encoding="utf-8"))
    if not isinstance(docs, list) or not docs:
        raise ValueError("pplx corpus must be a nonempty JSON list")
    tok = tokenizer(hf_dir)
    rows, needs_mean = [], []
    for index, doc in enumerate(docs):
        if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
            raise ValueError(f"corpus row {index} requires a string text")
        name = doc.get("name", str(index))
        text = doc["text"]
        ids = tok.encode(text, add_special_tokens=False).ids
        declared = doc.get("tokens", doc.get("ids", len(ids)))
        if isinstance(declared, list):
            matched = declared == ids
        else:
            matched = type(declared) is int and declared == len(ids)
        if not 1 <= len(ids) <= MAX_TOKENS or not matched:
            raise ValueError(f"{name}: {len(ids)} HF tokens, expected matching IDs/count within 1..{MAX_TOKENS}")
        row = {"name": name, "text": text, "tokens": len(ids), "mean": doc.get("vector")}
        if "vector" not in doc:
            needs_mean.append((row, ids))
        rows.append(row)
    if needs_mean:
        refs = means(hf_dir, (ids for _, ids in needs_mean))
        for (row, _), ref in zip(needs_mean, refs, strict=True):
            row["mean"] = ref
    for row in rows:
        ref = np.asarray(row.pop("mean"), dtype=np.float32)
        if ref.shape != (1024,) or not np.isfinite(ref).all():
            raise ValueError(f"{row['name']}: expected 1024 finite raw fp32 coordinates")
        expected = quantize(ref).astype(np.float64)
        if not np.any(expected):
            raise ValueError(f"{row['name']}: expected a nonzero quantized vector")
        row["expected"] = expected
    return rows


def wait_for_health(server, client, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Swift server exited {server.returncode} before readiness")
        try:
            with client.open(url + "/health", timeout=2) as response:
                status = json.load(response)
            if status.get("status") == "ok":
                return
            raise RuntimeError(f"Swift server reported {status}")
        except urllib.error.HTTPError as error:
            with error:
                status = json.load(error)
            if error.code != 503 or status.get("status") != "loading":
                raise RuntimeError(f"Swift server reported {status}") from error
        except (TimeoutError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f"Swift server did not become ready in {timeout:g}s")


def check_rows(rows, client, url, timeout):
    scores = []
    for row in rows:
        request = urllib.request.Request(
            url + "/v1/embeddings",
            data=json.dumps({"model": MODEL, "input": row["text"], "encoding_format": "float"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with client.open(request, timeout=timeout) as response:
            body = json.load(response)
        actual = np.asarray(body["data"][0]["embedding"], dtype=np.float64)
        count = body["usage"]["prompt_tokens"]
        if (count != row["tokens"] or actual.shape != (1024,) or not np.isfinite(actual).all()
                or not np.all((actual == np.rint(actual)) & (actual >= -128) & (actual <= 127))
                or not np.any(actual)):
            raise ValueError(f"{row['name']}: invalid Swift quantized vector or token count {count}")
        expected = row["expected"]
        cosine = float(actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        scores.append(cosine)
        print(f"{row['name']} tokens={count} cosine={cosine:.6f}", flush=True)
    print("SUMMARY", json.dumps({"n": len(scores), "min": min(scores), "median": median(scores),
                                  "minimum_required": MIN_COSINE}), flush=True)
    return 0 if min(scores) >= MIN_COSINE else 1


def run(dist, corpus, hf_dir, binary, timeout):
    binary = swift_binary(binary)
    check_source(hf_dir)
    rows = load_rows(corpus, hf_dir)
    cap = max(row["tokens"] for row in rows)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    url = f"http://127.0.0.1:{port}"
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryDirectory(prefix="pplx-verify-") as data_dir:
        log_path = Path(data_dir) / "server.log"
        with log_path.open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [binary, "serve", "--model-dir", str(Path(dist).resolve()), "--data-dir", data_dir,
                 "--host", "127.0.0.1", "--port", str(port), "--max-tokens", str(cap)],
                env=swift_environment(), stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                wait_for_health(server, client, url, timeout)
                return check_rows(rows, client, url, timeout)
            except (OSError, ValueError, RuntimeError) as error:
                log_tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
                raise SystemExit(f"{error}\n{log_tail}") from error
            finally:
                if server.poll() is None:
                    server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
