#!/usr/bin/env python3
"""Check on-demand model loading and the lifetime of a dedicated playground server."""
import http.client
import json
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def playground(binary, *args, idle=30):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [binary, "playground", "--no-open", "--host", "127.0.0.1", "--port",
               str(port), "--idle-timeout", str(idle), *args]
    process = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                status, body = call(port, "GET", "/playground/config")
                assert status == 200, (status, body)
                break
            except ConnectionError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        yield process, port, json.loads(body)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)


def call(port, method, path, payload=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        body = json.dumps(payload) if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body, headers)
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def main(binary):
    with tempfile.TemporaryDirectory() as tmp:
        # A recognized header with no model assets can display the UI on any build.
        directory = Path(tmp)
        (directory / "model.json").write_text(json.dumps({"source": {
            "repo": "Valen-Team/Valen-Decision-1.5-4B",
            "revision": "9b0cacc976d7fb72f252ee1adb1564f5f15e462c"}}))
        with playground(binary, "--model-dir", tmp, idle=2) as (server, port, config):
            assert config["kind"] == "decision" and config["images"] is True, config
            assert call(port, "GET", "/playground")[0] == 200
            assert server.wait(timeout=10) == 0

        with playground(binary, "--model", "embeddinggemma-2", "--runtime", "fake",
                        "--data-dir", tmp) as (server, port, config):
            assert config["kind"] == "embedding", config
            status, body = call(port, "POST", "/v1/embeddings", {"input": "test"})
            assert status == 200 and json.loads(body)["data"][0]["embedding"], body
            assert call(port, "POST", "/playground/stop", {})[0] in (200, 202, 204)
            assert server.wait(timeout=10) == 0


if __name__ == "__main__":
    main(sys.argv[1])
