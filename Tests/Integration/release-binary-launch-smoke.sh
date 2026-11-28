#!/usr/bin/env bash
set -euo pipefail

# Contracts that can regress even when a release build compiles and links: launch, version, help,
# doctor output, structured errors and exit codes, stdio MCP, and the HTTP listener lifecycle.
# Uses the fake runtime, so it needs neither a model nor a Neural Engine (CI runners have neither).
#
# Usage: Tests/Integration/release-binary-launch-smoke.sh [path-to-binary]

binary=${1:-.build/release/anerun}
binary=$(cd "$(dirname "$binary")" && pwd)/$(basename "$binary")
temp_root=${TMPDIR:-/tmp}
tmpdir=$(mktemp -d "${temp_root%/}/anerun-smoke.XXXXXX")
pid=""
cleanup() {
  if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  rm -rf "$tmpdir"
}
trap cleanup EXIT
fail() { echo "smoke test failed: $*" >&2; exit 1; }
unset ANERUN_MODEL ANERUN_MODEL_DIR ANEEMBED_MODEL ANEEMBED_MODEL_DIR
unset ANERUN_RUNTIME ANERUN_DATA_DIR ANEEMBED_RUNTIME ANEEMBED_DATA_DIR
data_dir="$tmpdir/data"
export ANERUN_RUNTIME=fake ANERUN_DATA_DIR="$data_dir"

if [[ ${ANERUN_CHECK_RELEASE_COREAI:-0} == 1 ]]; then
  # Conflicting source selectors reach the dispatcher without network access or model execution.
  if "$binary" models install --model valen-4b --from "$tmpdir" --repo invalid/example --runtime coreml --data-dir "$data_dir" --format json \
      >"$tmpdir/decision.out" 2>"$tmpdir/decision.err"; then
    fail "install with no decision source succeeded"
  fi
  expected=coreai_unavailable
  if (( $(sw_vers -productVersion | cut -d. -f1) >= 27 )); then expected=invalid_model_source; fi
  python3 -c 'import json, sys; assert json.load(sys.stdin)["reason"] == sys.argv[1]' \
    "$expected" < "$tmpdir/decision.err" || fail "unexpected decision dispatcher response"
  if [[ "$expected" == invalid_model_source ]]; then
    # An installed v0.3.0 backbone-only Valen descriptor must fail before serve starts listening.
    python3 - "$binary" "$tmpdir" <<'PYLEGACY'
import json, pathlib, subprocess, sys
binary, root = sys.argv[1], pathlib.Path(sys.argv[2]) / "legacy-valen"
root.mkdir()
model = dict(source=dict(repo="Valen-Team/Valen-Decision-1.5-4B", revision="9b0cacc976d7fb72f252ee1adb1564f5f15e462c"),
             architecture="h16g", assets=[], execution=[], stage_kinds=[], hidden=2560, width=128, context=4096,
             tiers=[4096], rotary_dim=64, heads=1, kv_heads=1, head_dim=64, block_mask=False)
(root / "model.json").write_text(json.dumps(model))
run = subprocess.run([binary, "serve", "--model-dir", str(root), "--port", "58311", "--runtime", "coreml",
                      "--format", "json"], capture_output=True, timeout=5)
error = json.loads(run.stderr)
assert run.returncode and error["reason"] == "invalid_bundle", error
assert "Legacy Valen" in error["detail"] and "models install --model valen-4b" in error["detail"], error
PYLEGACY
  fi
fi

version=$("$binary" --version)
[[ "$version" =~ ^anerun\ [0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "unexpected version output: $version"
"$binary" --help > "$tmpdir/help"
grep -q "^anerun " "$tmpdir/help" || fail "unexpected help heading"
grep -q "models install --model ID" "$tmpdir/help" || fail "install model selector missing from help"
for invocation in "embed x" "serve" "mcp" "similarity a b" "tokenize x" "bench" "models path" "models verify" "models warm" "models install" "doctor --verify"; do
  read -r -a command <<< "$invocation"
  set +e
  if [[ "${command[0]}" == serve ]]; then
    # A selection regression can start listening. Isolate its port and bound the check.
    python3 - "$binary" "$tmpdir/missing.out" "$tmpdir/missing.err" <<'PYSERVE'
import socket, subprocess, sys
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = str(sock.getsockname()[1])
with open(sys.argv[2], "wb") as out, open(sys.argv[3], "wb") as err:
    try:
        result = subprocess.run([sys.argv[1], "serve", "--port", port, "--format", "json"],
                                stdout=out, stderr=err, timeout=5)
    except subprocess.TimeoutExpired:
        sys.exit(124)
sys.exit(result.returncode)
PYSERVE
    code=$?
  elif [[ "${command[0]}" == mcp ]]; then
    "$binary" mcp </dev/null >"$tmpdir/missing.out" 2>"$tmpdir/missing.err"; code=$?
  else
    "$binary" "${command[@]}" --format json </dev/null >"$tmpdir/missing.out" 2>"$tmpdir/missing.err"; code=$?
  fi
  set -e
  [ "$code" -eq 64 ] && [ ! -s "$tmpdir/missing.out" ] || fail "$invocation without a model: exit $code"
  if [[ "${command[0]}" == mcp ]]; then
    grep -q 'bad_request' "$tmpdir/missing.err" && grep -q 'missing_model' "$tmpdir/missing.err" &&
      grep -q 'pplx-embed-v1-0.6b' "$tmpdir/missing.err" || fail "mcp missing model YAML"
    continue
  fi
  python3 - "$tmpdir/missing.err" <<'PYERR'
import json, sys
error = json.load(open(sys.argv[1]))
assert error["error"] == "bad_request" and error["reason"] == "missing_model", error
assert "embeddinggemma-2" in error["detail"] and "pplx-embed-v1-0.6b" in error["detail"], error
assert "--model" in error["hint"], error
PYERR
done

"$binary" serve --print-config --format json >"$tmpdir/config.json"
python3 - "$tmpdir/config.json" <<'PYCONFIG'
import json, sys
config = json.load(open(sys.argv[1]))
for name in ["model", "model-dir", "model-name"]:
    assert config[name]["value"] == "", config[name]
PYCONFIG
"$binary" models install --runtime coreml --print-config --format json >"$tmpdir/config.json"
python3 - "$tmpdir/config.json" <<'PYCONFIG'
import json, sys
config = json.load(open(sys.argv[1]))
assert config["model"]["value"] == "", config["model"]
PYCONFIG
"$binary" models install --model embeddinggemma-2 --runtime coreml --print-config --format json >"$tmpdir/config.json"
python3 - "$tmpdir/config.json" <<'PYCONFIG'
import json, sys
config = json.load(open(sys.argv[1]))
assert config["model"]["value"] == "embeddinggemma-2", config["model"]
assert config["model-name"]["value"] == "embeddinggemma-2", config["model-name"]
assert config["model-dir"]["value"] == "", config["model-dir"]
PYCONFIG
code=0
"$binary" models install --model-dir "$tmpdir/absent-package" --runtime coreml --print-config --format json \
  >"$tmpdir/config.json" 2>"$tmpdir/config.err" || code=$?
[ "$code" -eq 64 ] && [ ! -s "$tmpdir/config.json" ] && grep -q 'missing_model' "$tmpdir/config.err" ||
  fail "print-config install with a directory selector: exit $code"
[ ! -e "$data_dir" ] || fail "printing install configuration created the data directory"
for bad in unknown_model conflicting_options; do
  case "$bad" in
    unknown_model) flags=(--model unknown);;
    conflicting_options) flags=(--model embeddinggemma-2 --model-dir "$tmpdir/other");;
  esac
  set +e
  "$binary" serve --print-config "${flags[@]}" --format json >"$tmpdir/config.json" 2>"$tmpdir/config.err"
  code=$?
  set -e
  [ "$code" -eq 64 ] && [ ! -s "$tmpdir/config.json" ] && grep -q "$bad" "$tmpdir/config.err" ||
    fail "print-config $bad: exit $code"
done
export ANERUN_MODEL=embeddinggemma-2

"$binary" serve --print-config --format json >"$tmpdir/config.json"
python3 - "$tmpdir/config.json" <<'PYCONFIG'
import json, sys
config = json.load(open(sys.argv[1]))
assert config["model-name"]["value"] == "embeddinggemma-2", config["model-name"]
assert config["model-dir"]["value"] == "", config["model-dir"]
PYCONFIG

legacy="$data_dir/models/legacy-revision"
mkdir -p "$legacy"
printf '%s\n' '{"name":"embeddinggemma-2-ane","files":[]}' >"$legacy/manifest.json"
printf '%s\n' legacy-revision >"$data_dir/models/current"
model_path=$("$binary" models path --runtime coreml)
[[ "$model_path" == */models/legacy-revision ]] || fail "unexpected legacy model path: $model_path"

package="$data_dir/models/embeddinggemma-2/selection-fixture"
mkdir -p "$package"
printf '%s\n' '{"name":"embeddinggemma-2-ane","files":[]}' >"$package/manifest.json"
printf '%s\n' selection-fixture >"$(dirname "$package")/current"
"$binary" serve --runtime coreml --print-config --format json >"$tmpdir/config.json"
python3 - "$tmpdir/config.json" <<'PYCONFIG'
import json, sys
config = json.load(open(sys.argv[1]))
assert config["model-name"]["value"] == "embeddinggemma-2", config["model-name"]
assert config["model-dir"]["value"].endswith("/models/embeddinggemma-2/selection-fixture"), config["model-dir"]
PYCONFIG

code=0
"$binary" bench --model pplx-embed-v1-0.6b --image "$tmpdir/absent.jpg" --format json \
  >"$tmpdir/bench.out" 2>"$tmpdir/bench.err" || code=$?
[ "$code" -eq 65 ] && [ ! -s "$tmpdir/bench.out" ] && grep -q 'images_unsupported' "$tmpdir/bench.err" ||
  fail "pplx benchmark with an image: exit $code"

"$binary" doctor --format json >"$tmpdir/doctor.json" 2>"$tmpdir/doctor.err" || fail "doctor exited nonzero"
python3 - "$tmpdir/doctor.json" <<'PY' || fail "doctor output failed validation"
import json, sys
doctor = json.load(open(sys.argv[1]))
missing = {"aneembed", "macos", "macos_build", "neural_engine", "model", "compile_cache", "ready", "problems"} - doctor.keys()
if missing:
    raise SystemExit(f"doctor is missing {sorted(missing)}")
PY

"$binary" embed "smoke test" --dimensions 128 --format json >"$tmpdir/embed.json" 2>/dev/null || fail "embed exited nonzero"
python3 - "$tmpdir/embed.json" <<'PY' || fail "embed output failed validation"
import json, math, sys
out = json.load(open(sys.argv[1]))
vector = out["data"][0]["embedding"]
assert len(vector) == 128 and abs(math.sqrt(sum(x * x for x in vector)) - 1) < 1e-4, out
PY

set +e
"$binary" embed x --dimensions 300 >/dev/null 2>"$tmpdir/err"; code=$?
set -e
[ "$code" -eq 64 ] && grep -q 'invalid_dimensions' "$tmpdir/err" || fail "bad --dimensions: exit $code"
long_text=$(head -c 3000 /dev/zero | tr '\0' 'x')
"$binary" embed x "$long_text" --dimensions 128 --format json >"$tmpdir/truncated.json" 2>"$tmpdir/err" ||
  fail "permissive embed exited nonzero"
printf '"x"\n{"text":"%s","id":"long"}\n' "$long_text" |
  "$binary" embed --stdin --dimensions 128 >"$tmpdir/truncated.jsonl" 2>"$tmpdir/err" ||
  fail "permissive embed --stdin exited nonzero"
"$binary" similarity x "$long_text" --format json >"$tmpdir/similarity.json" 2>"$tmpdir/err" ||
  fail "permissive similarity exited nonzero"
"$binary" tokenize "$long_text" --format json >"$tmpdir/tokenize.json" 2>"$tmpdir/err" ||
  fail "tokenize exited nonzero"
python3 - "$tmpdir" <<'PY' || fail "truncation outputs failed validation"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
embed = json.loads((root / "truncated.json").read_text())["data"]
jsonl = [json.loads(line) for line in (root / "truncated.jsonl").read_text().splitlines()]
similarity = json.loads((root / "similarity.json").read_text())
tokenized = json.loads((root / "tokenize.json").read_text())["data"][0]
for rows in (embed, jsonl):
    assert [r["index"] for r in rows] == [0, 1]
    assert "truncated" not in rows[0] and rows[1]["truncated"] is True
    assert rows[1]["tokens"] < tokenized["count"]
assert jsonl[1]["id"] == "long"
assert similarity["truncated_indices"] == [1] and similarity["tokens"][1] == embed[1]["tokens"]
PY

set +e
"$binary" embed "$long_text" --no-truncate >/dev/null 2>"$tmpdir/err"; code=$?
set -e
[ "$code" -eq 65 ] && grep -q 'context_length_exceeded' "$tmpdir/err" || fail "embed --no-truncate: exit $code"
set +e
printf '"%s"\n' "$long_text" | "$binary" embed --stdin --no-truncate >/dev/null 2>"$tmpdir/err"; code=$?
set -e
[ "$code" -eq 65 ] && grep -q 'context_length_exceeded' "$tmpdir/err" || fail "embed --stdin --no-truncate: exit $code"
set +e
"$binary" similarity x "$long_text" --no-truncate >/dev/null 2>"$tmpdir/err"; code=$?
set -e
[ "$code" -eq 65 ] && grep -q 'context_length_exceeded' "$tmpdir/err" || fail "similarity --no-truncate: exit $code"

# models install never runs on the fake runtime, and an override must pin a full commit (checked before any network use).
set +e
"$binary" models install >/dev/null 2>"$tmpdir/err"; code=$?
set -e
[ "$code" -eq 64 ] && grep -q 'fake_runtime' "$tmpdir/err" || fail "models install with the fake runtime: exit $code"
for revision in .. ../x a/b; do
  set +e
  "$binary" models install --runtime coreml --repo junmo-k/embeddinggemma-2-ane --revision "$revision" >/dev/null 2>"$tmpdir/err"; code=$?
  set -e
  [ "$code" -eq 64 ] && grep -q 'invalid_model_source' "$tmpdir/err" || fail "models install --revision $revision: exit $code"
done

printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"embed_text","arguments":{"text":"hi","dimensions":256}}}' \
  '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"embed_text","arguments":{"dimensions":256}}}' \
  | "$binary" mcp >"$tmpdir/mcp.jsonl" 2>"$tmpdir/mcp.err" || fail "mcp exited nonzero"
python3 - "$tmpdir/mcp.jsonl" <<'PY' || fail "mcp output failed validation"
import json, sys
responses = {r["id"]: r for r in map(json.loads, open(sys.argv[1]))}
assert set(responses) == {1, 2, 3, 4}, responses.keys()
assert responses[1]["result"]["serverInfo"]["name"] == "aneembed"
assert [t["name"] for t in responses[2]["result"]["tools"]] == ["embed_text", "embed_image", "similarity"]
ok = json.loads(responses[3]["result"]["content"][0]["text"])
assert len(ok["data"][0]["embedding"]) == 256 and not responses[3]["result"].get("isError")
err = json.loads(responses[4]["result"]["content"][0]["text"])
assert responses[4]["result"]["isError"] and err["reason"] == "missing_input", err
PY

# stdio lines are capped at --max-body: an over-long message is answered and skipped, the next one still works.
python3 -c 'import sys; sys.stdout.write("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\",\"pad\":\"" + "x" * (3 << 19) + "\"}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n")' \
  | "$binary" mcp --max-body 1 >"$tmpdir/mcp-long.jsonl" 2>"$tmpdir/mcp-long.err" || fail "mcp with a long line exited nonzero"
python3 - "$tmpdir/mcp-long.jsonl" <<'PY' || fail "mcp long-line output failed validation"
import json, sys
responses = [json.loads(line) for line in open(sys.argv[1])]
assert len(responses) == 2, responses
too_long = next(r for r in responses if r["id"] is None)
assert too_long["error"]["code"] == -32600 and "exceeds the 1048576-byte limit" in too_long["error"]["message"], too_long
assert any(r["id"] == 2 and r["result"] == {} for r in responses), responses
PY
set +e
printf '%s\n' "$(head -c 1100000 /dev/zero | tr '\0' 'x')" | "$binary" embed --stdin --max-body 1 >/dev/null 2>"$tmpdir/err"; code=$?
set -e
[ "$code" -eq 65 ] && grep -q 'line_too_long' "$tmpdir/err" || fail "embed --stdin with a line over --max-body: exit $code"

port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
"$binary" serve --port "$port" 2>"$tmpdir/serve.err" &
pid=$!
for _ in $(seq 1 100); do
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  sleep 0.1
done
curl -sf "http://127.0.0.1:$port/health" | grep -q '"ok"' || { cat "$tmpdir/serve.err" >&2; fail "serve did not become healthy"; }
python3 "$(dirname "${BASH_SOURCE[0]}")/serve-contract.py" "http://127.0.0.1:$port" || fail "serve contract"
kill "$pid"; wait "$pid" 2>/dev/null || true; pid=""

# Connections beyond --max-connections are answered 503 and closed; a freed slot serves again.
"$binary" serve --port "$port" --max-connections 2 2>"$tmpdir/serve2.err" &
pid=$!
for _ in $(seq 1 100); do
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  sleep 0.1
done
python3 - "$port" <<'PY' || { cat "$tmpdir/serve2.err" >&2; fail "connection limit"; }
import socket, sys, time
port = int(sys.argv[1])
held = [socket.create_connection(("127.0.0.1", port), timeout=10) for _ in range(2)]
time.sleep(0.5)
with socket.create_connection(("127.0.0.1", port), timeout=10) as extra:
    reply = b""
    while chunk := extra.recv(65536):
        reply += chunk
assert reply.startswith(b"HTTP/1.1 503") and b"too_many_connections" in reply, reply[:120]
held.pop().close()
time.sleep(0.5)
with socket.create_connection(("127.0.0.1", port), timeout=10) as again:
    again.sendall(b"GET /health HTTP/1.1\r\nConnection: close\r\n\r\n")
    reply = b""
    while chunk := again.recv(65536):
        reply += chunk
assert reply.startswith(b"HTTP/1.1 200"), reply[:120]
print("PASS connection limit: 503 beyond --max-connections, served again after one closed")
PY
kill "$pid"; wait "$pid" 2>/dev/null || true; pid=""
# The second model has its own vector, image and tokenizer contract under the same transport.
"$binary" serve --model pplx-embed-v1-0.6b --max-tokens 32768 --port "$port" 2>"$tmpdir/pplx.err" &
pid=$!
for _ in $(seq 1 100); do
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  sleep 0.1
done
curl -sf "http://127.0.0.1:$port/health" | grep -q '"ok"' || { cat "$tmpdir/pplx.err" >&2; fail "pplx serve did not become healthy"; }
python3 "$(dirname "${BASH_SOURCE[0]}")/serve-contract.py" "http://127.0.0.1:$port" --model pplx-embed-v1-0.6b ||
  fail "pplx serve contract"
kill "$pid"; wait "$pid" 2>/dev/null || true; pid=""
printf '%s\n' '{"text":"stdin example","id":"one"}' |
  "$binary" embed --model pplx-embed-v1-0.6b --stdin --dimensions 17 --encoding base64 >"$tmpdir/pplx.jsonl"
python3 - "$tmpdir/pplx.jsonl" <<'PYTEST'
import base64, json, struct, sys
data = json.load(open(sys.argv[1]))
values = struct.unpack("<17f", base64.b64decode(data["embedding"]))
assert data["id"] == "one" and data["tokens"] == len("stdin example".encode())
assert all(v == round(v) and -128 <= v <= 127 for v in values), values
PYTEST
set +e
"$binary" embed --model pplx-embed-v1-0.6b --image "$tmpdir/does-not-exist" >"$tmpdir/pplx.out" 2>"$tmpdir/err"; code=$?
set -e
[ "$code" -eq 65 ] && grep -q 'images_unsupported' "$tmpdir/err" || fail "pplx image rejection before file read: exit $code"

echo "release binary smoke test passed"
