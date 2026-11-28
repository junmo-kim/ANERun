# ANERun (`anerun`)

[![CI](https://github.com/junmo-kim/ANERun/actions/workflows/ci.yml/badge.svg)](https://github.com/junmo-kim/ANERun/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Platform: macOS 15+ · Apple Silicon](https://img.shields.io/badge/platform-macOS%2015%2B%20%C2%B7%20Apple%20Silicon-lightgrey)

Run local embeddings and decisions on Apple silicon with one Swift executable. Embeddings support macOS 15 and later through the CLI, OpenAI-compatible HTTP API and MCP. Decision models use Core AI on macOS 27. The decision model table lists the hardware verified for each portable package.

## Embedding models

| Model ID | Inputs | Maximum context | Output |
|---|---|---:|---|
| [embeddinggemma-2](https://huggingface.co/junmo-k/embeddinggemma-2-ane) | Text and images | 8,192 tokens | 768 normalized values, or normalized prefixes of 512, 256 or 128 |
| [pplx-embed-v1-0.6b](https://huggingface.co/junmo-k/pplx-embed-v1-0.6b-ane) | Text | 32,768 tokens | 1,024 signed int8 values, with prefixes of 1–1,024 |
| [qwen3-embedding-0.6b](https://huggingface.co/junmo-k/qwen3-embedding-0.6b-ane) | Text | 32,768 tokens including EOS | Normalized float vector, 32–1,024 dimensions (default 1,024) |

Each server process serves one model. Select it with `--model ID` or point to a verified local package with `--model-dir DIR`. For a v0.1.1 configuration, add `--model embeddinggemma-2` or set `ANERUN_MODEL=embeddinggemma-2` to keep using its existing package. Keep each index and its queries on the same model and dimensions.

## Local decisions

Install the model that fits your task. Each command downloads a pinned source package, compiles it on your Mac, and verifies its placement before activation. Installation requires Xcode 27 and the Metal Toolchain. Allow additional disk space for compiled assets and the native cache.

| Decision model | Input | Download | Verified hardware | Model license |
|---|---|---:|---|---|
| [Valen-4B](https://huggingface.co/junmo-k/valen-4b-ane) | Text, structured state, optional image | 9.75 GB | M4 Mac mini (h16g) | Apache-2.0 |
| [Elephant-1](https://huggingface.co/junmo-k/elephant-1-ane) | Text, structured state, optional image | 18.84 GB | M5 Ultra Mac Studio (h17d) | MIT; Qwen base terms included |
| [GLiDE-9B](https://huggingface.co/junmo-k/glide-9b-ane) | Text and structured state | 15.88 GB | M5 Ultra Mac Studio (h17d) | Fastino Community License |

**Built with Fastino GLiDE.** See the linked model cards for licensing and comparisons with the original checkpoints.

```sh
anerun models install --model elephant-1
anerun serve --model elephant-1 --port 8391
```

On a 16 GiB M4 Mac mini, select Valen explicitly with `anerun models install --model valen-4b` and `anerun serve --model valen-4b --port 8391`. On the final v0.3.1 executable, six short text inputs took 0.76–1.54 s on the first HTTP request to a fresh server and 0.38–0.45 s on the immediate repeat. A 133-token image request took 1.40 s and 0.82 s respectively. A larger image and a 456-token text input each took roughly 1.4–1.5 s on repeat in a separate traced run. Keeping the model loaded saves first-request time and consumes accelerator memory between requests. [Installation, memory and parity measurements](docs/PERFORMANCE.md#valen-4b-on-apple-m4)

For GLiDE-9B, use `anerun models install --model glide-9b` and select `--model glide-9b` when serving.

```sh
curl -sS http://127.0.0.1:8391/v1/decisions -H 'Content-Type: application/json' \
  -d '{"state":"The buyer wants their money back.","questions":[{"name":"action","kind":"choice","instructions":"Select the requested action.","candidates":[{"key":"refund","description":"Return the payment."},{"key":"replace","description":"Send another item."}]}]}'
```

Use `anerun playground --model elephant-1` for the local browser interface. The server loads the selected bundle on first request and supports configurable idle unload. See the [DecisionRuntime package](Packages/DecisionRuntime/README.md) for source packing and destination preparation.

### Serve embeddings and decisions together

Start one server per model, using separate terminals and ports. For example, after installing both packages:

```sh
# Terminal 1: POST /v1/embeddings on port 18180
anerun serve --model embeddinggemma-2 --port 18180
```

```sh
# Terminal 2: POST /v1/decisions on port 8391
anerun serve --model elephant-1 --port 8391 --idle-timeout 0
```

The decision server loads on its first request. An idle timeout of zero keeps it resident; set a positive number of seconds to release its memory between requests. Use `glide-9b` on the decision port to run GLiDE alongside the embedding server.

On an M5 Ultra, the same EmbeddingGemma 2 process (64-token cap) stayed loaded while the decision server switched from Elephant-1 to GLiDE-9B. Six pairs of overlapping warm HTTP requests per model returned the same embedding vectors and decision probabilities as separate requests. Median decision latency for the tested text input was 846 ms with Elephant-1 and 507 ms with GLiDE-9B.

With [llama-swap's matrix router](https://github.com/mostlygeek/llama-swap/blob/v273/docs/kb/guides/routing/groups-and-matrix.md), the set `"embeddinggemma-2 & (elephant-1 | glide-9b)"` allows the embedding server to stay loaded while the decision model changes.

## Choosing an embedding model for your memory budget

There are two separate costs: **running the model** on your Mac and **storing its vectors** in your search index. All three models ran on a 16 GB M4 Mac mini. The measurements below cover different workloads, so they establish neither a minimum RAM requirement nor a ranking of total memory use.

| Model | When to choose it | Measured M4 process memory | Raw storage per vector at full width |
|---|---|---|---:|
| EmbeddingGemma 2 | Search text and images in the same index. Text context tops out at 8,192 tokens. | 140–270 MB **idle**, with the default 2,048-token cap | 3 KiB (768 × float32) |
| PPLX | Index text up to 32K tokens and keep all 1,024 coordinates in a store that supports signed int8. | **2.24 GiB peak footprint** across loading and inference in the 32K suite | 1 KiB (1,024 × int8) |
| Qwen3 | Search text up to 32K tokens with an instruction on the query. Choose 32–1,024 float dimensions to fit your index. | **81 MiB sampled peak footprint** with a 64-token cap and short text, **1.98 GiB** in the 32K run | 4 KiB (1,024 × float32), or 128 B at 32 dimensions |

**Leave room for ANE buffers.** Process footprint alone misses memory that macOS wires for the accelerator. In a mixed PPLX run on the M4, **system-wide** wired memory peaked at 8.69 GiB and fell to 2.46 GiB after 12 seconds idle. The Qwen3 run recorded process memory only. For EmbeddingGemma 2, a separate **M5 Ultra** measurement at the default cap isolated the ANE buffers: 5.25 GB wired during startup self-tests, 266 MB during a short query and zero wired when idle. These readings have different scopes and should be read separately. [Memory measurements](docs/PERFORMANCE.md)

**On a memory-constrained Mac**, start with the default 2,048-token cap and raise it when a single input needs more context. For a large index, compare storage at the dimensions you intend to use. The byte counts above assume packed numeric storage and exclude index overhead. ANERun sends PPLX coordinates as JSON numbers or Float32 base64, so the index must store them as int8 to realize the 1 KiB figure.

## Install

Build from this checkout on an Apple silicon Mac. Use Xcode 16 or newer for embeddings on macOS 15 or newer.

```sh
swift build -c release
mkdir -p "$HOME/.local/bin"
install -m 755 .build/release/anerun "$HOME/.local/bin/anerun"
export PATH="$HOME/.local/bin:$PATH"
anerun --version
```

Choose one model ID from the table above. Set it once in your shell for install,
diagnostics and serving. For example, to use Qwen3:

```sh
export ANERUN_MODEL=qwen3-embedding-0.6b
anerun models install
anerun doctor
```

The same commands work with `ANERUN_MODEL=pplx-embed-v1-0.6b` or
`ANERUN_MODEL=embeddinggemma-2`. You can also pass `--model ID` on each command.
The installer downloads a pinned package, verifies its files, compiles it on
your Mac and warms the functions for your context limit. The default is 2,048
tokens. For a larger context, set `ANERUN_MAX_TOKENS` before installation and
use the same value when serving. The maximum for each model is in the table above.

Portable packages on Hugging Face: [EmbeddingGemma 2](https://huggingface.co/junmo-k/embeddinggemma-2-ane),
[PPLX](https://huggingface.co/junmo-k/pplx-embed-v1-0.6b-ane) and
[Qwen3](https://huggingface.co/junmo-k/qwen3-embedding-0.6b-ane).
To install a locally converted package, add `--from DIR`. EmbeddingGemma 2 can
reuse its v0.1.1 installation at the original path.

To override the source, give both `--repo OWNER/NAME` and `--revision` with a full 40-character commit SHA.

Installs for the same model share a lock. Each model has its own active revision, and the installer switches it atomically after verification and warmup. `anerun models path --model ID` shows the selected package. `anerun models warm --model ID` prepares it after a change to the package path, macOS build, executable or context limit. `serve` also warms and self-tests before reporting ready.

The release binary is ad-hoc signed. A tarball downloaded with a browser carries a quarantine attribute; clear it with `xattr -d com.apple.quarantine anerun` (Homebrew, mise and curl do not set it).

## Usage

With your installed model selected, start the server:

```sh
anerun serve
```

From another terminal, send a query. The server uses the model selected at startup.

```sh
curl -s http://127.0.0.1:18180/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":"How do I reset my router?","input_type":"query"}'
```

For one-off CLI requests, select the corresponding installed model:

```sh
anerun embed --model qwen3-embedding-0.6b --prompt query "How do I reset my router?"
anerun embed --model pplx-embed-v1-0.6b "Hold the reset button for ten seconds."
anerun embed --model embeddinggemma-2 --image photo.jpg --dimensions 256
```

`embed` (including JSONL input) and `similarity` truncate text to the model's configured context limit by default. Use `--no-truncate` to reject overlong text with `context_length_exceeded` (exit 65). An embedding row carries `"truncated": true` only when its text was clipped. A similarity result includes `truncated_indices` when clipping occurs, with `0` for the first operand and `1` for the second. `tokenize` shows the full token sequence.

For pplx, use `anerun serve --model pplx-embed-v1-0.6b --max-tokens 32768` after installing that model. The 32K path runs attention in tiles and can give short queries time between predictions. Allow several minutes for a full-length request on an M4. One-off commands load and self-test the model each time. For regular use keep `serve` or `mcp` running.

### Semantic search playground

Run `anerun playground --model qwen3-embedding-0.6b` after installing the model. Enter documents and a query to rank them by cosine similarity. EmbeddingGemma 2 also accepts images. The page keeps document vectors in memory and loads the model on the first embedding request. Use `--no-open` to open the URL yourself. The server listens on loopback and exits after five idle minutes by default or when you click Stop. Set `--idle-timeout SECONDS` to change the idle interval (0 disables it).

### MCP

```json
{ "mcpServers": { "anerun": { "command": "anerun", "args": ["mcp", "--model", "embeddinggemma-2"] } } }
```

Tools include `embed_text` and `similarity` for embedding models and `embed_image` for EmbeddingGemma 2. The text tools expose the context limit in `_meta["aneembed/max_context_length"]` in `tools/list`. They truncate overlong text by default and report clipping with the same fields as the CLI. Pass `"truncate": false` in the tool arguments to reject overlong text.

The same tools are reachable over HTTP as `POST /mcp` and as one-shot `QUERY /{tool}?<options>`. For a strict `QUERY` call, use `?truncate=false` or put `"truncate": false` in the JSON arguments body. Over stdio a message (one line) is limited to `--max-body`: a longer one is skipped and answered with a `-32600` error (`id` null). At most 16 messages are handled at once; further input is not read until one finishes. `embed --stdin` and `tokenize --stdin` stop with `line_too_long` (exit 65) at a line over `--max-body`. `embed_image` checks the number of images before decoding any of them, and each image's size before decoding it.

## API

| Endpoint | |
|---|---|
| `POST /v1/embeddings` | OpenAI-compatible. `input`: string, list of strings, or (EmbeddingGemma 2 only) a list of llama-server image items `{"content": [{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}]}` (texts and images are not mixed in one request) |
| `GET /v1/models` | OpenAI's model object (`id`, `object`, `created`, `owned_by`) plus `max_context_length`, `max_image_tokens`, `dimensions`, `image_preprocessing` |
| `GET /health` | `200` ready: `{"status": "ok"}`, or `{"status": "degraded", "not_served": [...], "max_context_length", "max_image_tokens"}` when loaded functions failed the self-test (or `"selftest": "skipped"` under `--selftest skip`); `503` while loading (the first start compiles for the Neural Engine) or after a load failure |
| `POST /tokenize` | llama-server's: `{"content": "..."}` → `{"tokens": [ids]}` |
| `POST /detokenize` | llama-server's: `{"tokens": [ids]}` → `{"content": "..."}` |
| `POST /mcp` | MCP JSON-RPC, one message per request |
| `QUERY /{tool}` | `embed_text` (UTF-8 body), `embed_image` (image body), `similarity` (JSON body); YAML by default, `?format=json` |

Request fields of `/v1/embeddings`:

- `encoding_format` is `float` (a JSON numeric array) or `base64` (little-endian Float32 bytes for all three models). With pplx the numeric values are signed int8 coordinates in either encoding.
- `dimensions` defaults to the model's full output width. EmbeddingGemma 2 and Qwen3 normalize the selected prefix. pplx takes the prefix of the mean-pooled vector, applies `tanh`, scales by 127 and rounds to signed int8 codes. Use cosine similarity to compare quantized vectors.
- `input_type` accepts `query` and `document`. EmbeddingGemma 2 adds the corresponding prompt prefix. PPLX embeds the provided text as-is in either mode. Qwen3 adds its retrieval instruction to queries and uses document text as supplied. For a custom Qwen3 instruction, pass the full `Instruct: …\nQuery:…` text and omit `input_type`. Images require EmbeddingGemma 2.
- `truncate` defaults to `true` for text. Clipped response items carry `"truncated": true`. Send `false` to reject text over the configured context limit with `400 context_length_exceeded`.
- The response names the server's selected model or its `--model-name` alias. The request's `model` and `user` fields are accepted and ignored.

`usage.prompt_tokens_per_second` reports effective input throughput. It divides the tokens actually used across the batch by elapsed processing time, including tokenization or image preparation, scheduling and inference. The timer starts after HTTP parsing and ends before JSON serialization. llama-swap v273 reads this oMLX-compatible field for `/v1/embeddings`.

`usage.prompt_tokens` counts the model tokens used. EmbeddingGemma 2 includes `<bos>` and `<eos>` for text and counts an image as its soft tokens plus four. PPLX has no automatic boundary tokens. Qwen3 counts the appended EOS token and any query instruction. Errors use OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`. Server-side failures carry a stable code and a generic message, with details in the server log. Invalid UTF-8 in JSON bodies receives `400 invalid_json`; raw `QUERY /embed_text` receives `400 invalid_utf8`.

`/tokenize` and `/detokenize` use the selected model's vocabulary. For pplx, `add_special` has no effect and `parse_special` controls recognition of the tokenizer's special tokens. `/tokenize` accepts at most 1 MiB of text, counts beyond the embedding limit and can return token pieces via `with_pieces`. `/detokenize` decodes the IDs back to text, including special tokens, and replaces invalid UTF-8 with U+FFFD. The response is capped at 1 MiB plus any boundary token bytes. Content over these limits receives `413 content_too_long`.

For EmbeddingGemma 2, `add_special` defaults to `false` and adds `<bos>` and `<eos>` when enabled. `parse_special` defaults to `true` and recognizes control tokens such as `<bos>`. The six user-defined tokens such as `<|channel>` are always recognized. A token piece contains the ID and its text or, for incomplete UTF-8, its bytes. `--long-concurrency` bounds the number of parallel tokenize and detokenize requests.

For valid requests the responses were compared with llama-server's (b12428 with `ggml-org/embeddinggemma-2-GGUF`) and are identical: every option combination on 16 hand-written texts (spaces, newlines, emoji, CJK compatibility characters, byte-fallback characters, control and user-defined added tokens), default options and `add_special` on 311 texts sampled from the tokenizer golden set, and `/detokenize` of all their ids. Where llama-server is lenient this API is strict, like the rest of it: a missing or non-string `content` (llama-server returns no tokens, and accepts an undocumented list of strings and ids), a missing `tokens` (llama-server returns empty content) and a flag that is not a boolean (llama-server uses the default) are `400`, and an id outside the vocabulary is `400 invalid_token` with `param: tokens[i]` (llama-server answers `500`).

HTTP/1.1 with keep-alive. `HEAD /health` answers with the `GET` response's headers, `Content-Length` included, and no body. A body needs `Content-Length`: chunked uploads are `411`, and ambiguous framing (`Transfer-Encoding` with `Content-Length`, other transfer codings, or repeated `Content-Length` headers with different values) is `400` and closes the connection. A request head must arrive within 60 s (of connecting, or of the previous response on a kept-alive connection); a body may take longer as long as no pause between its bytes exceeds 60 s. A response must be sent within 60 s plus 1 s per 64 KiB; a client that stops reading is disconnected, so it cannot hold a connection slot.

## Configuration

Every option has a flag and an environment variable. Precedence is flag, then `ANERUN_*`, then the corresponding legacy `ANEEMBED_*`, then the default. Empty environment values are ignored. `anerun serve --print-config` shows each value and where it came from.

| Flag | Environment | Default | |
|---|---|---|---|
| `--model` | `ANERUN_MODEL` | required unless `--model-dir` is set | model ID |
| `--model-dir` | `ANERUN_MODEL_DIR` | required unless `--model` is set | local model package directory |
| `--data-dir` | `ANERUN_DATA_DIR` | `~/Library/Application Support/ANEEmbed` | installed models and state |
| `--host` | `ANERUN_HOST` | `127.0.0.1` | anything else exposes an unauthenticated API |
| `--port` | `ANERUN_PORT` | `18180` | |
| `--max-tokens` | `ANERUN_MAX_TOKENS` | `2048` | 8192 maximum for EmbeddingGemma 2 and 32768 for PPLX and Qwen3 |
| `--model-name` | `ANERUN_MODEL_NAME` | selected model ID | response alias |
| `--max-body` | `ANERUN_MAX_BODY` | `72` (MiB) | 8 image data URLs of up to 8 MiB plus JSON; also what all connections together may buffer of bodies over 1 MiB ([memory](docs/PERFORMANCE.md#memory)) |
| `--max-connections` | `ANERUN_MAX_CONNECTIONS` | `64` | open HTTP connections; more are answered `503 too_many_connections` (`Retry-After: 1`) and closed |
| `--max-inputs` / `--max-images` | `ANERUN_MAX_INPUTS` / `ANERUN_MAX_IMAGES` | `2048` / `8` | per request |
| `--query-concurrency` / `--image-concurrency` / `--long-concurrency` | `ANERUN_*_CONCURRENCY` | `8` / `2` / `2` | separate lanes so long documents and images cannot hold short queries back; `--long-concurrency` also bounds concurrent `/tokenize` and `/detokenize` requests |
| `--runtime` | `ANERUN_RUNTIME` | `coreml` | `fake` runs without a model, for testing integrations |
| `--selftest` | `ANERUN_SELFTEST` | `require` | `skip` serves every function without the [self-test](#self-test): **unsafe**, a miscompiled function returns wrong vectors without an error |


## Best practices

- **Use matching vectors for search.** Index documents and embed queries with the same model, dimensions and runtime. Set `input_type` to `document` or `query` to apply the model-specific formatting described in [API](#api).
- **EmbeddingGemma 2 image backfills: one client, 4–8 images per request.** More clients barely add throughput: the Neural Engine runs one call at a time, and large JPEGs are decoded on the CPU, at most `--image-concurrency` (default 2) at a time. On an M4 with 12–24 MP photos, one client reached 1.19 images/s at 1 image per request, 1.46 at 4 and 1.51 at 8. Two or three clients reached 1.56–1.59 (+3–5 %), but short queries sent meanwhile waited about 2.5 times longer (p90 0.66–0.70 s instead of 0.28 s). If searches run at the same time, prefer one client. Downscaling very large photos on the client speeds up decoding, at the cost of pixels that differ slightly from the official preprocessing.
- **Long documents: chunk them, or raise `--max-tokens`.** The default 2048 keeps start time and memory low. Chunk long documents to embed content beyond the configured context limit. In the EmbeddingGemma 2 benchmark, chunks of about 512 tokens are best; an 8192-token text is processed at about a third of that rate ([details](docs/PERFORMANCE.md#anerun-across-machines)).
- **Keep `serve` (or `mcp`) running.** One-off CLI commands load and self-test the selected model each time.

## Performance and memory

| Model on M4 (16 GB) | Short text | Long text |
|---|---:|---:|
| EmbeddingGemma 2 | 5.1 ms at 32 tokens (warm median, in process) | 1.46 s at 8,192 tokens (warm median) |
| PPLX | 16.5 ms at 32 tokens (HTTP p50, short requests alone) | 165.79–184.06 s at 32,768 tokens (warm inference, three inputs) |
| Qwen3 | 15.1–20.3 ms at 3–32 tokens (HTTP) | 172.96 s at 32,768 tokens (one HTTP request) |

The [benchmark details](docs/PERFORMANCE.md) give the input sets, timing boundaries and memory measurements.

### EmbeddingGemma 2 across machines

| ANERun, warm median | Apple M4 (16 GB) | M5 Max (64 GB) | M5 Ultra (256 GB) |
|---|---:|---:|---:|
| short query | 5.1 ms | 3.2 ms | 3.4 ms |
| 512 / 2048 tokens | 31 / 211 ms | 26 / 167 ms | 27.5 / 168 ms |
| 4096 / 8192 tokens (`--max-tokens 8192`) | 515 ms / 1.46 s | 410 ms / 1.16 s | 424 ms / 1.19 s |
| image, 280-token budget | 488 ms | 383 ms | 378 ms |

All three with `anerun bench` in process. A text runs in the smallest function that holds it, so latency steps at each function size (for example at 769, 1281, 2049, 3073, 4097 and 6145 tokens) and is flat in between.

- **Process memory**: about 140–380 MB, all pageable. A request keeps at most 3 Neural Engine calls of its short texts in flight, so even a batch of 2048 texts holds only a few input buffers (2048 × 512-token texts: 24 MiB).
- **Neural Engine buffers**: owned by the process but outside its footprint, 5.3 GB at the default settings and 13.3 GB with `--max-tokens 8192`. They are wired only while used: for about a minute after start (the self-test), and during and about 10 s after each call. Idle, none is wired and the OS may discard them; the next call recreates them, which costs up to about 0.1 s once.
- **Request bodies**: all connections share a budget of one `--max-body` (72 MiB). Bodies up to 1 MiB, which covers a single text of up to 8192 tokens, never wait; larger ones (image uploads, big batches) queue for it.
- **Images**: a JPEG streams from the decoder into the resize, so a 40-megapixel baseline JPEG adds about 4 MiB. Progressive JPEG and other formats need the whole image: up to about 310 MiB per concurrent decode.

### EmbeddingGemma 2 context and startup costs

| `--max-tokens` | 2048 (default) | 8192 |
|---|---|---|
| functions loaded | 12 (7 `embed_S`, 2 `winp`, 3 vision) | 16 (+ 4 `winl`) |
| first install, compile included (M5 Ultra) | 242 s | + 383 s |
| start with the compile cached, self-test included (M5 Ultra) | 4.9 s | 9.1 s |
| Neural Engine buffers wired during the start | 5.3 GB | 9–13 GB |
| longest text (M5 Ultra) | 2048 tokens, 171 ms | 8192 tokens, 1.19 s |

## EmbeddingGemma 2 at a glance

The same 16 GB Apple M4 (macOS 27), the same inputs, warm medians, measured on a quiet machine. Every runtime runs the official google/embeddinggemma-2 model; accuracy is the lowest cosine to the official fp32 vectors over the same 26 texts. Details, versions and flags: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

| Apple M4, 16 GB | **ANERun** (Neural Engine) | llama.cpp (Metal GPU) | MLX (Metal GPU) | official PyTorch, fp32 (MPS GPU) |
|---|---:|---:|---:|---:|
| short query (32 tokens) | **5.4 ms** | 14 ms | 15–17 ms | 59–72 ms |
| 2048-token document | **213 ms** | 406 ms | 396 ms | 616 ms |
| 8192-token document | **1.33 s** | 2.9 s | 3.5 s | 6.6 s (13.4 GB process) |
| image (official 280-token budget) | 518 ms | 575 ms¹ | **468 ms**² | 958 ms |
| 10 × 2048-token documents, one request | **2.1 s** | 4.0 s | 4.0 s | 6.4 s³ |
| 8 clients sending short queries | **440 req/s** | 189 req/s | n/a⁴ | n/a⁴ |
| process memory, idle | **140–270 MB** | 0.8–1.2 GB | 5.1–5.7 GB held | 6.7 GB |
| memory wired while idle | **none** | 2.1–2.6 GB | none | none |
| uses the GPU | **no** | yes | yes | yes |
| lowest text cosine vs official fp32 | 0.99997 | 0.999997 (BF16) / 0.99989 (Q8_0) | 0.99994 (bf16) / 0.99976 (8bit) | 1 (reference) |

ANERun and llama.cpp are measured as servers over loopback HTTP; MLX and PyTorch have no embedding server here and are measured in process, which spares them the HTTP round trip. For llama.cpp and MLX each cell shows the best of BF16 and 8-bit. The ANERun column was measured again on 2026-11-28 with the v0.3.2 package: warm HTTP medians, five rounds per case, two passes alternating with the v0.3.1 package, which reproduced its earlier numbers within 0.5 % for 8192 tokens and the image and ran up to 23 % faster for short inputs than on 2026-11-15. The other columns and the cosine row are from 2026-11-15. On an M5 Ultra ANERun took 3.4 ms, 168 ms and 1.19 s for the three text sizes with the v0.3.1 package.

¹ At `--image-min-tokens 280` (304 tokens, cosine 0.99934). llama.cpp's default budget is 134 tokens: 217 ms, but cosine 0.9963 to the official image vector.

² In process, with PyTorch and torchvision added for the image processor; its cosine to the official image vector was not measured.

³ Only with the user logged out: the process reached 13.1 GB. With apps open it ran out of memory, as did the 8192-token case (6.6 s logged out, 13.4 GB).

⁴ No embedding server; measured in process only.

## EmbeddingGemma 2 features

- **Leaves the GPU alone.** Nothing runs on the GPU, so a local LLM keeps all of it.
- **Memory the OS can take back.** The process itself uses about 140–380 MB, all of it pageable: under memory pressure it is compressed or swapped like any app. The Neural Engine's working buffers are wired only while a call runs and for about 10 s after it; idle, they are purgeable. A Metal runtime keeps its buffers wired, or held, for as long as the model is loaded. Request data is never written to disk.
- **Fast.** One Neural Engine call per text up to 8192 tokens, short texts packed together, and separate lanes so long documents and images do not hold short queries back.
- **Accurate.** Cosine ≥ 0.99994 to the official fp32 model on every public test input, texts up to 8192 tokens and images; median 0.999995 for short queries and 0.999998 for texts over 2048 tokens (see [Accuracy](#accuracy)).
- **Checks itself.** Every model function is tested against golden outputs when it loads, and one the Neural Engine compiler got wrong is not used (see [Self-test](#self-test)).
- **OpenAI drop-in, CLI and MCP** from one binary: `POST /v1/embeddings` with text and images, `encoding_format` float/base64, Matryoshka `dimensions` 768/512/256/128; `anerun embed`, `anerun serve`, `anerun mcp`.

## EmbeddingGemma 2 trade-offs

- **The first install compiles for the Neural Engine:** about 4 minutes (M5 Ultra) to 5.5 minutes (M5 Max), plus about 6 more for `--max-tokens 8192`. The OS caches the result; an upgraded `anerun` or a macOS update compiles again. llama.cpp and MLX start in seconds.
- **llama.cpp BF16 is closer to the official vectors** (0.999997 against ANERun's 0.99997). Both are far closer than the 8-bit ports.
- **Texts over 2048 tokens need `--max-tokens 8192`**, which compiles four more functions. At start the self-test wires about 9 GB of Neural Engine buffers for about a minute; on a 16 GB Mac, start it when memory is free.
- **The Neural Engine runs one call at a time.** While a request of large images is processed, short queries wait for it (on an M4: query p90 0.28 s while one client uploads 8 photos of 12–24 MP per request, 0.66 s with three such clients). See [Best practices](#best-practices) for bulk image jobs.
- **Apple silicon and macOS 15 or later only.** Text up to 8192 tokens (2048 by default); images at the model's official 280-token budget; HEIC decodes slightly differently from the official path (see [Image input formats](#image-input-formats)).

## Design

Each model ships as portable Core ML source. ANERun compiles the selected package locally and warms it at its installed path. Changing the executable, package path or macOS build can trigger compilation again.

**PPLX** uses bidirectional attention and mean pooling. It uses fused functions through 8,192 tokens and tiled attention for longer inputs. The output uses the publisher's signed int8 encoding.

**Qwen3** uses causal attention and pools the last token (EOS). A packed function handles inputs up to 64 tokens. Longer inputs use per-layer functions and tiled attention. ANERun normalizes the requested output dimensions.

### EmbeddingGemma 2 text and vision graphs

**Fixed shapes, packed.** The Neural Engine runs fixed-shape graphs; enumerated or flexible shapes moved the whole graph to the CPU in our tests. So the model package has one function per size, all sharing one copy of the weights: `embed_64` … `embed_1280` with full attention, and above that windowed functions (below). A request's texts are packed back to back into as few calls as possible, with a block-diagonal attention mask so packed texts never see each other (positions need no reset: RoPE is relative). The planner picks the cheapest set of calls from the measured per-call latency of each size, and short queries that arrive while the Neural Engine is busy are merged into the next call.

**Windowed attention, folded.** EmbeddingGemma 2 has 24 layers; 20 of them only look 512 tokens to each side. Full attention over S rows costs S² in those layers for nothing, so the windowed functions split the rows into blocks of 512 and let each block attend to the 1536 rows around it (512 on each side), with the mask passed in as a `[S, 1536]` band. The 4 global layers attend to every row. That keeps one fixed-shape call per text up to 8192 tokens (`winl3072` … `winl8192`) and makes 1281–2048-token calls cheaper (`winp1536`, `winp2048`, which still pack). Attention uses a "fold" layout: the query heads that share a K/V head are folded into the row dimension, because the usual `repeat_interleave` layout is miscompiled (next paragraph).

**Miscompiles, and the self-test.** The Neural Engine compiler runs on your Mac and a wrong graph does not fail, it returns wrong numbers. Converting this model, the usual attention layout came back with three of four query heads zeroed in the global layers, a sliding-window mask built into the graph as a constant was computed wrongly, and padding inside the graph was correct but about 3× slower. Rebuilt with the usual layout, `winp2048`, `winl4096` and `winl8192` score cosine 0.96, 0.95 and 0.89 on the Neural Engine. The package therefore passes every mask as an input, uses the fold layout, and ships one golden test per function that `anerun` runs before serving (see [Self-test](#self-test)).

## Accuracy

These checks measure cosine similarity between ANERun outputs and vectors from the original model on matching inputs.

| Model | Reference and inputs | Observed cosine |
|---|---|---:|
| EmbeddingGemma 2 | Original FP32, 14 long texts of 4,097–8,192 tokens on M4 | 0.999993 minimum |
| PPLX | Original FP32 with the publisher's int8 output quantizer, 22 inputs on M4 | 0.998850 minimum |
| Qwen3 | Original FP32, six short inputs on M4 and one 32,768-token input | 0.999802 minimum on short inputs, 0.999897 at full context |

The [M4 measurements](docs/PERFORMANCE.md) include model-specific input lengths and the packaged v0.2.0 checks.

### EmbeddingGemma 2 validation corpus

The reference is google/embeddinggemma-2 in fp32 through its official `transformers` stack on the CPU. Mean pooling over all tokens followed by L2 normalization matches `sentence-transformers` at cosine 1.0000000. Every input went through `anerun` 0.3.1 with this release's package over HTTP on a 16 GB M4 (macOS 27.0.1) with the 8,192-token cap; the previous package gives the earlier numbers on the same Mac, identical to those measured on an M5 Ultra. Inputs are public: multilingual READMEs (en/ko/ja/zh/fr) of an Apache-2.0 project, the Korean and Japanese constitutions (public domain), MIT-licensed technical documents and this repository's MIT-licensed converter source (corpus v2), cut to exact token counts with the document prompt; 13 images bundled with scikit-learn, networkx, llama.cpp and stable-diffusion.cpp plus synthetic ones, at the official 280-token budget. `usage.prompt_tokens` matched the reference token count in all 167 requests.

| Input | n | min cos | median cos |
|---|---:|---:|---:|
| text, one per request, 37–128 tokens | 24 | 0.999991 | 0.999995 |
| text, one per request, 129–512 tokens | 24 | 0.999947 | 0.999976 |
| text, one per request, 389–2048 tokens (every function size, and one token over each) | 70 | 0.999945 | 0.999976 |
| text, 10–40 short texts (9–200 tokens) per request, packed | 200 | 0.999946 | 0.999975 |
| text, 2049–4096 tokens (incl. 3072 / 3073) | 12 | 0.999976 | 0.999998 |
| text, 4097–8192 tokens (incl. 6144 / 6145) | 14 | 0.999993 | 0.999998 |
| image, one per request | 13 | 0.999960 | 0.999984 |
| images, up to 8 per request | 13 | 0.999960 | 0.999984 |

The default `--max-tokens 2048` gives the same numbers for everything up to 2048 tokens (same functions). The model package itself, checked through the Python host contract (`tools/convert/verify.py`), gives the same picture per function: 380 packed and single texts up to 2048 tokens at ≥ 0.999936 and 44 long texts at ≥ 0.999996. To reproduce: `tools/convert/fetch.py data`, `corpus.py` and `images.py` rebuild the inputs from pinned public sources, `scripts/accuracy/reference.py` writes the reference vectors (needs PyTorch and the original model), `scripts/accuracy/compare.py` queries a running `anerun serve` (standard library only).

### Image input formats

The official path reads an image with transformers' `load_image` (PIL): RGB, EXIF orientation applied, alpha dropped. `anerun` decodes JPEG with its own decoder (libjpeg-turbo's algorithms, bit-exact with PIL) and every other format with ImageIO, taking the decoded values without colour management. The cosines in this table were measured with the package of v0.3.1 and earlier; the decoded-pixel comparison does not depend on the package. Measured like the table above (over HTTP, M5 Ultra; JPEG: the 4 JPEG files among the public images), with the 13 public images re-encoded in each format (`scripts/accuracy/formats.py`; 3 files whose data URLs were over the 8 MiB limit are left out; the reference reads the same files with `load_image`, HEIC through `pillow-heif` 1.8.1, libheif 1.23.6):

| Format | Decoder | n | min cos | median cos | Decoded pixels vs PIL |
|---|---|---:|---:|---:|---|
| JPEG | own | 4 | 0.999954 | 0.999966 | identical (bit-exact on the golden test images) |
| PNG | ImageIO | 13 | 0.999954 | 0.999979 | identical |
| WebP, lossy (quality 90) and lossless | ImageIO | 26 | 0.999949 | 0.999978 | identical |
| GIF (256 colours) | ImageIO | 13 | 0.999950 | 0.999979 | identical |
| TIFF, BMP | ImageIO | 24 | 0.999954 | 0.999977 | identical |
| animated GIF / WebP / PNG (3 frames) | ImageIO, first frame | 38 | 0.999950 | 0.999979 | identical (frame 0) |
| HEIC (8-bit, written by `sips`) | ImageIO | 13 | 0.997272 | 0.999463 | differ: mean 0.4–1.0 levels, 11.8 on a noise image |
| CMYK / YCCK JPEG | ImageIO | | | | differ: mean 32, max 130 levels (synthetic test image) |

- **Animated images** use the first frame only, as the official transformers path does; other frames are ignored.
- **HEIC** is the one format that does not match: ImageIO's HEVC decode is about 0.2–0.8 levels darker per channel even on flat areas and differs strongly at sharp colour edges (chroma upsampling), compared with libheif. It is not colour management (both files and decoders are sRGB): the same HEIC pixels decoded by PIL and sent as PNG score min 0.999913. Matching it would need a HEVC decoder in `anerun`; convert HEIC to PNG or JPEG where this matters. Note that PIL itself reads HEIC only with the `pillow-heif` plugin.
- **CMYK JPEG** falls back to ImageIO, which converts colours differently from PIL.
- **Progressive JPEGs** with more than 100 scans, or with scans out of the order the format defines, are refused (`400 image_decode_failed`, "JPEG refused: …") instead of decoded; libjpeg and PIL would decode the latter with a warning. Standard progressive files (10 scans) are unaffected.
- Images are limited to 40 megapixels and to 8 MiB as a data URL (about 6 MiB of image file); formats ImageIO cannot read return `400 image_decode_failed`.

The tokenizer is checked token for token against Hugging Face `tokenizers` on 12,420 generated texts (every id must match), and image preprocessing bit for bit against `Gemma4ImageProcessor`.

## Self-test

The EmbeddingGemma 2 package ships one golden test per function (`selftest.json` / `selftest.bin`, 16 tests, expectations from the original model in fp32), and `anerun` runs the test of every function it loads before serving. Only a function that passes in this run serves (an allowlist): one below the tolerance (cosine 0.999; packing functions also unused rows ≤ 0.001, images relative error 0.05; a package may tighten these, not loosen them), one with a NaN or infinite output, or one the package has no test for is not used. A larger function, a shorter text limit or fewer image sizes take its place, the log gives the limits that remain, `/health` reports `degraded`, and `anerun doctor` lists it (records of functions a smaller `--max-tokens` did not load are kept). On an M5 Ultra (macOS 27.0.1) every function scores cosine ≥ 0.99990; the same functions rebuilt with the usual attention layout score 0.89–0.96 on the Neural Engine and fail (see [Design](#design)).

A package needs a valid `manifest.json` with its function and file signatures. Normal loads require readable self-test data and serve only validated routes. The diagnostic option `--selftest skip` skips numerical reference checks. pplx still checks tensor signatures and test predictions for availability. Each load checks source sizes and hashes assets such as the tokenizer and the self-test files. `anerun models verify --model ID` hashes all source files and checks that local compiled packages are complete. The self-test verifies what the compiled packages compute.

## Known pitfalls

- **Graph-constant masks miscompile on the ANE.** The sliding-window mask of Gemma's local layers is an input (`local_bias`, or the `band_bias` of the windowed functions), not a constant in the graph.
- **EmbeddingGemma 2 graph shapes.** Every text function has fixed shapes (64…1280 tokens with full attention, 1536 and 2048 windowed and packable, 3072…8192 windowed for one text); several short texts are packed into one call with block-diagonal attention.
- **The first compile takes minutes**, and the OS caches the result. After changing the binary or model path, verify readiness with the same invocation you plan to use for the service. `doctor` reports the recorded warm state, and `serve` loads and self-tests before reporting healthy.
- **Memory reporting**: the Neural Engine's buffers do not show in the process footprint (`footprint -w` lists them as `(neural)`), and freed malloc pages only become reclaimable about a minute later, so short-lived measurements overstate the process's own memory and miss the Neural Engine's.
- **EmbeddingGemma 2 with llama.cpp GGUF**: text vectors are compatible; image preprocessing in llama.cpp differs from the official recipe, so image vectors are not interchangeable.

## Running as a service

Choose your installed model ID for the argument after `--model`. Set the first `ProgramArguments` entry to the absolute path printed by `command -v anerun` on your Mac.

```xml
<!-- ~/Library/LaunchAgents/com.github.junmo-kim.anerun.plist -->
<plist version="1.0"><dict>
  <key>Label</key><string>com.github.junmo-kim.anerun</string>
  <key>ProgramArguments</key><array><string>/opt/homebrew/bin/anerun</string><string>serve</string><string>--model</string><string>embeddinggemma-2</string></array>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/tmp/anerun.log</string>
</dict></plist>
```

There is no authentication or TLS. To serve other machines, keep `--host 127.0.0.1` and put a reverse proxy in front.

## Build and test

```bash
swift build -c release
CI=true swift test                           # unit, golden and fake-runtime contract tests
Tests/Integration/release-binary-launch-smoke.sh .build/release/anerun
swift run -c release jpeg-fuzz --iterations 200000 Tests/Fixtures/images   # JPEG decoder mutation fuzzing
ANEEMBED_TEST_MODEL_DIR=/path/to/model swift test --filter ModelTests   # real model, needs the ANE
anerun bench --model embeddinggemma-2 --image photo.jpg --buckets --breakdown
```

The test binary is another executable, so the first `ModelTests` run compiles the model for the Neural Engine again (minutes).

[`tools/convert`](tools/convert/README.md) builds the packages from pinned model revisions. Its reproducibility checks cover weights, normalized programs and self-test data.

## License

MIT for this code. The model packages keep their own licenses. EmbeddingGemma 2 by Google DeepMind is Apache-2.0. The converter builds its text and vision packages from the Google checkpoint. EmbeddingGemma 2 is also subject to the Gemma Prohibited Use Policy. pplx-embed-v1-0.6b by Perplexity AI is MIT. Qwen3-Embedding-0.6B by Qwen is Apache-2.0. See the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) and [NOTICE](NOTICE).

