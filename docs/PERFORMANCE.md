# Performance and memory

## Valen-4B on Apple M4

Measured on a 16 GiB M4 Mac mini (h16g), macOS 27.0.1. In the seven-input suite, each case went to a separate, initially unloaded server process. The client measured the whole first HTTP round trip and one immediate repeat. All 14 responses from the final executable matched the pinned publisher FP32 reference on the selected answer and token counts. The largest absolute difference across candidate probabilities was 0.000419, on a Korean negation input.

Each row identifies the measured executable by its SHA-256 prefix. The text ranges span six inputs. Each image row records one pair.

| Signed executable | Input | Input tokens | First request after prepare | Immediate repeat |
|---|---|---:|---:|---:|
| `6bbbfc2a` (source-install candidate) | Six English and Korean text inputs | 70–110 | 0.733–0.974 s | 0.381–0.418 s |
| `6bbbfc2a` | One red-circle image with text | 133 | 1.371 s | 0.821 s |
| `1bcd833e` (pinned-install candidate) | Same six text inputs | 70–110 | 0.764–1.164 s | 0.381–0.621 s |
| `1bcd833e` | Same image | 133 | 1.349 s | 0.775 s |
| `4f41e019` (final v0.3.1) | Same six text inputs | 70–110 | 0.759–1.544 s | 0.382–0.450 s |
| `4f41e019` | Same image | 133 | 1.400 s | 0.815 s |

Two additional inputs exercised larger device functions with the same final executable. Both requests for each input matched the publisher FP32 reference on the choice and on input, output and compute token counts. Prompt token IDs, role spans and candidate order also matched. Each reference distribution strongly favored one candidate.

| Extended input | Input / output / compute tokens | Answer | Max absolute probability difference | First HTTP | Repeat HTTP |
|---|---:|---|---:|---:|---:|
| 512 × 384 PNG, 768 patches | 261 / 0 / 261 | `red_circle` | 0.0000329414457153 | 3.530 s | 1.415 s |
| Refund-policy text crossing 128-token chunk boundaries | 456 / 0 / 456 | `reject` | 0.0000074493225862 | 2.020 s | 1.489 s |

The image trace recorded `p1024` and `featurescale`. All eight attention layers used `c512` for both extended inputs. These HTTP times are single traced observations with possible concurrent CPU test activity. Numerical parity on hardware was verified through `c512`. The declared `c1024`, `c2048` and `c4096` tiers remain untested for probability parity.

For `6bbbfc2a`, starting with the 9.75 GB source package already on disk, compiling and preparing a new model store took 391.64 s. All 101 assets were compiled for the destination. The OS retained its existing specialization cache. Allow additional space for compiled assets, scratch files and the OS cache.

In a separate run of the same image with `6bbbfc2a`, the highest sampled server RSS during inference was 49.3 MiB. With the model loaded and idle, the system-wide ANE driver reported 8.85 GiB. Five seconds after idle unload it reported 3.11 MiB. Use a positive `--idle-timeout` to release the model between bursts. The next request then pays the load cost again.

## pplx-embed-v1-0.6b on Apple M4

Measured with the Swift release build on a 16 GiB Mac mini, macOS 27.0.1.
The model defaults to 2,048 tokens. With `--max-tokens 32768`, fused functions
serve through 8,192 tokens and the long path uses one workspace for tiled attention.
The table gives warm inference times on fixed token IDs. The reference uses the
original FP32 model with the publisher's int8 output quantizer. Across 22 inputs,
the lowest cosine similarity was 0.9992044 before output quantization and
0.9988503 between the returned int8 vectors and the quantized reference.

| Input | Swift | Python ANE prototype | Cosine vs. reference int8 |
|---|---:|---:|---:|
| 8,180 tokens | 13.26 s | 11.1 s, fused | 0.999747 |
| 16,380 tokens | 48.24 s | 50.7 s | 0.999729 |
| 32,760 tokens | 178.06 s | 226.7 s | 0.999707 |
| 32,768 tokens, synthetic code, Korean and mixed Unicode | 165.79–184.06 s | | 0.999037–0.999585 |

At 32,760 tokens, Swift spent 163.15 s in attention, 8.97 s in pre/post
predictions and 5.93 s on the host critical path. The prototype spent 199.6 s
in attention and 15.7 s on the host. The Swift hardware suite peaked at
2.24 GiB physical footprint and 1.25 GiB RSS, including load and subsequent
inference. The prototype suite across three long texts peaked at 4.78 GiB
footprint and 3.07 GiB RSS. These process measurements exclude system-wide
wired memory. The MLX bf16 comparison took 86.9 s at 32K on the same M4.

The uncompiled source package is 2.41 GB. The local package with all
compiled functions uses about 3.9 GiB. Neural Engine preparation depends
on the cap, macOS build, executable and package path. At the default cap
a first local install took about 1.5 minutes. At the 32K cap a new server
binary became ready in 426 s and a cached warmup took 39 s.

In a mixed loopback run with the 32K cap, one 32,768-token request took
151.4 s. All 30 concurrent 32-token requests completed before it, and a
second long request queued behind it. Short-request p50/p95 was
16.5/17.8 ms alone and 112.5/241.8 ms beside the long request. The
server process peaked at 2.15 GiB footprint. System-wide wired memory
peaked at 8.69 GiB during the changing fused lengths and reached
2.46 GiB after 12 s idle. Swap peaked 508 MiB above the starting level
and finished 124 MiB below it. A client with a 60 s read timeout timed
out on a full-length request. Set the response timeout above the
expected inference time for long requests.

## Qwen3 Embedding on Apple M4

Qwen3-Embedding-0.6B ran on a 16 GiB Mac mini with macOS 27.0.1. Each row
compares the returned normalized vector with the original FP32 model on the
same token IDs. Lengths include EOS and any query instruction. Times cover
individual loopback HTTP requests after the server became ready.

| Input | Tokens | HTTP latency | Cosine vs. original FP32 |
|---|---:|---:|---:|
| Six short inputs, including a query also requested at 37 dimensions | 3–32 | 15.1–20.3 ms | 0.999802 minimum |
| Multilingual text and code | 2,060 | 6.690 s | 0.999873 |
| Multilingual text and code at full context | 32,768 | 172.963 s | 0.999897 |

The short requests used a 64-token cap. Across startup and the short requests,
the largest sampled server footprint was 84,542,856 bytes (81 MiB rounded).
The long requests used a 32,768-token
cap, one request at each length. The full-context server took 61.045 s to
start and self-test. Its peak process footprint was 1.98 GiB and peak RSS
was 1.08 GiB, sampled once per second. System-wide ANE buffers are accounted
for separately by macOS.

The long-context measurements above come from model qualification. A separate
M4 check of the packaged v0.2.0 candidate returned a 1,024-dimensional Qwen3
vector with cosine 0.999944 against FP32 on an eight-token input. On the same
executable, all 1,024 PPLX int8 coordinates and all 1,536 coordinates of two
EmbeddingGemma 2 vectors matched their previous M4 outputs exactly.

## ANERun across machines

The remaining tables cover EmbeddingGemma 2.

| | Apple M4 (16 GB) | M5 Max (64 GB) | M5 Ultra (256 GB) |
|---|---:|---:|---:|
| macOS | 27.0.1 | 26.7 | 27.0.1 |
| Neural Engines | 1 | 1 | 2 |
| 32-token query | 5.1 ms | 3.2 ms | 3.4 ms |
| 512 tokens | 30.6 ms | 26.2 ms | 27.5 ms |
| 1024 tokens | 83.0 ms | 70.1 ms | 71 ms |
| 2048 tokens | 211 ms | 167 ms | 168 ms |
| 4096 tokens (`--max-tokens 8192`) | 515 ms | 410 ms | 424 ms |
| 8192 tokens (`--max-tokens 8192`) | 1.46 s | 1.16 s | 1.19 s |
| 10 × 32 tokens, one request | 17.8 ms | 13.1 ms | 8.8 ms |
| 10 × 2048 tokens, one request | 2.11 s | 1.67 s | 0.86 s |
| one 640 × 480 image (official 280-token layout) | 488 ms | 383 ms | 378 ms |
| 8 clients sending single queries | 492 req/s | 747 req/s | 1,325 req/s |
| first install, compile included (default / `+ --max-tokens 8192`) | 260 s / + 389 s | 331 s / + 349 s | 242 s / + 383 s |
| process memory, idle | 140–270 MB | 186 MB | 155–166 MB |

- **How measured.** Every column is `anerun bench`, which sends its requests through the same router as `serve` in process, so only the loopback round trip is missing. Each cell is a warm median on texts of exactly N tokens including `<bos>`/`<eos>` and the prompt, at the default `--max-tokens 2048` unless the row says otherwise: 10 runs, 3 above 2048 tokens, 5 for the batches. Process memory is `serve`'s, idle after load.
- **What differs.** Single texts take about the same time on an M5 Max and an M5 Ultra: one request runs on one Neural Engine. The M5 Ultra's second Neural Engine shows in concurrent and batched work: 8 clients, 10 × 32 tokens and 10 × 2048 tokens, where it took about half the M5 Max's time. From 512 tokens up the M4 took 1.17–1.26 times as long as the M5 Max (1.6 times for a 32-token query), and served two thirds of its requests per second with 8 clients.
- **Throughput by length.** In tokens per second a single text is processed fastest at about 512 tokens: 16.7k on the M4, 19.5k on the M5 Max and 18.6k on the M5 Ultra. Shorter texts pay the fixed cost of a call (a 32-token query runs in the 64-token function, 6–10k), longer ones the attention, which grows with the square of the length: 12–17k from 768 to 1024 tokens, 8–10k at 4096 and 5.6–7.1k at 8192. A text just past a function's size runs in the next one (see [Latency steps](#latency-steps-m5-ultra)). Several texts in one request, or concurrent requests, raise the total, most on the M5 Ultra: 36k tokens/s for 10 × 32 tokens and 24k for 10 × 2048.
- **M4.** Measured on 2026-11-16 with the v0.1.0 release binary in a fresh data directory, while other work ran on the machine (swap 5.7–6.4 GB in use). The comparison with the GPU runtimes below was measured over loopback HTTP on a quiet machine the day before. Every function passed its self-test (12/12, and 16/16 with `--max-tokens 8192`). With 8 clients the median query took 15.8 ms.
- **Image.** On the M5 Ultra the image was 640 × 488 (266 soft tokens).
- **M5 Max.** Every function passed its self-test on macOS 26.7's Neural Engine compiler. The memory is after 2 minutes idle, when wired memory was 72 MB above where it was before start.
- **M5 Ultra.** The numbers were taken while another Neural Engine embedding service was running on the same machine. With 8 clients the median query took 5.9 ms. Its memory is idle under `serve`; `bench` ends at 221 MB after 150 s.

## Apple M4 (16 GB): ANERun and the GPU runtimes

Measured on 2026-11-15 on one Mac mini M4 (16 GB, macOS 27). Browsers and other heavy apps were closed, swap was under 4 GB at each start, and the 8192-token cases ran with the user logged out. Every runtime runs the official google/embeddinggemma-2 weights. The inputs are synthetic and public texts cut to exactly N tokens with the official tokenizer (prompt, `<bos>` and `<eos>` included), plus a 640×480 gradient PNG. Each cell is the warm median of 10 runs (3 for 8192 tokens). ANERun and llama.cpp were measured as servers over loopback HTTP, client-side time including JSON. MLX and PyTorch have no embedding server here and were measured in process.

Versions:
- ANERun: this repository, `--max-tokens 2048` (8192 for the 8192-token row).
- llama.cpp: b12387 Metal, `ggml-org/embeddinggemma-2-GGUF` BF16 and Q8_0, `--pooling mean -ngl 99`.
- MLX: 0.32.4 with mlx-vlm at the commit its model card names, `mlx-community/embeddinggemma-2-bf16` and `-8bit`.
- PyTorch: 2.7.1 with sentence-transformers 6.3.0, `google/embeddinggemma-2`.

| | ANERun | llama.cpp BF16 | llama.cpp Q8_0 | MLX bf16 | MLX 8bit | PyTorch fp32 MPS | PyTorch bf16 MPS | PyTorch fp32 CPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32-token query | 5.4 ms | 14.3 ms | 14.1 ms | 17 ms | 15 ms | 59–72 ms | 76 ms | 47 ms |
| 512 tokens | 31 ms | | | | | 116 ms | 167 ms | 327 ms |
| 2048 tokens | 213 ms | 406 ms | | 396 ms | 416 ms | 616 ms | 598 ms | 1,575 ms |
| 8192 tokens | 1.33 s | 2.94 s | 2.90 s | 3.54 s | 3.68 s | 6.6 s | swap guard | 13.9 s |
| image (official 280-token layout) | 518 ms | 575 ms (≥ 280 tokens) | | 468 ms | – | 958 ms | 747 ms | 2,065 ms |
| 10 × 32 tokens | 19 ms | 53 ms | | 41.6 ms | | 75 ms | 120 ms | 201 ms |
| 10 × 2048 tokens | 2.10 s | 4.03 s | | 4.01 s | 4.14 s | 6.44 s (logged out) | 6.5 s | 15.5 s |
| 8 clients × 256 queries | 440 req/s | 189 req/s | | – | – | – | – | – |
| lowest / median text cosine vs fp32 | 0.99997 / 0.99998 | 0.999997 / 0.999998 | 0.99989 / 0.99992 | 0.99994 / 0.99996 | 0.99976 | 1 | | reference |

- **ANERun column:** The ANERun column was measured again on 2026-11-28 with the v0.3.2 package: warm HTTP medians, five rounds per case, two passes alternating with the v0.3.1 package, which reproduced its earlier numbers within 0.5 % for 8192 tokens and the image and ran up to 23 % faster for short inputs than on 2026-11-15. The other columns and the cosine row are from 2026-11-15. Harness: `scripts/bench/http_bench.py`.
- **Empty cells** were not measured in that configuration.
- **"Swap guard"**: the run was stopped once swap grew 2 GB.
- **Logged out**: the fp32 MPS 8192-token and 10 × 2048 cases completed only with the user logged out (process 13.4 / 13.1 GB); with apps open they ran out of memory.
- **Images:** llama.cpp's default image budget is 134 tokens: 217 ms, but cosine 0.9963 to the official image vector. At `--image-min-tokens 280` it uses 304 tokens (575 ms, 0.99934). MLX bf16 encodes the image in 468 ms (in process, 270 tokens, PyTorch and torchvision added for its image processor; cosine not measured).

Memory:

| | ANERun | llama.cpp | MLX | PyTorch MPS |
|---|---|---|---|---|
| process memory, idle after load | 140–270 MB | 0.8 GB (Q8_0) / 1.2 GB (BF16) | 5.1–5.7 GB held | 6.7 GB |
| wired while idle | none: Neural Engine buffers (4.7 GB after start) are released 5–10 s after the last call | 2.1 GB (Q8_0) / 2.6 GB (BF16) of Metal buffers, wired through 120 s idle | none | none (Metal memory allocated, not wired) |
| uses the GPU | no (0 bytes of Metal memory) | yes | yes | yes |

**Start time:** llama.cpp is ready in 2–8 s and MLX in 3.5 s. ANERun starts in 8.5 s with its Neural Engine compile cached. The first start after installing a new binary takes 2.5 minutes (6.5 more for `--max-tokens 8192`).

**`--max-tokens 8192` on 16 GB:** at start the self-test wired 9.4 GB of Neural Engine buffers. They fell to 6.7 GB while running 8192-token texts and to 3.0 GB 30 s later. The process itself peaked at 2.0 GB.

**Bulk images with queries alongside (ANERun, M4).** Synthetic 12–24 MP JPEGs (2–4 MB each), 32 per run, sent to `anerun serve`, while one client sends a short query every 100 ms:

| image clients × images per request | images/s | query median | query p90 | query p99 |
|---|---:|---:|---:|---:|
| 1 × 1 | 1.19 | 5.7 ms | 374 ms | 448 ms |
| 1 × 2 | 1.42 | 15.6 ms | 389 ms | 438 ms |
| 1 × 4 | 1.46 | 15.4 ms | 292 ms | 425 ms |
| 1 × 8 | 1.51 | 15.8 ms | 283 ms | 444 ms |
| 2 × 4 | 1.56 | 6.7 ms | 698 ms | 805 ms |
| 3 × 8 | 1.59 | 15.9 ms | 657 ms | 755 ms |

Decoding a large JPEG on the CPU (about 1.8 s for 24 MP) dominates; more clients add 3–5 % throughput and about 2.5× the query tail. The server's footprint peaked at 484 MB.

## Latency steps (M5 Ultra)

`anerun bench` on the M5 Ultra as in the table above. A text runs in the smallest function that holds it, so latency steps at each function size and stays flat in between:

| tokens | latency |
|---|---:|
| 32 / 128 | 3.4 / 5.5 ms |
| 512 | 27.5 ms |
| 768 / 769 | 46.4 / 70.4 ms |
| 1024 | 71 ms |
| 1280 / 1281 | 94.0 / 114.5 ms |
| 2048 | 168 ms |
| 2049 / 3072 (`--max-tokens 8192`) | 281 / 287 ms |
| 3073 / 4096 | 422 / 424 ms |
| 4097 / 6144 | 769 / 759 ms |
| 6145 / 8192 | 1168 / 1187 ms |

The steps are at 769 (`embed_1024`), 1281 (`winp1536`), 2049 (`winl3072`), 3073, 4097 and 6145 tokens. Measured per call (`bench --buckets`): `embed_64` … `embed_1280` 3.2 / 5.2 / 10.7 / 27.3 / 46.1 / 70.7 / 93.5 ms, `winp1536` / `winp2048` 115 / 171 ms, `winl3072` … `winl8192` 286 / 424 / 771 / 1169 ms.

## `--max-tokens` trade-offs

The default favours start time and memory; the model's limit is 8192 tokens. Functions above the limit are not loaded or compiled.

| `--max-tokens` | 2048 (default) | 8192 |
|---|---|---|
| functions loaded | 12 (7 `embed_S`, 2 `winp`, 3 vision) | 16 (+ 4 `winl`) |
| first install, compile included (M5 Ultra) | 242 s | + 383 s |
| start with the compile cached, self-test included | 4.9 s | 9.1 s |
| self-test at load | 1.1 s | 3.9 s |
| process memory | 221 MB | 444 MB |
| text latency | up to 171 ms (2048 tokens) | 2049–3072: 0.28 s, –4096: 0.42 s, –6144: 0.77 s, –8192: 1.19 s |

Functions load four at a time: one by one, a cached start took 10.5 s instead of 4.9 s (default) and 17.5 s instead of 9.1 s (8192), but loading four at a time raises the brief peak of the process's memory during the start from 0.4 GB to 1.4 GB (`/usr/bin/time -l`, peak footprint) before it settles to the numbers above.

## Memory

Measured on an M5 Ultra with `footprint -w` and `vmmap`, macOS 27.0.1, `anerun serve` with the release binary. An `anerun` process has two kinds of memory.

**Its own memory** (`phys_footprint`, what Activity Monitor shows): at the default `--max-tokens 2048`, 155–166 MB idle and 267–383 MB five seconds after a mix of 200 requests (221 MB in `bench` after 150 s); with `--max-tokens 8192`, 210–243 MB idle and 283–599 MB after the mix (444 MB in `bench`). It is malloc memory (190–240 MB after the mix at the default settings), the Core ML input/output surfaces (`IOSurface`, 62 MB / 147 MB) and memory-mapped files, mostly the 256 MB token table, of which 9–31 MB was resident, all clean. All of it is pageable: with `memory_pressure -p 12` pushing the system to 12 % available memory, 292 of the process's 294 MB of dirty memory, the IOSurfaces included, was compressed or swapped out; once the pressure was gone, the first requests took 47 / 62 / 143 ms (query / 500 / 1500 tokens; 5 / 47 / 117 ms before), then warm again. Starting needs more for a moment: loading peaks at about 1.4 GB, which falls back within one to two minutes.

**The Neural Engine's buffers.** Core ML gives each loaded function working memory for the Neural Engine. It is owned by the process but not part of its footprint (`footprint` lists it as `Owned physical footprint (unmapped) (neural)`): 5.25 GB at the default settings (12 functions), 13.3 GB with `--max-tokens 8192` (16 functions). It is **wired only while it is used**:

| | default (2048) | `--max-tokens 8192` |
|---|---:|---:|
| right after start (the self-test runs every function), for about 60 s | 5.25 GB wired |
| idle | 0 wired (5.25 GB purgeable) |
| during and about 10 s after a short query | 266 MB wired |
| … a 1617-token text | 579 MB wired |
| … a 6631-token text | |
| 5 s after a mix of 200 requests (queries, 500/1500-token texts, images; 8192 also 8100-token texts) | 3.1 GB wired |

Idle buffers are purgeable: under the same memory pressure the OS discarded 1.6 GB of them (5.25 → 3.65 GB) without any error, and the next calls recreated them. That has a cost: the first request after the buffers were unwired or discarded took 19–139 ms for a short query (3–5 ms warm) and 0.12–0.14 s more for a 6631-token text. Under steady load the buffers of the functions in use stay wired; the "mix" row, taken 5 s after the last request, shows the scale.

**Request bodies** share one budget across all HTTP connections: one body of `--max-body` (72 MiB by default). A body over 1 MiB (an image upload) reserves its `Content-Length` once its head has arrived and before any of it is read, and waits, first come first served, while the budget is spent. Meanwhile its connection does not read, so TCP flow control slows the client. The wait is bounded by the body idle timeout (60 s): a body still waiting then is answered `503 body_budget_busy` with `Retry-After: 1` and the connection is closed, so connections cannot pile up behind a body that is held for long. A body is freed, returning its share, as soon as it is no longer needed: for `/v1/embeddings` once the request is parsed and its images decoded, for `/tokenize` and `/detokenize` once the request is parsed, for the other routes after the handler. Bodies up to 1 MiB, which covers a single text of up to 8192 tokens (8192 tokens of text is well under 100 KB), never wait; `--max-connections` bounds them (64 MiB at the defaults). Buffered request bodies therefore take at most 72 + 64 MiB at the defaults, instead of 64 × 72 MiB (4.5 GiB).

**Image decoding** is transient malloc memory, up to the 40 megapixel limit. A JPEG goes from the decoder to the resize row by row, so the full-size image never exists. Measured as the peak footprint `ImagePrep.prepare` adds (`jpeg-fuzz --peak`, M4, macOS 27, synthetic photo-like images): a sequential (baseline) JPEG, decoded one MCU row at a time, adds 3–4 MiB at 12, 24 and 40 MP, with 4:4:4 or 4:2:0 chroma, as this does not grow with the image height. A progressive JPEG needs all of its DCT coefficients (2 bytes per sample) before any pixel: 232 MiB at 40 MP with 4:4:4 chroma, 118 MiB with 4:2:0. A 48 MP JPEG (8064 × 6048, over the limit and refused) would take about 4 MiB sequential and 282 / 143 MiB progressive. Other formats go through ImageIO, which decodes the whole image: a 40 MP PNG peaks at 306 MiB. A file of a few hundred bytes can declare 40 MP, so plan for about 310 MiB per concurrent decode: `--image-concurrency` (default 2) bounds them. CI fails if a 40 MP 4:4:4 JPEG adds more than 16 MiB (baseline) or 256 MiB (progressive).

**Tokenizer requests** (`/tokenize`, `/detokenize`) run on the CPU and are bounded by size and by count. `content` is at most 1 MiB, and the ids of one `/detokenize` request may stand for at most 1 MiB of bytes (plus room for `<bos>`/`<eos>`); at most `--long-concurrency` (default 2) of them are worked on at once. Measured on an M4 (macOS 27, release binary with the fake runtime and the real tokenizer, over loopback HTTP), a 1 MiB text took 1.2–1.4 s end to end, whether prose, one long word, byte-fallback characters or with `with_pieces`, and the server peaked at 124 MB after four such requests. A response is a few MB: 0.9–1.2 MB of ids for 1 MiB of prose or one long word, 4.2 MB for 1 MiB of byte-fallback characters (three tokens each), 4.3 MB for the long word with `with_pieces`. `/detokenize` text is counted before invalid UTF-8 becomes U+FFFD and before JSON escaping, so its response can reach a few times 1 MiB.

For comparison, a Metal (GPU) LLM server running on the same Mac had 168 GB of `IOAccelerator (graphics)` memory, all of it wired, at the time of the measurement.
