---
name: anerun
description: >
  Local text and image embeddings on the Apple Neural Engine (ANERun, macOS).
  Embed with EmbeddingGemma 2 or text-only pplx-embed-v1-0.6b or qwen3-embedding-0.6b, compare two inputs by cosine
  similarity, count tokens, or run an OpenAI-compatible /v1/embeddings server, without a cloud API and
  without using the GPU. Reach for this when an agent needs semantic search, deduplication, clustering or
  retrieval over local data, needs to embed a query and documents consistently, or needs to check how
  similar two texts or a text and an image are. Triggers: embed text, embedding vector, semantic
  similarity, cosine similarity, vector search, RAG embeddings, OpenAI embeddings endpoint locally,
  image embedding, text-to-image search, token count for EmbeddingGemma. Apple Silicon + macOS 15+.
user_invocable: true
---

# anerun

`anerun` embeds texts and images on the Neural Engine. Put the binary on your `PATH` (README → Install)
and install a model once with `anerun models install --model ID`. Select `embeddinggemma-2` for
text and images or either `pplx-embed-v1-0.6b` or `qwen3-embedding-0.6b` for text. Pass `--model ID`, set `ANERUN_MODEL=ID`, or
use `--model-dir DIR` for a local package. The commands below assume you set `ANERUN_MODEL`.
Output is YAML by default. Add
`--format json` to parse. Errors go to stderr as `error` / `reason` / `detail` / `hint` with an exit code.

## Which command

| You need | Command |
| --- | --- |
| Is it set up? What to fix? | `anerun doctor` |
| Embed a search query | `anerun embed --prompt query "<query>"` |
| Embed documents for an index | `anerun embed --prompt document "<doc>" ...` or JSONL: `anerun embed --stdin --prompt document < docs.jsonl` |
| Embed an image | `anerun embed --image <path>` |
| Similarity of two inputs | `anerun similarity "<a>" "<b>"` (`--image-a` / `--image-b` for images) |
| Count tokens before sending | `anerun tokenize "<text>"` |
| Many calls / a long session | keep `anerun serve` (HTTP) or `anerun mcp` (stdio) running |

## Rules that matter

- For EmbeddingGemma 2 retrieval use `--prompt query` or `--prompt document` (or `input_type` over
  HTTP/MCP). pplx embeds the supplied text as-is.
- All three models default to 2,048 tokens. Set `--max-tokens` up to 8,192 for EmbeddingGemma 2 or
  32,768 for pplx and Qwen. Overlong text is truncated by default. Use `--no-truncate` with `embed` or
  `similarity`, or pass `"truncate": false` in HTTP/MCP JSON arguments to reject it with
  `context_length_exceeded`. CLI and MCP embeddings mark clipped rows with `"truncated": true`.
  Similarity reports clipped operand indices in `truncated_indices` (0 for the first, 1 for the second).
  Allow several minutes for full-length 32K requests on an M4.
- Compare vectors from the same model and dimensions. EmbeddingGemma 2 returns normalized vectors.
  pplx returns signed int8 coordinates in numeric arrays or base64 Float32. Use cosine similarity for pplx.
- A one-off command loads the model each time. For loops, use `serve` or `mcp`.
- EmbeddingGemma 2 resizes images to its configured budget of up to 280 soft tokens.

## HTTP (anerun serve, default http://127.0.0.1:18180)

```bash
curl -s localhost:18180/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["first", "second"], "input_type": "document", "dimensions": 256}'
curl -s -X QUERY 'localhost:18180/embed_text?input_type=query&format=json' -H 'Content-Type: text/plain' --data-binary 'my query'
curl -s -X QUERY 'localhost:18180/embed_image?format=json' -H 'Content-Type: image/jpeg' --data-binary @photo.jpg
```

`GET /health` is 503 while the model loads (the first start after install or a macOS update compiles for
the Neural Engine, which can take minutes); wait for 200. A 200 with `"status": "degraded"` serves, but some
functions failed the self-test: `max_context_length` there is the text limit that remains.

## MCP

`anerun mcp` (stdio) exposes `embed_text` (`text` or `texts`, `input_type`, `dimensions`,
`encoding_format`), `embed_image` (`data` base64, `images`, or a local `path`), and `similarity`
(`a`/`b` texts or `a_image`/`b_image`/`a_path`/`b_path`). Failures come back as `isError` results with a
structured reason; fix the input as the `reason` says rather than retrying unchanged.
