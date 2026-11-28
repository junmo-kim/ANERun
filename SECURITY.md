# Security Policy

## Threat model

ANERun computes embeddings locally. It sends nothing anywhere except when you run
`anerun models install`, which downloads a pinned model revision from Hugging Face and checks every
file against SHA-256 hashes (the manifest's own hash is compiled into the binary once a model is
published). CI fails if a networking API appears outside `HTTPServer.swift` and `ModelDownloader.swift`.

**`anerun mcp` (stdio)**: no listening socket. Tools may read local image files given as `path`.

**`anerun serve` (HTTP)**: listens on `127.0.0.1:18180` by default. There is **no authentication and no
TLS**; binding another address lets anyone who can reach the port use it, and the server logs a warning
when you do. Over HTTP, `path` arguments are refused, so the server cannot be used to read files. Request
bodies are capped (`--max-body`, 72 MiB), image data URLs at 8 MiB, decoded images at 40 megapixels,
headers at 64 KiB; idle connections are closed after 60 seconds.

The realistic attack surface:

- a malicious image (decoder bug or resource exhaustion): JPEG is decoded by ANERun's own decoder,
  other formats by ImageIO;
- a malicious JSON body (parser limits: nesting depth 256, sizes as above);
- an unauthenticated caller on a network where you exposed `serve`.

## Reporting a vulnerability

Please report security issues privately via GitHub's *Security → Report a vulnerability* (private
advisory), not a public issue. Include reproduction steps and the input that triggers the problem.
