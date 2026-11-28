# Changelog

## v0.3.1

[Valen-4B](https://huggingface.co/junmo-k/valen-4b-ane) brings local text and image decisions to the M4 Mac mini (h16g) on macOS 27. Across nine text and image inputs, every choice and token count matched the original model on both first and repeated requests. The largest candidate probability difference was 0.042 percentage points. On the tested 16 GiB Mac, six short text requests took 0.38–0.45 seconds on repeat. A 133-token image request took 0.82 seconds on repeat, and a larger 768-patch image took 1.42 seconds in a separate traced run. If you installed a v0.3.0 Valen bundle, reinstall with `anerun models install --model valen-4b` on h16g. Other chips require a matching local device-operator bundle.

## v0.3.0

ANERun can now answer questions about local data with [Elephant-1](https://huggingface.co/junmo-k/elephant-1-ane) or [GLiDE-9B](https://huggingface.co/junmo-k/glide-9b-ane) on the Apple Neural Engine. Both models accept text and structured state, and Elephant-1 also accepts images. Install a model once, then try it in the browser playground or send a request to `POST /v1/decisions` while your embedding server stays online. On an M5 Ultra Mac Studio running macOS 27, all choices across ten test inputs matched the original models, and every candidate probability was within 0.1 percentage points of its reference.

## v0.2.0

ANEEmbed is now ANERun. Alongside EmbeddingGemma 2 for text and images, ANERun supports two text embedding models with up to 32,768 tokens: PPLX produces compact int8 vectors, and Qwen3-Embedding supports retrieval instructions and configurable vector dimensions. Use `anerun playground --model ID` to rank your own documents against a query on your Mac. To keep using an existing EmbeddingGemma 2 installation, change the executable to `anerun` and add `--model embeddinggemma-2`.

## v0.1.1

`POST /tokenize` and `POST /detokenize` in llama-server's format (`add_special`, `parse_special`, `with_pieces`), so clients that measure or split their texts with llama-server's tokenizer API work unchanged; for EmbeddingGemma 2 the responses match llama-server's. Content is limited to 1 MiB, and invalid requests are refused with a `400` where llama-server is lenient. `docs/PERFORMANCE.md` now compares the M4, M5 Max and M5 Ultra in one table, measured the same way. See the README for details.

## v0.1.0

First release. EmbeddingGemma 2 text and image embeddings on the Apple Neural Engine from one dependency-free Swift binary: an OpenAI-compatible `/v1/embeddings` server, MCP over stdio and HTTP, and CLI commands. Nothing runs on the GPU and idle memory stays around 200 MB; texts up to 8192 tokens run entirely on the Neural Engine, and every function is checked against reference vectors before it serves. See the README for setup, limits and measurements.
