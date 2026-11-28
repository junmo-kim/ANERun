# Model conversion and verification

These scripts build and verify Core ML packages for EmbeddingGemma 2, pplx-embed-v1-0.6b and Qwen3-Embedding-0.6B against pinned public sources. Published packages are available for [EmbeddingGemma 2](https://huggingface.co/junmo-k/embeddinggemma-2-ane), [PPLX](https://huggingface.co/junmo-k/pplx-embed-v1-0.6b-ane) and [Qwen](https://huggingface.co/junmo-k/qwen3-embedding-0.6b-ane). `anerun` does not need them; they exist so that the package can be rebuilt and audited.

| Script | What it does |
|---|---|
| `sources.json` | Every input, pinned: the Hugging Face repositories (revision + sha256) and the evaluation texts and images (GitHub commit, Wikisource page revision or Python package version, path, license, sha256). |
| `fetch.py` | `models`: downloads the selected model inputs at pinned revisions. The default fetches only the named Google checkpoint data files in `sources.json`. A verified local file is reused offline. `data`: downloads the evaluation files (and copies the ones that ship inside pinned packages). Checks every sha256. |
| `eg2.py` | The EmbeddingGemma 2 text encoder re-written in PyTorch for the ANE: `Bucket` (full attention) and `Window` (block-local sliding attention over a band; global attention over all keys, in query and key blocks for the long `winl` functions) modules, plus `Reference`, an fp32 ground truth equal to Hugging Face. |
| `eg2_vision.py` | Converts the Google vision weights into the three fixed Swift functions and extracts the host position table. |
| `host.py` | Executable spec of the host contract (token rows, packing, band bias, long-text feed, vision feed) in numpy. The Swift runtime mirrors it. |
| `build.py` | Builds the package, stage by stage (below). |
| `selftest.py` | Generates `selftest.json` / `selftest.bin` (fp32 expectations) and runs them: `selftest.py run <dist>`. |
| `manifest.py` | Writes `manifest.json` (called by `build.py`). |
| `verify.py` | The package on CPU + ANE through `host.py` against `eg2.Reference` (fp32): `text` (every packing function: single, packed, mixed) and `long` (2049–8192 tokens, including the lengths at and just over each size). |
| `verify_hf.py` | Anchors to Hugging Face: `eg2.Reference` against `transformers` / `sentence-transformers` fp32 (`anchor`), and the image path on CPU + ANE against `transformers` fp32 for all three image budgets (`images`). |
| `corpus.py`, `images.py` | Build the evaluation corpus (`corpus.json`, 12 documents) and image folder (13 images) from the fetched files. |
| `selftest_negative.py` | Optional negative control: rebuilds `winp2048_b512`, `winl4096_b512` and `winl8192_b512` with the usual `repeat_interleave` attention and runs their self-tests on the CPU and on the ANE (results below). Since the long functions' global layers moved to `eg2.blocked_attention`, only `winp2048_b512` still injects the fault into the global layers. |

`scripts/accuracy/` at the repository root is the other half: it measures a running `anerun serve` (the Swift runtime, over HTTP) against the same fp32 reference on the same public data. The EG2 verification modes here use the Python host contract. The pplx verifier uses the Swift server.

## Setup

macOS 15 or later on Apple silicon, Xcode (for `xcrun coremlcompiler`), Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
cd tools/convert
uv venv --python 3.12.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`requirements.txt` pins every package of the environment that built the published package. Conversion and the fp32 references run on the CPU only: no step selects MPS or the GPU (torch has no separate CPU wheel for macOS; the PyPI wheel is used). For EG2, allow space for the raw Google snapshot (1.5 GB), the intermediate text and vision packages, and the compiled distribution. The first `selftest.py run` on a new directory also waits for the ANE compile (about 10 minutes for all functions on an M5 Ultra). Run scripts with `python -I` as shown below to isolate the Python import path. Model loading uses local safetensors and installed SDK classes. EG2 includes the pinned Jinja template used by the Sentence Transformers cross-check. The final corpus row comes from this repository’s pinned MIT-licensed converter and host source.

## Qwen3-Embedding-0.6B

From this directory, with the converter environment above:

```sh
W=../../.cache/qwen
.venv/bin/python -I fetch.py models --model qwen3-embedding-0.6b --dest "$W/models"
HF="$W/models/Qwen3-Embedding-0.6B"
.venv/bin/python -I build.py --model qwen3-embedding-0.6b --hf-dir "$HF" --out "$W/dist"
.venv/bin/python -I qwen_reference.py "$HF" "$W/reference" --parity --long
.venv/bin/python -I qwen_verify.py "$W/dist" "$W/reference" --binary ../../.build/release/anerun
```

The 64-row function packs short sequences with a separate causal mask and EOS selector per input. Longer inputs use the shared pre/post projection schedule with causal attention at 2,048, 8,192 or 32,768 keys. The host keeps the residual stream in float32 and pools the final EOS row. The HTTP verifier checks token IDs, vector norms, normalized prefixes, cosine to the independent CPU reference, and over-limit rejection. It prints one JSON record per result, including the raw API vector and elapsed time. Set `--cap 64` for a short smoke test.

## pplx-embed-v1-0.6b

```bash
W=../../.cache/pplx
.venv/bin/python -I fetch.py models --model pplx-embed-v1-0.6b --dest "$W/models"
HF="$W/models/pplx-embed-v1-0.6b"
.venv/bin/python -I build.py --model pplx-embed-v1-0.6b --hf-dir "$HF" --out "$W/dist"
.venv/bin/python -I selftest.py run "$W/dist" --binary ../../.build/release/anerun
.venv/bin/python -I verify.py pplx "$W/dist" corpus.json --hf-dir "$HF" --binary ../../.build/release/anerun
```

The build checks the pinned revision and source digests, then runs
`convert,merge,compile,assets,selftest,manifest`. Use `--stages` to run a subset and
`--force` to rebuild existing Core ML packages. Intermediate conversion packages
default to `<out>/../work-pplx` and can be placed elsewhere with `--work`. Their
cache keys include the pinned source, graph code, function definitions and
package versions. After a merge, run the compile stage before regenerating the
manifest if the output contains local compiled packages.

`PPLXText.mlpackage` contains the three packed functions, the 2K and 4K fused
functions and the four 8K partitions. The 28 pre/post pairs and the 16K and 32K
weightless attention functions have separate packages. The manifest validates
the saved Core ML names, dtypes and shapes against the Swift package contract.
The assets retain the float32 token table and final norm, tokenizer and configuration.
The decomposed attention graphs score each key block directly. They scale
probabilities by 32 for the value matmul and undo that scale after combining
block means. The gain preserves small FP16 probabilities while the normalized
value matmul fits in range. RMSNorm divides the square root of epsilon by the
row scale before squaring so small and zero rows retain the intended epsilon.

```bash
.venv/bin/python -I test_references.py
.venv/bin/python -I test_pplx_math.py
.venv/bin/python -I ../../scripts/golden/pin_pplx_quantization.py \
  --check ../../Sources/EmbedCore/Support/PPLXQuantization.swift
```

`selftest.py run` invokes Swift `models warm` with the supplied model directory.
It checks all routes by default and accepts `--max-tokens` to verify a smaller cap.
`verify.py pplx` starts one loopback Swift server through the largest corpus length.
Supply a nonempty JSON list of documents with `text`. Optional `tokens` is the HF
token count or the exact token ID list. An optional `vector` supplies 1,024 raw,
unnormalized fp32 coordinates. For text-only documents the verifier computes
fp32 means using the installed Qwen3 model with all keys visible to every token.
The model specifies 1,024 unnormalized base coordinates, mean pooling and a 32K context. See the [pinned Perplexity model card](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b/blob/2c4d510dd4a732063c31a0f70193e35067b51fd8/README.md).

The hard quantizer rounds `127 * tanh(mean)` to the nearest integer (ties to even).
The reference checks the Swift
token counts and requires a cosine of at least 0.995 for every embedding.

## Building the EmbeddingGemma 2 package

```bash
W=../../.cache/eg2                             # any new directory per package version (.cache/ is git-ignored)
.venv/bin/python -I fetch.py models --dest $W/models     # Google raw weights and pinned data files
.venv/bin/python -I build.py --hf-dir $W/models/embeddinggemma-2 --out $W/dist
.venv/bin/python -I selftest.py run $W/dist               # SELFTEST PASS 16/16 ...
```

Any directory works for `--dest` and `--out`; intermediate per-function packages go to `<out>/../work` (`--work` to change). Stages, in order (`--stages` to run a subset; each skips finished work unless `--force`):

1. **convert**: checks the sha256 of `model.safetensors`, then traces each of the 13 functions and converts it with coremltools to an fp16 ML Program (minimum macOS 15), one `.mlpackage` per function:
   - `embed_64` … `embed_1280` (7): full attention, packing;
   - `winp1536_b512`, `winp2048_b512`: windowed attention, packing;
   - `winl3072_b512`, `winl4096_b512`, `winl6144_b512`, `winl8192_b512`: windowed attention, one long text per call.
2. **merge**: merges them into one multifunction `EmbeddingGemma2Text.mlpackage`; identical weight constants are stored once (363 MB instead of 3.4 GB).
3. **compile**: `xcrun coremlcompiler compile` to `EmbeddingGemma2Text.mlmodelc`.
4. **assets**: extracts `embeddings.bf16` (unchanged bf16 rows) from `model.safetensors`, copies `tokenizer.json`, converts the vision encoder and projection into `vision_70`, `vision_140` and `vision_280`, compiles the merged package, and writes the first 1024 learned position rows on each axis to `position_embeddings.f16`. It writes `LICENSE` (Apache-2.0 text from the pinned `transformers` wheel) and `NOTICE`.
5. **selftest**: one golden test per function (16: 13 text, 3 vision), expectations from the fp32 reference (Hugging Face fp32 for the vision functions). The token ids are the tokenized `LICENSE`.
6. **manifest**: `manifest.json` with every file's sha256 and size, function signatures, host contract and provenance.

The vision adapter uses the installed Transformers projection, MLP and rotary layers with fixed-shape PyTorch attention. Swift supplies patch coordinates, learned position embeddings, key validity and spatial pooling weights. The result of each function is a `[1, budget, 512]` fp16 array. See the [pinned Google model card](https://huggingface.co/google/embeddinggemma-2/blob/f9972ac22e696a105c482202646979e33e4b5a1e/README.md).

### Comparing a rebuild to an existing release

Compare `embeddings.bf16`, `tokenizer.json`, `position_embeddings.f16` and the self-test vectors by hash. The position table's SHA-256 is `17de9480acc23638c0634a97ed89e51ed664433256d7d6f109587dcd91c79124`.

To test the new vision graph against saved release expectations, place the existing `selftest.json`, `selftest.bin` and host embedding tables alongside the rebuilt compiled model in a scratch directory. Run `selftest.py run <scratch> --functions vision_70,vision_140,vision_280` and repeat with `--cpu`. Package UUIDs, protobuf field serialization and Core ML compiler metadata may vary across builds.

## Verifying against the original model

```bash
.venv/bin/python -I fetch.py data --dest $W/sources
.venv/bin/python -I corpus.py $W/sources $W/data               # corpus.json, 12 documents
.venv/bin/python -I images.py $W/sources $W/data/images        # 13 images
HF=$W/models/embeddinggemma-2
.venv/bin/python -I verify_hf.py anchor $W/dist $W/data/corpus.json --hf-dir $HF   # reference == official fp32
.venv/bin/python -I verify_hf.py images $W/dist $W/data/images --hf-dir $HF        # image path vs official fp32
.venv/bin/python -I verify.py text $W/dist $W/data/corpus.json --hf-dir $HF        # buckets, packing
.venv/bin/python -I verify.py long $W/dist $W/data/corpus.json --hf-dir $HF        # 2049..8192 tokens (winl)
```

Each prints one line per case and a final `SUMMARY` with the minimum and median cosine per category. The current corpus is `aneembed-corpus/2`. Its first eleven document IDs and texts match v1. Row twelve is `anerun-eg2-converter-v2` (`tools/convert/eg2.py`, one newline, then `tools/convert/host.py`, as of corpus v2; frozen copies in `tools/convert/fixtures/` are pinned in `sources.json`, so later code changes do not change the corpus). This keeps a code document eligible for 8,192-token cases. The published v1 row `hf-eg2-modeling-code` identifies a different text. Existing v1 `corpus.json` files remain valid inputs to the verifiers. Accuracy reference files record the corpus version and input SHA-256 so cross-version comparisons can match individual inputs.

The corpus mixes English, Korean, Japanese, Chinese and French READMEs, the Korean and Japanese constitutions, technical documents and source code; the images are photos, diagrams, a plot and synthetic patterns of odd sizes. For the Swift runtime, run `scripts/accuracy/reference.py` with the same `corpus.json` and image folder, then `scripts/accuracy/compare.py` against `anerun serve`.

## ANE caveats

The Neural Engine compiler runs on the user's Mac, per macOS build, and a miscompiled graph raises no error: it returns wrong numbers. What we hit, and what the code does about it:

- **Attention uses the fold layout** (`eg2.attend_fold`): the query heads that share a K/V head are folded into the row dimension. The usual `repeat_interleave` layout returns zeros for query heads 1–3 of the global layers (one K/V head) at long sequence lengths; an explicit K/V copy is wrong as well. `selftest_negative.py` reproduces it: rebuilt with `repeat_interleave`, `winp2048_b512`, `winl4096_b512` and `winl8192_b512` fail their self-tests on CPU + ANE (min cosine 0.960, 0.952, 0.885; M5 Ultra, macOS 27.0.1), while on the CPU the first two pass (0.99994, 0.99953) and `winl8192_b512` scores 0.9975 (fp16 precision of that layout). Do not change the attention layout without running the self-test on the ANE. These failures were measured with the v0.1 package; the global layers of `winl` now use `eg2.blocked_attention` (below), which `selftest_negative.py` does not replace, so its `winl` rebuilds put `repeat_interleave` in the sliding layers only.
- **Masks are runtime inputs and RoPE tables are graph constants.** A sliding-window mask built into the graph and a runtime positions input both produced wrong results. The windowed functions therefore take the band as an input too (`band_bias`, `[1,1,S,1536]`, see `host.band_bias`).
- **Long global attention runs in blocks and divides by a matmul** (`eg2.blocked_attention`, `winl` only). The ANE softmax loses accuracy as the number of keys grows, and one `[4·S, S]` score tensor per global layer is slow. Each 128-row query block attends to key blocks of 2048 keys (1024 for `winl3072`), combined flash-style, and every denominator is `exp(s - max) @ 1` instead of a sum. In fp32 it equals the full attention (cos 1.00000000 against `eg2.Reference` at 3000 and 6000 tokens). On an M4 (minimum cos against fp32 over 9 texts, 3 for the new `winl8192`; p50 at the full length):

  | function | before | after |
  |---|---|---|
  | `winl3072_b512` | 352 ms, 0.999967 | 345 ms, 0.999977 |
  | `winl4096_b512` | 507 ms, 0.999959 | 491 ms, 0.999977 |
  | `winl6144_b512` | 961 ms, 0.99994 | 829 ms, 0.999976 |
  | `winl8192_b512` | 1457 ms, 0.99989 | 1240 ms, 0.999979 |

  "Before" is the v0.1 package's function, "after" a standalone conversion of the same function with this code (`build.convert_one`); both were timed over 5 calls at the full length and compared with `eg2.ref_embed` by a harness outside this repository, and the release measures the rebuilt package again with `verify.py long` and `anerun bench --buckets`; the texts include the lengths just over each smaller size (2049, 3073, 4097, 6145). The first compile takes 1.3–1.6x longer (`winl8192_b512` 170 s → 274 s on the M4). `winp` and `embed_S` are unchanged. The build skips stages and per-function packages that already exist, so build the next package into fresh `--out` and `--work` directories (or with `--force`); an older `winl` package would pass the self-test as well, since the inputs and expected vectors are the same.
- **GELU is written out where precision matters** (`eg2.gelu`, `eg2_vision.mlp`). Core ML's `gelu` op runs on the Neural Engine with a table-like approximation that is worst near zero, where most MLP inputs lie (mean absolute error 7.2e-4 after a real projection, up to 6.2e-3 on [-1, 1); the explicit `x·(0.5 + 0.5·tanh(x·(c + c′x²)))` form: 3.0e-4). The Neural Engine's `sigmoid` is less precise still, so sigmoid-based forms do not help. Written out, each MLP adds a few elementwise passes over its widest tensor: on an M4, 0–2 % for `embed_64` / `embed_128`, about 10 % for the `winl` functions and 13–20 % for `embed_256` … `winp2048`. `eg2.EXACT_GELU_SIZES` therefore writes it out for 64, 128 and the long functions and keeps the op for 256–2048 rows; the vision functions always write it out. Package results on an M4 against fp32 references (`verify.py` for text, `verify_hf.py images` for images): `embed_64` median cosine 0.999974 → 0.999995, long texts minimum 0.999976 → 0.999996, images at budget 280 median 0.999979 → 0.999984.
- **Windowed sliding layers slice, they do not pad.** Each 512-row query block attends to the key rows `b*512-512 … b*512+1023` taken as static slices clipped to `[0, S)`; padding the keys inside the graph (`F.pad`) gave the same numbers but made the layer about 3x slower on the ANE.
- **Do not trust `MLComputePlan` for correctness.** It reports which device each operation is planned on, not whether the ANE computes it correctly. Only comparing outputs catches a miscompile.
- **Run the self-test** (`selftest.py run <dist>`, or `anerun`, which runs it at load) on every Mac and macOS version you ship to, with `cpuAndNeuralEngine`. `selftest.py run <dist> --cpu` runs the same tests on the CPU.
- Fixed shapes only: variable (enumerated) shapes move the whole graph to the CPU.

## License

The scripts are part of ANERun and MIT licensed (see `LICENSE` at the repository root). The package they build contains EmbeddingGemma 2 by Google (Apache-2.0, Gemma Prohibited Use Policy) with its text and vision weights converted locally. The package includes `LICENSE` and the `NOTICE` in this directory.
