# Qwen tokenizer fixtures

`tokenizer.json.xz` contains the unmodified `tokenizer.json` from the
[Perplexity pplx-embed-v1-0.6b revision](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b/tree/2c4d510dd4a732063c31a0f70193e35067b51fd8)
(MIT). Its uncompressed SHA-256 is `c6fb5c5bbba5fa5f8332edfb6d8aa67bd7fb3d75365b1765f108201698eaebf5`.

`golden.jsonl.xz` preserves the tokenizer spike's 6,220 rows and 551,244 IDs.
Its uncompressed SHA-256 is `8830d983af6e957dd7a4a2b2cefaa876fd3f646dc5969cfb7875aee9a8953731`.
The generator uses synthetic cases and seeded Unicode samples, plus one
138,301-byte excerpt reconstructed from the public ANEEmbed v0.1.1 commit
[`b1044567ca6a59ba467c56f9db50dfe91bed2656`](https://github.com/junmo-kim/ANEEmbed/commit/b1044567ca6a59ba467c56f9db50dfe91bed2656).

The reference is [Hugging Face tokenizers 0.23.3](https://pypi.org/project/tokenizers/0.23.3/).
The golden IDs use `add_special_tokens=True` and the default
`encode_special_tokens=False`, corresponding to `parseSpecial: true`.
The Swift tests also cover `addSpecial: false` and the focused reference for
`parseSpecial: false`, which sets HF's `encode_special_tokens=True`.
The golden `pieces` record the isolated regex split of the raw text, before
normalization and added-token handling.

To regenerate from the repository root, use Python 3.12 with
`tokenizers==0.23.3` installed and the v0.1.1 commit available locally.

```sh
curl -fL -o /tmp/pplx-tokenizer.json \
  https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b/resolve/2c4d510dd4a732063c31a0f70193e35067b51fd8/tokenizer.json
python Tests/Fixtures/Qwen/make_golden.py /tmp/pplx-tokenizer.json
python scripts/golden/pin_qwen_classes.py
```
