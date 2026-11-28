# Export Elephant-1

Use a local snapshot of `baselabs/elephant-1` at revision `b3f8a93462fb1051a9de031e94a7ff380e254072` and the Core AI conversion environment described in the package README. The exporter extracts the 256 slot-label rows from the language-model head. The RELEASE exporter folds these rows into the masked device readout. The model uses 61 embedding banks with gain 256, compensated on ANE. Every backbone stage applies the three FP16 rounding functions.

Run these commands from `Packages/DecisionRuntime`.

```sh
python3 Tools/elephant/export.py "$CHECKPOINT" "$EXPORT"
python3 Tools/prepare.py "$EXPORT" "$BUNDLE" --architecture "$ANE_ARCH"
```

The default export supports 1,024 tokens per state and question combined. Set
`--tiers` on the exporter to build other attention capacities up to 1,024 tokens. Image inputs
use the checkpoint's Qwen processor geometry, with at most 1,024 vision
patches per image. By default the runtime keeps the longest state prefix that
fits the packed capacity, preserving the question, candidates and image.
Set `truncate` to `false` for strict overflow rejection.
It prepends the supplied image and a newline to the text inside the state
block. Questions branch from the same tokens with separate recurrent state.
The runtime currently recomputes the prefix for each branch and encodes the
image once. `usage.input_tokens` counts the prefix once per request.
`internal_usage.compute_tokens` and the throughput numerator count the prefix
for every branch.

To score the English, Korean, structured-state and image regression requests on CPU, run the following with PyTorch, Transformers and Pillow. The script loads the Qwen3.5 model from local data files, applies the block attention mask and selects the slot logits. Its output includes token layouts, scores and the original requests.

```sh
python3 Tools/elephant/reference.py "$CHECKPOINT" Tests/DecisionRuntimeTests/Fixtures/red-circle.png "$REFERENCES"
```


On h17d the qualified backbone uses scale-64 recurrence, block masks and 24
physical stages. The 12 stages with overlapping GLiDE cache keys include a
digest of their lifted weights in their entry names. The remaining stages use
their original names. The four-input vision position operator consumes integer
coordinates and FP16 fractions. The five-input result operator computes the
response statistics after the majority-tail head.

For a publisher rebuild, compare the compiled keys, entry attributes and native
resource hashes against the qualified prepared bundle. Converter exports can
differ in serialized source bytes. Verify the six GLiDE and four Elephant
original request fixtures through the signed executable before substituting a
new product. The absolute tolerance is 0.001 for every candidate probability.
