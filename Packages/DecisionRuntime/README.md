# DecisionRuntime

A Swift library for macOS 27 on Apple silicon. It accepts a resolved local bundle URL. Valen-4B and Elephant-1 accept text with an optional image. GLiDE-2B, GLiDE-9B, K-Decision, KnowLine and Jeffan accept text. Each request preserves the order of questions and candidates. Pass JSON state as its serialized text, preserving the desired property order and whitespace.

## Prepare a bundle

To export Valen-4B or GLiDE-2B/9B, use a local snapshot of the pinned Hugging Face model. The snapshot needs model data and tokenizer files. The exporter checks the snapshot against the published file digests in `Tools/checkpoints.json`. Run the following from this package directory on an Apple silicon Mac with Xcode 27 and Apple's Core AI Python conversion SDK. The exporters use PyTorch and safetensors. The optional text exporters and CPU comparisons also use Transformers.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r Tools/requirements.txt
python Tools/export_base.py valen-4b "$CHECKPOINT" "$EXPORT"
```

For source graphs, the packer retries a partially placed stage up to eight total attempts. Set `--compile-attempts N` to change the per-stage limit. A source hash mismatch or a compiler error stops the build immediately.

Use an exported decision directory containing `model.json`, its Core AI stages and a `checkpoint` directory. The input can contain chip-independent `.aimodel` graphs or AOT `.aimodelc` assets for the destination chip. Building from graphs needs Xcode 27. The packer verifies the source identities and placement of all compiled entry points. It preserves the selected checkpoint's licenses. It creates a new directory and publishes it only after preparation succeeds.

```sh
python3 Tools/prepare.py "$EXPORT" "$BUNDLE" --architecture "$ANE_ARCH"
swift run -c release decision-bench prepare "$BUNDLE"
```

For GLiDE-9B, use `python Tools/export_base.py glide-9b "$CHECKPOINT" "$EXPORT"` with the pinned publisher snapshot. The export includes 61 embedding banks, the recurrent blend, readout and response statistics alongside the 32 backbone stages. The packer copies the text rotary table and rewrites all 96 asset names relative to the installed bundle. A prepared GLiDE-9B bundle requires no host embedding or readout weights. Older backbone-only GLiDE-9B bundles require a new export and prepare.

GLiDE-9B uses the ANE for scaled token embeddings, hidden updates, projection and probabilities. Swift tokenizes, addresses the embedding banks, constructs masks, copies tensors and selects the highest-probability label. Set `ANERUN_TRACE_DECISION=1` to record the Core AI functions dispatched per decision. For qualification, compare each candidate probability with the pinned CPU fixture using an inclusive absolute tolerance of 0.001.

Perform the second command at the permanent installation path on the destination machine, using the same program identity that serves requests when integrating. It specializes the model and loads its functions to populate the OS cache. The prepare process can then exit. Installed inference requires only the Swift executable and bundle.

## Use the engine

```swift
import Foundation
import DecisionRuntime

func refundDecision(bundleURL: URL, idleTimeout: Duration?) async throws -> DecisionResponse {
    // Installation, once at the final path.
    try await DecisionEngine(bundleURL: bundleURL).prepare()

    // Server lifetime. The first decision loads the selected bundle on demand.
    let engine = DecisionEngine(bundleURL: bundleURL, idleTimeout: idleTimeout)
    let input = DecisionInput(state: "The keyboard arrived broken. I want a refund.", questions: [
        DecisionQuestion(name: "intent", kind: .choice, instructions: "What does the customer request?",
                         candidates: [
                            DecisionCandidate(key: "refund", description: "Refund"),
                            DecisionCandidate(key: "replacement", description: "Replacement"),
                         ])
    ])
    let response = try await engine.decide(input)
    await engine.unload()
    return response
}
```

`load()` loads the tokenizer and the profile's local resources. Backbone functions load on first use. The engine serializes requests and unloads, including across suspensions inside Core AI. A later decision reloads after explicit unload or idle expiry. The caller supplies the idle timeout. Omitting it keeps the session resident until explicit unload or engine release.

`DecisionInput` defaults to `truncate: true`. The same default applies to the optional JSON boolean `truncate` on `anerun decide --stdin` and `POST /v1/decisions`. When the full prompt exceeds the bundle context, the engine keeps a whole-character prefix of the state that fits every prompt. It preserves the questions, candidates, image tokens and prompt scaffolding, and includes GLiDE's eight-token reserve in each prompt's budget. Set `truncate` to `false` to reject overflow. An invalid request or a prompt that exceeds the budget with empty state returns an input error. A clipped state may end inside serialized JSON.

Successful clipped responses include `usage.truncated: true`. Unclipped responses omit that field. `usage.input_tokens` counts the logical input tokens. Elephant counts the shared state prefix once per request. `usage.prompt_tokens_per_second` divides the number of tokens processed by the backbone by processing time in the loaded session, including prompt preparation, vision and decision inference. For Elephant, the processed count includes the recomputed prefix for every question and is available as `internal_usage.compute_tokens`. Model loading and time waiting for the session are excluded. llama-swap v273 can proxy `/upstream/<model>/v1/decisions`; its Activity recorder does not meter that route.

### Elephant-1 prompt provenance

ANERun renders Elephant-1 prompts in Swift using the local bundle's tokenizer and slot table. The pinned Base Labs release ships the chat-role, image and assistant markers in `chat_template.jinja`, and specifies block-bidirectional attention and slot-LM readout in model configuration and `serve.patch`. The publisher's reference pins OneJev for the rest of the frozen `qev-labels-v2` template, including the system and question wording, fallback descriptions and slot order. The model snapshot on its own provides only the chat wrapper and model settings. The ANERun converter, runtime and CPU reference require no OneJev installation. [Publisher snapshot](https://huggingface.co/baselabs/elephant-1/tree/b3f8a93462fb1051a9de031e94a7ff380e254072) · [OneJev template](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/prompt.py) · [NOTICE](NOTICE)

Different wording would change the tokens given to the trained model. The existing CPU and ANE references use the publisher-pinned wording.

## Verify

`swift test` checks the token IDs for 19 English, Korean, JSON and image inputs, including the Valen FP32 Mixer's token spans. The reference probabilities come from the pinned CPU models listed in NOTICE. The synthetic image is a red circle on a white background. The miniature BPE table in the tests retains the merge ancestry of the fixture token IDs. Its parent tokenizer SHA-256 is `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.

```sh
DECISION_TEST_BUNDLE="$BUNDLE" DECISION_TEST_MODEL=valen-4b swift test -c release
swift run -c release decision-bench golden "$BUNDLE" Tests/DecisionRuntimeTests/Fixtures/goldens.json valen-4b
```

The executable emits epoch timestamps around each `decide` call. `Tools/measure.py` accepts the executable, bundle path, fixture directory, model name and output file. It measures loopback HTTP send through response JSON parsing with a persistent Swift worker. Each case runs five unload/reload cycles with two additional warm requests per cycle. ANE wired tags cover the entire machine. The HTTP shim is a benchmark fixture.

## Optional experimental text checkpoints

`k-decision-9b` selects K-Decision v1. A single causal pass scores the symbols before
all question markers. Choices accept 2 to 26 candidates. Predicates return P(Y).
The published server uses the legal-symbol softmax at temperature 1. Images and score
questions are unsupported. Export a local, pinned checkpoint using the same Core AI conversion environment.

```sh
python Tools/export_text.py k-decision-9b "$CHECKPOINT" "$EXPORT"
python Tools/prepare.py "$EXPORT" "$BUNDLE" --architecture "$ANE_ARCH"
```

The exporter reads local weights, checks the pinned configuration and tokenizer, and writes a new source bundle. The fixture file
`Tests/DecisionRuntimeTests/Fixtures/text-decisions.json` includes predeclared
answers for independent smoke tests. Each `request` can be sent to `POST /v1/decisions`.

`knowline-9b-gen1` uses one causal pass per question, repeats short text requests
per the upstream 3,000-token rule, and reads a space-prefixed option label.
State clipping checks both layouts so a longer prefix that fits in a single read
under that rule can be retained. The checkpoint folds its calibration into the
final norm. Use temperature 1. Choice
and score return entropy-based confidence. For a score the value is the expected
level. Binary questions return P(true). The local profile accepts text only.

`jeffan-9b-1211` implements the two-pass native checkpoint configuration.
It renders each question separately, uses noncausal full attention, and repeats
layer blocks 4–7, 8–11 and 12–15 with the upstream residual blend. The last token
selects an option. The native Decision Index readout returns the expected score
level and P(true) for binary questions. This text profile does not process images.

To preserve previously qualified GLiDE-9B graphs, assemble their retained RELEASE sources with Python 3.13. Supply the verified prepared directory, the original portable backbone directory and the source directories containing the matching operator assets. The command checks every graph against the source hash in the prepared manifest before making the new source directory available.

```sh
python Tools/pack_glide9_source.py "$PREPARED" "$BACKBONE_SOURCE" "$EXPORT" \
  --operator-dir "$BANKS_AND_BLEND" --operator-dir "$HEAD_SOURCE" --operator-dir "$STATS_SOURCE"
```

### Elephant-1 on h17d

The publisher export and prepare commands are in [the Elephant exporter guide](Tools/elephant/README.md). For a rebuild from prepared assets, use the audited h17d device baseline with its 24 backbone stages, vision graph, position and rounding functions, rotary and fraction tables, and publisher weights. Build a new prepared directory using the RELEASE converter and Xcode 27.

```sh
python Tools/elephant/pack.py "$ELEPHANT_DEVICE_BASELINE" "$ELEPHANT_PREPARED"
anerun models install --model elephant-1 --from "$ELEPHANT_PREPARED"
```

The packer retains the baseline graphs and checks their placement. It exports 61 gain-256 token embedding banks, the compensating device scale and the majority-tail slot readout. The runtime resolves weight-specific entry names from the manifest. It uses ANE functions for learned image positions, feature scaling, backbone updates, slot probabilities and response statistics. The host decodes images, prepares pixels and masks, addresses constant tables, and maps selected slots to labels.

### Portable source installation

A portable-v1 revision contains one architecture-free runtime `model.json`, root-level
`.aimodel` directories for every declared graph and final-form sidecars, including
`bpe.bin`, `tokens.json`, model-specific tables/readout and notices. It uses the packed
descriptor's field names and exact exported function names. Each asset record has
`asset`, `functions` and `source_hash` (the SHA-256 of `main.mlirb`, also stored as
32 bytes in `main.hash`). Vision uses `vision_asset`, `vision_functions` and
`vision_source_hash`. Auxiliary graphs include `residual_rounding[]`,
`embedding_lookup.assets[]` and the model's single-asset records.

On the publisher, index a clean source root for each independently qualified architecture.

```sh
python3 tools/decision_manifest.py "$SOURCE_ROOT" --model elephant-1 --portable --architecture h17d
```

On the target Mac, use macOS 27, Xcode 27 and the Metal Toolchain selected by
`xcrun --find coreai-build`. The installer uses Swift and the local compiler.

```sh
anerun models install --model elephant-1 --from "$SOURCE_ROOT" --data-dir "$ISOLATED_STORE"
# Alternatively, use a portable inventory published at the root of an immutable HF revision.
anerun models install --model elephant-1 --repo "$HF_REPO" --revision "$HF_COMMIT" --data-dir "$ISOLATED_STORE"
```

A fresh install copies or downloads the listed source files into destination-side staging,
checks sizes and hashes, and compiles each asset for this device. It releases each staged
source asset once its AOT has passed verification. It verifies every
declared entry's placement, compiled identity and resource. The final model directory contains local AOT, sidecars and the
source inventory and receipt. Core AI then prepares at that permanent path. Before
activating, a compiled function load that throws Core AI's `nilError` gets one targeted
recovery attempt. The installer verifies that asset's native resource and placement,
retrieves and rehashes only the affected source asset, loads its function in a separate process, then repeats
ordinary compiled prepare in a fresh process. The installer checks native entry symbols and resource bytes against local
AOT, including the graph payload for resource-free assets. Unknown cache layout or a
mismatch fails installation and preserves the previous active pointer. A failed prepare
or audit may leave a complete inactive derived revision for retry. Budget space for
staged source, growing AOT and the separate native cache; same-volume APFS clones may
reduce source-copy costs. An existing prepared AOT directory can still be installed
with `--from` as a separate packed installation.
