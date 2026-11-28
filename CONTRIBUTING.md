# Contributing to ANERun

ANERun is a **single-binary, zero-dependency** Swift tool that runs embedding models on the Apple Neural Engine. Staying small and dependency-free is a core goal, so changes that add third-party packages, GPU or CPU inference paths, or other models are unlikely to land without a strong reason.

## Build and test

```bash
swift build -c release
swift test                                                  # must stay green
Tests/Integration/release-binary-launch-smoke.sh .build/release/anerun
ruby Tests/Integration/release-workflow-contract.rb
```

Use Xcode 16 or later; the deployment target is macOS 15 (the first release with multifunction Core ML models). APIs newer than macOS 15 must be behind `@available`.

What runs where:

| Tests | Needs | CI |
|---|---|---|
| `EmbedCoreTests`: JSON, tokenizer rules and golden ids, token table, packing, bias/pool builders, image preprocessing, config, store | nothing | yes |
| `ServiceTests`: service, HTTP/OpenAI contract, QUERY, MCP through the fake runtime | nothing | yes |
| `ModelTests`: real model on the Neural Engine | an EmbeddingGemma 2 package (`ANEEMBED_TEST_MODEL_DIR` or an installed model) | skipped (`CI` is set) |
| `PPLXHardwareTests`: exact lengths, full and prefix cosines, route boundaries, per-layer timing | a verified pplx package and an fp32 reference corpus | skipped (`CI` is set) |
| `Tests/Integration/*.sh`, `serve-contract.py`: the built binary over real sockets, fake runtime | `python3` | yes |

`serve-contract.py` also runs against a real server: `python3 -I Tests/Integration/serve-contract.py http://127.0.0.1:18180 [--reference ref.json]`.

Run the pplx hardware suite with `ANEEMBED_TEST_MODEL_DIR=/path/to/package ANEEMBED_PPLX_REFERENCE=/path/to/reference.json swift test -c release --filter PPLXHardwareTests`. The reference JSON contains objects with `name`, `text`, `tokens` and the 1,024-coordinate raw fp32 mean `vector`. Use the pinned model and tokenizer from the converter instructions. Include an 8,192-token sample for the forced-route test and three diverse texts of exactly 32,768 tokens. The suite loads the full cap and reports per-layer host and prediction times. A full run on an M4 can take tens of minutes. [Model converter](tools/convert/README.md)

## Golden data

Test fixtures are generated, never hand-edited, and contain no third-party text or images.

```bash
uv venv /tmp/anerun-golden
VIRTUAL_ENV=/tmp/anerun-golden uv pip install tokenizers transformers torch torchvision pillow numpy
/tmp/anerun-golden/bin/python -I scripts/golden/make_tokenizer_golden.py <tokenizer.json> Tests/Fixtures/tokenizer-golden.jsonl.xz
/tmp/anerun-golden/bin/python -I scripts/golden/make_image_golden.py Tests/Fixtures/images
```

For a larger local tokenizer check (not stored), write any corpus as one JSON string per line, then:

```bash
/tmp/anerun-golden/bin/python -I scripts/golden/make_tokenizer_golden.py <tokenizer.json> /tmp/extra.jsonl.xz --lines corpus.jsonl
ANEEMBED_EXTRA_GOLDEN=/tmp/extra.jsonl.xz swift test --filter matchesExtraGoldenCorpus
```

## Pull requests

- One logical change per PR; keep the diff focused, with tests for behaviour changes.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `build:`, `chore:`, with an optional scope such as `feat(server):`); the subject is short and lowercase, the body says why. The release notes group the subjects by type.
- Performance changes come with a measurement (before/after, machine, command).

## Releasing

A release is cut by pushing a `vX.Y.Z` tag. `.github/workflows/release.yml` tests and builds the stripped binary with the Core AI SDK, verifies the same packaged tarball (checksum, contents, signature, version, smoke test) on macOS 15, and only then publishes one `anerun-vX.Y.Z-macos-arm64.tar.gz` with its `.sha256`.

For a local ad-hoc-signed release candidate, run the following on Apple silicon with Xcode 27 selected. The archive contains one executable named `anerun`. The signature uses identifier `anerun`. Keep this basename and the final install path for device qualification.

```sh
swift build -c release --product anerun -Xswiftc -DANERUN_REQUIRE_COREAI -Xlinker -s
scripts/package-rc.sh .build/release/anerun dist
(cd dist && shasum -a 256 -c anerun-v0.3.0-macos-arm64.tar.gz.sha256)
verify=$(mktemp -d)
tar -xzf dist/anerun-v0.3.0-macos-arm64.tar.gz -C "$verify"
codesign --verify --strict "$verify/anerun"
ANERUN_CHECK_RELEASE_COREAI=1 Tests/Integration/release-binary-launch-smoke.sh "$verify/anerun"
```

1. Bump `aneembedVersion` in `Sources/EmbedCore/Support/Version.swift`.
2. Rename the unreleased heading to `## vX.Y.Z` in `CHANGELOG.md`; that section becomes the release-notes keynote.
3. Commit (`chore(release): vX.Y.Z`), tag and push the tag.
4. After the release succeeds, update `junmo-kim/homebrew-tap`'s `Formula/anerun.rb` with the published tarball URL and SHA-256. For the first ANERun release, add `"aneembed": "anerun"` to the tap's `formula_renames.json`. Install both command names from the archive and check a fresh install and an upgrade from an existing `aneembed` keg.
