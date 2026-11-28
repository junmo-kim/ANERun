#!/usr/bin/env bash
# Package a locally built arm64 anerun. The same signed payload and version produce the same archive.
set -euo pipefail
binary=${1:?usage: package-rc.sh BINARY OUTPUT_DIRECTORY}
out=${2:?usage: package-rc.sh BINARY OUTPUT_DIRECTORY}
mkdir -p "$out"
stage=$(mktemp -d "$out/.anerun-package.XXXXXXXX")
trap 'rm -rf "$stage"' EXIT

install -m 755 "$binary" "$stage/anerun"
codesign --force --sign - --identifier anerun --timestamp=none "$stage/anerun"
codesign --verify --strict "$stage/anerun"
version=$("$stage/anerun" --version)
[[ "$version" =~ ^anerun\ ([0-9]+\.[0-9]+\.[0-9]+)$ ]]
name="anerun-v${BASH_REMATCH[1]}-macos-arm64.tar.gz"
test "$(lipo -archs "$stage/anerun")" = arm64
TZ=UTC touch -t 200001010000 "$stage/anerun"
COPYFILE_DISABLE=1 tar --no-xattrs --uid 0 --gid 0 --uname root --gname wheel \
  -C "$stage" -cf - anerun | gzip -n >"$stage/$name"
(cd "$stage" && shasum -a 256 "$name" >"$name.sha256")
mv "$stage/$name" "$stage/$name.sha256" "$out/"
