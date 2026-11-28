"""Verify an offline weights-and-tokenizer snapshot against the pinned Hub revision."""
import hashlib
import json
from pathlib import Path


def verify_text_checkpoint(checkpoint, repo, revision):
    sources = json.loads(Path(__file__).with_name("checkpoints.json").read_text())
    source = sources.get(repo)
    if source is None or source["revision"] != revision:
        raise ValueError(f"Unknown checkpoint: {repo}@{revision}")
    verify_files(checkpoint, source["files"])


def verify_files(checkpoint, files):
    for name, digest in files.items():
        if Path(name).name != name:
            raise ValueError(f"Invalid checkpoint filename: {name}")
        path = checkpoint / name
        if not path.is_file() or path.with_name(name + ".aria2").exists():
            raise ValueError(f"Incomplete checkpoint file: {name}")
        # HF uses raw SHA-256 for LFS payloads and Git blob IDs for Git objects.
        with path.open("rb") as stream:
            if len(digest) == 64:
                actual = hashlib.sha256()
            elif len(digest) == 40:
                actual = hashlib.sha1(f"blob {path.stat().st_size}\0".encode("ascii"))
            else:
                raise ValueError(f"Invalid pinned digest: {name}")
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError(f"Checkpoint digest mismatch: {name}")

    index = "model.safetensors.index.json"
    if (checkpoint / index).exists():
        if index not in files:
            raise ValueError("Unpinned weight index")
        shards = set(json.loads((checkpoint / index).read_text())["weight_map"].values())
        if not shards or any(shard not in files or len(files[shard]) != 64 for shard in shards):
            raise ValueError("Unpinned weight shard")
    elif len(files.get("model.safetensors", "")) != 64:
        raise ValueError("Unpinned model.safetensors")
