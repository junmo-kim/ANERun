import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from checkpoint_integrity import verify_files, verify_text_checkpoint


class CheckpointIntegrityTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.files = {}
        self.write("config.json", b"{}", lfs=False)

    def write(self, name, data, lfs=True):
        (self.root / name).write_bytes(data)
        blob = f"blob {len(data)}\0".encode() + data
        self.files[name] = hashlib.sha256(data).hexdigest() if lfs else hashlib.sha1(blob).hexdigest()

    def test_data_only_snapshot_checks_git_and_lfs_bytes(self):
        self.write("model.safetensors", b"original")
        verify_files(self.root, self.files)
        (self.root / "model.safetensors").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "digest mismatch: model.safetensors"):
            verify_files(self.root, self.files)
        self.write("model.safetensors", b"original")
        (self.root / "config.json").write_bytes(b"[]")
        with self.assertRaisesRegex(ValueError, "digest mismatch: config.json"):
            verify_files(self.root, self.files)

    def test_sharded_checkpoint_requires_a_digest_for_each_referenced_shard(self):
        shard = "model-00001-of-00001.safetensors"
        self.write("model.safetensors.index.json", json.dumps({"weight_map": {"w": shard}}).encode(), lfs=False)
        self.write(shard, b"weights")
        verify_files(self.root, self.files)
        del self.files[shard]
        with self.assertRaisesRegex(ValueError, "Unpinned weight shard"):
            verify_files(self.root, self.files)

    def test_unrecognized_revision_is_rejected_before_opening_inputs(self):
        with self.assertRaisesRegex(ValueError, "Unknown checkpoint"):
            verify_text_checkpoint(self.root, "fastino/GLiDE-2B", "main")
