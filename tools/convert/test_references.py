"""Offline checks for the model fetch contract and bidirectional CPU reference."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from transformers import Qwen3Config, Qwen3Model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import load_sources
from pplx_reference import means


class ReferenceTests(unittest.TestCase):
    def test_model_sources_are_pinned_data(self):
        sources = load_sources()
        self.assertEqual([m["id"] for m in sources["models"]], ["google/embeddinggemma-2"])
        for model in [*sources["models"], sources["pplx"], sources["qwen"]]:
            self.assertRegex(model["revision"], r"^[0-9a-f]{40}$")
            self.assertIn("model.safetensors", model["sha256"])
            for name, digest in model["sha256"].items():
                self.assertIn(Path(name).suffix, {".safetensors", ".json", ".model", ".md", ".txt", ".jinja"})
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_versioned_code_fixture_comes_from_the_repository(self):
        sources = load_sources()
        corpus = sources["corpus"]
        fixture = corpus["code_fixture"]
        for entry in fixture["sources"]:
            path = Path(__file__).resolve().parents[2] / entry["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
        self.assertEqual((corpus["historical_code_fixture"]["version"], corpus["historical_code_fixture"]["name"]),
                         ("aneembed-corpus/1", "hf-eg2-modeling-code"))
        self.assertNotEqual(fixture["name"], corpus["historical_code_fixture"]["name"])
        self.assertTrue(all(not item["path"].endswith(".py") for item in sources["packaged"]))

    def test_pplx_uses_full_attention_with_only_data_files(self):
        torch.manual_seed(23)
        config = Qwen3Config(vocab_size=32, hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=4, intermediate_size=32, num_hidden_layers=2, use_cache=False)
        model = Qwen3Model(config).eval()
        ids = torch.tensor([[1, 9, 2, 3, 7]])
        with torch.inference_mode():
            all_keys = torch.ones(1, 1, 1, ids.shape[1], dtype=torch.bool)
            expected = model(ids, attention_mask={"full_attention": all_keys}).last_hidden_state.mean(1)[0]
            causal = model(ids).last_hidden_state.mean(1)[0]
        self.assertFalse(torch.allclose(expected, causal))
        with TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            path = Path(tmp) / "config.json"
            spec = config.to_dict()
            spec.update(model_type="bidirectional_pplx_qwen3", use_bidirectional_attention=True,
                        auto_map={"AutoConfig": "configuration.Custom", "AutoModel": "modeling.Custom"})
            path.write_text(json.dumps(spec))
            actual, = means(tmp, ids.tolist())
        torch.testing.assert_close(torch.from_numpy(actual), expected, atol=2e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
