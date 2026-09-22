"""Offline correctness checks; no pretrained models or private data required."""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import algo
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class SteeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.train_from_iterator(["def f(x): return old(x) new(x) clean(x)"],
                                      trainers.BpeTrainer(vocab_size=300,
                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                          special_tokens=["[UNK]", "[EOS]"]))
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="[EOS]", unk_token="[UNK]")

    def setUp(self):
        torch.manual_seed(7)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(self.tokenizer), hidden_size=16,
                                           intermediate_size=32, num_hidden_layers=2,
                                           num_attention_heads=2, num_key_value_heads=2,
                                           max_position_embeddings=512))
        self.engine = algo.Engine(model, self.tokenizer, 0, 512)
        self.item = {"task": "first", "mean": torch.zeros(16), "scale": torch.ones(16),
                     "weight": torch.linspace(-1, 1, 16), "bias": torch.tensor(0.),
                     "vector": torch.linspace(-.1, .1, 16), "strength": .3}

    def test_zero_strength_restores_base_and_weights_stay_frozen(self):
        ids = [4, 5, 6]
        before = {k: v.clone() for k, v in self.engine.model.state_dict().items()}
        baseline = self.engine.forward(ids).logits
        zero = dict(self.item, strength=0.)
        with self.engine.steering([zero], len(ids)):
            actual = self.engine.forward(ids).logits
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        for key, value in before.items():
            torch.testing.assert_close(value, self.engine.model.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in self.engine.model.parameters()))
        self.assertEqual(len(self.engine.layer._forward_hooks), 0)

    def test_cached_decode_matches_teacher_forcing_without_target_leakage(self):
        prefix, completion = [4, 5, 6], [7, 8]
        with self.engine.steering([self.item], len(prefix)):
            full = self.engine.forward(prefix + completion).logits
        model = self.engine.model
        with torch.no_grad(), self.engine.steering([self.item], len(prefix)):
            initial = model(torch.tensor([prefix]), use_cache=True)
            next_output = model(torch.tensor([[completion[0]]]), past_key_values=initial.past_key_values,
                                use_cache=True)
        torch.testing.assert_close(initial.logits[:, -1], full[:, len(prefix)-1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(next_output.logits[:, -1], full[:, len(prefix)], atol=1e-6, rtol=1e-5)

    def test_bank_sum_order_and_hook_cleanup(self):
        other = dict(self.item, strength=.7, weight=-self.item["weight"])
        with self.engine.steering([self.item, other], 1):
            first = self.engine.forward([4]).logits
        with self.engine.steering([other, self.item], 1):
            second = self.engine.forward([4]).logits
        torch.testing.assert_close(first, second)
        with self.assertRaises(RuntimeError):
            with self.engine.steering([self.item], 1):
                raise RuntimeError("intentional")
        self.assertEqual(len(self.engine.layer._forward_hooks), 0)

    def test_direction_and_gate(self):
        feat = {"prompt": torch.randn(8, 16) - 1, "deprecated": torch.randn(8, 16) - 1,
                "replacement": torch.randn(8, 16) + 1, "retain": torch.randn(8, 16) + 1}
        args = argparse.Namespace(lr=.05, weight_decay=0., strength=1., gate_steps=50, cosine_weight=.1)
        item = algo.fit_gate(feat, args)
        torch.testing.assert_close(item["vector"], feat["replacement"].mean(0) - feat["deprecated"].mean(0))
        self.assertGreater(algo.gate_metrics(item, feat)["prompt_true_positive_rate"], .9)
        self.assertLess(algo.gate_metrics(item, feat)["retain_false_positive_rate"], .1)

    def test_oversized_pair_keeps_retain_evaluation(self):
        row = {"id": "oversized", "prompt": "def f(x): return ",
               "replacement": "new(x) " * 1000, "deprecated": "old(x)",
               "retain": "def clean(x): return abs(x) + 1"}
        metrics, details = algo.evaluate_rows(self.engine, [row], [], 0)
        self.assertEqual(metrics["oversized_pairs"], 1)
        self.assertEqual(metrics["pair_samples"], 0)
        self.assertGreater(metrics["retain_nll"], 0)
        self.assertIn("pair_skipped", details[0])

    def test_generated_api_categories_and_aliases(self):
        row = {"deprecated api": ["numpy.product"], "replacement api": "numpy.prod",
               "alias dict": {"np.product": "numpy.product", "np.prod": "numpy.prod"}}
        cases = [("np.product(x)", "deprecated"), ("np.prod(x)", "correct_rep"),
                 ("np.sum(x)", "mismatch"), ("np.prod(x); np.product(x)", "deprecated"),
                 ('# np.product(x)\n"np.prod(x)"', "mismatch"), ("", "mismatch"),
                 ("np.prod(", "correct_rep"), ("np.production(x)", "mismatch")]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(algo.classify_generation(text, row)["outcome"], expected)
        examples = [algo.classify_generation(text, row) for text, _ in cases]
        counts = algo.generation_counts(examples)
        self.assertEqual(counts["evaluated"], 8)
        self.assertEqual(counts["no_dep_count"], 6)
        self.assertEqual(counts["correct_rep_count"], 2)
        self.assertEqual(counts["mismatch_count"], 4)
        self.assertEqual(counts["both_dep_and_rep_count"], 1)
        self.assertIsNone(algo.generation_counts([])["no_dep_rate"])

    def test_hf_test_schema_and_manifest(self):
        raw = {"category": "up-to-dated", "source": "fixture", "function": "x = np.prod(a)",
               "probing input": "x = ", "deprecated api": ["numpy.product"],
               "replacement api": "numpy.prod", "alias dict": {"np.prod": "numpy.prod"}}
        normalized = algo.normalize(raw, training=False)
        self.assertEqual(normalized["library"], "numpy")
        self.assertEqual(normalized["replacement"], "")
        self.assertEqual(normalized["deprecated"], "")
        self.assertEqual(normalized["retain"], raw["function"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            family = root / "codellama"
            for name, rows in (("D_forget.json", [raw]), ("D_test.json", [raw, raw]),
                               ("D_test_U_dep.json", [{**raw, "id": 7, "y_pos": "np.prod(a)"}])):
                algo.write_json(family / name, rows)
            with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download") as download:
                api.return_value.dataset_info.return_value.sha = "pinned-revision"
                download.side_effect = lambda repo, filename, **kwargs: str(root / filename)
                algo.fetch_data(SimpleNamespace(family="codellama", revision="main", output=str(root)))
                self.assertEqual(download.call_count, 3)
                self.assertTrue(all(call.kwargs["revision"] == "pinned-revision" for call in download.call_args_list))
            manifest = algo.data_source(family / "D_forget.json", family / "D_test.json")
            self.assertEqual(manifest["family"], "codellama")
            self.assertEqual(algo.test_subsets(family / "D_test.json"), {0: "U_dep", 1: "U_nondep"})
            with self.assertRaises(ValueError):
                algo.check_model_family("deepseek/model", manifest)
            algo.write_json(family / "D_test_U_dep.json", [raw, raw, raw])
            with self.assertRaises(ValueError):
                algo.data_source(family / "D_forget.json", family / "D_test.json")
            with self.assertRaises(ValueError):
                algo.test_subsets(family / "D_test.json")

    def test_prepare_cli_resume_evaluate_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for library in ("alpha", "beta"):
                for i in range(5):
                    rows.append({"id": f"{library}{i}", "library": library,
                                 "category": "up-to-dated" if i == 4 else "outdated",
                                 "probing input": f"def {library}{i}(x):\n    return ",
                                 "function": f"def {library}{i}(x): return old(x)",
                                 "y_pos": "new(x)", "y_neg": "old(x)",
                                 "retain": f"def clean_{library}{i}(z): return abs(z) + {i}",
                                 "deprecated api": ["old"], "replacement api": "new", "alias dict": {}})
            test = copy.deepcopy(rows[:1])
            test[0].update(category="up-to-dated", y_pos="")
            algo.write_json(root / "forget.json", rows)
            algo.write_json(root / "test.json", test)
            def run(*arguments):
                with patch.object(sys, "argv", ["algo.py", *map(str, arguments)]):
                    algo.main()
            prepared = root / "prepared.json"
            run("prepare", "--forget", root / "forget.json", "--test", root / "test.json", "--output", prepared)
            data = algo.read_json(prepared)
            self.assertEqual(data["excluded"]["test_context_overlap"], 1)
            self.assertEqual(sum(r["category"] == "up-to-dated" for task in data["tasks"]
                                 for split in ("train", "validation") for r in task[split]), 2)
            for task in data["tasks"]:
                self.assertFalse({r["context_hash"] for r in task["train"]} &
                                 {r["context_hash"] for r in task["validation"]})
                self.assertFalse({r["retain"] for r in task["train"]} &
                                 {r["retain"] for r in task["validation"]})
            model_path = root / "model"
            self.engine.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            checkpoint_dir = root / "checkpoints"
            common = ["--model", model_path, "--device", "cpu", "--dtype", "float32", "--data", prepared]
            training = ["train", *common, "--output", checkpoint_dir, "--max-length", "256",
                        "--max-samples", "2", "--gate-steps", "5", "--validation-samples", "1"]
            run(*training, "--stop-after", "1")
            first = algo.load_checkpoint(checkpoint_dir / "step_001.pt")
            run(*training, "--resume", checkpoint_dir / "step_001.pt")
            second = algo.load_checkpoint(checkpoint_dir / "step_002.pt")
            self.assertEqual(len(second["bank"]), 2)
            torch.testing.assert_close(first["bank"][0]["weight"], second["bank"][0]["weight"], rtol=0, atol=0)
            output = root / "report.json"
            run("evaluate", *common, "--checkpoints", checkpoint_dir, "--max-samples", "1",
                "--max-new-tokens", "2", "--output", output)
            self.assertEqual(len(algo.read_json(output)["stages"]), 3)
            run("evaluate", *common, "--checkpoints", checkpoint_dir, "--split", "test", "--output", output)
            metrics = algo.read_json(output)["stages"][-1]["groups"][0]["metrics"]
            self.assertEqual(metrics["pair_samples"], 0)
            self.assertIsNone(metrics["replacement_mean_logp"])
            # Raw evaluation must include the overlapping row excluded by prepare,
            # and the test row without a replacement completion.
            run("evaluate-api", "--model", model_path, "--device", "cpu", "--dtype", "float32",
                "--checkpoint", checkpoint_dir / "step_002.pt", "--forget", root / "forget.json",
                "--test", root / "test.json", "--max-new-tokens", "2", "--output", output)
            report = algo.read_json(output)
            self.assertEqual(report["stages"][0]["datasets"]["D_forget"]["summary"]["evaluated"], 10)
            self.assertEqual(report["stages"][0]["datasets"]["D_test"]["summary"]["evaluated"], 1)
            algo.write_json(root / "D_test_U_dep.json", test)
            run("evaluate-api", "--model", model_path, "--device", "cpu", "--dtype", "float32",
                "--checkpoint", checkpoint_dir / "step_002.pt", "--forget", root / "forget.json",
                "--test", root / "test.json", "--max-new-tokens", "2", "--output", output)
            subsets = algo.read_json(output)["stages"][0]["datasets"]["D_test"]["subsets"]
            self.assertEqual(subsets["U_dep"]["evaluated"], 1)
            self.assertEqual(subsets["U_nondep"]["evaluated"], 0)
            test.append({**test[0], "probing input": ""})
            algo.write_json(root / "test.json", test)
            valid, audit = algo.generation_dataset(root / "test.json")
            self.assertEqual(len(valid), 1)
            self.assertEqual(audit["total_rows"], 2)
            self.assertEqual(audit["skipped_rows"], 1)
            run("generate", "--model", model_path, "--device", "cpu", "--dtype", "float32",
                "--checkpoint", checkpoint_dir / "step_002.pt", "--prompt", "def f(x): return ",
                "--max-new-tokens", "2")


if __name__ == "__main__":
    unittest.main()
