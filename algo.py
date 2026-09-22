"""Continual API steering adapted from MLLMEraser; base-model weights stay frozen."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import torch
import torch.nn.functional as F


class ContextWindowError(ValueError):
    pass


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hash(text):
    return fingerprint(" ".join(text.split()))


def normalize(row, task_by="library"):
    # The supplied data uses y_pos=replacement and y_neg=unwanted completion.
    keys = {"prompt": "probing input", "replacement": "y_pos", "deprecated": "y_neg",
            "retain": "retain"}
    result = {}
    for key, source in keys.items():
        value = row.get(key, row.get(source))
        if key in ("replacement", "deprecated") and isinstance(value, str) and not value.strip():
            result[key] = ""
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Record {row.get('id')}: missing/non-string {source}")
        result[key] = value
    for key in ("id", "library", "category", "deprecated api", "replacement api", "alias dict"):
        result[key] = row.get(key)
    if not isinstance(result["library"], str) or not result["library"]:
        raise ValueError("Each record needs a library")
    old = result["deprecated api"]
    old = [old] if isinstance(old, str) else old
    if not isinstance(old, list) or not all(isinstance(x, str) for x in old):
        raise ValueError("deprecated api must be a string or list of strings")
    result["deprecated api"] = old
    result["task"] = result["library"] if task_by == "library" else result["library"] + ":" + "|".join(sorted(old))
    result["context_hash"] = code_hash(result["prompt"])
    result["group_hash"] = code_hash(row.get("function") or result["prompt"])
    return result


def prepare(args):
    invalid = []

    def load(path, split):
        rows = []
        for row in read_json(path):
            try:
                rows.append(normalize(row, args.task_by))
            except ValueError as error:
                invalid.append({"split": split, "id": row.get("id"), "reason": str(error)})
        if not rows:
            raise ValueError(f"No valid records in {path}")
        return rows

    train, test = load(args.forget, "forget"), load(args.test, "test")
    test_hashes = {r["context_hash"] for r in test}
    test_code = {r["group_hash"] for r in test} | {code_hash(r["retain"]) for r in test}
    source_code = {r["group_hash"] for r in train if r["category"] == "outdated"}
    retain_pool = sorted({r["retain"] for r in train
                          if code_hash(r["retain"]) not in test_code | source_code})
    random.Random(args.seed).shuffle(retain_pool)
    if len(retain_pool) < 2:
        raise ValueError("No disjoint retain pool; supply independent clean retain examples")
    boundary = max(1, min(len(retain_pool) - 1, round(len(retain_pool) * args.val_fraction)))
    val_retain, train_retain = retain_pool[:boundary], retain_pool[boundary:]
    grouped = defaultdict(list)
    seen = set()
    excluded = Counter()
    for row in train:
        if row["category"] != "outdated":
            excluded["not_outdated"] += 1
            continue
        if not row["replacement"] or not row["deprecated"]:
            excluded["missing_pair"] += 1
            continue
        if row["context_hash"] in test_hashes or row["group_hash"] in test_code:
            excluded["test_context_overlap"] += 1
            continue
        key = (row["task"], row["context_hash"], row["replacement"], row["deprecated"])
        if key in seen:
            excluded["duplicate_pair"] += 1
            continue
        seen.add(key)
        grouped[row["task"]].append(row)
    order = args.task_order.split(",") if args.task_order else sorted(grouped)
    if set(order) != set(grouped) or len(order) != len(set(order)):
        raise ValueError("--task-order must contain each task exactly once")
    tasks = []
    for i, name in enumerate(order):
        rows = grouped[name]
        # Keep shared functions AND shared prompts in one connected split group.
        parents = {}

        def find(key):
            parents.setdefault(key, key)
            if parents[key] != key:
                parents[key] = find(parents[key])
            return parents[key]

        for row in rows:
            parents[find("p:" + row["context_hash"])] = find("f:" + row["group_hash"])
        for row in rows:
            row["split_hash"] = find("f:" + row["group_hash"])
        contexts = sorted({r["split_hash"] for r in rows})
        if len(contexts) < 2:
            raise ValueError(f"{name}: at least two distinct training contexts required")
        random.Random(args.seed + i).shuffle(contexts)
        n_val = max(1, min(len(contexts) - 1, round(len(contexts) * args.val_fraction)))
        val_hashes = set(contexts[:n_val])
        task = {"name": name, "train": [], "validation": []}
        for row in rows:
            split = "validation" if row["split_hash"] in val_hashes else "train"
            pool = val_retain if split == "validation" else train_retain
            offset = int(row["context_hash"][:8], 16) % len(pool)
            for j in range(len(pool)):
                candidate = pool[(offset + j) % len(pool)]
                if not api_hits(candidate, row)[0]:
                    row["retain"] = candidate
                    break
            else:
                raise ValueError(f"No clean retain example for {name}")
            task[split].append(row)
        tasks.append(task)
    payload = {"version": 1, "seed": args.seed, "task_by": args.task_by,
               "tasks": tasks, "test": test, "excluded": dict(excluded), "invalid": invalid,
               "retain_pool": {"train": len(train_retain), "validation": len(val_retain)},
               "test_missing_replacement": sum(not r["replacement"] for r in test)}
    write_json(args.output, payload)
    print(json.dumps({"tasks": [{"name": t["name"], "train": len(t["train"]),
                                  "validation": len(t["validation"])} for t in tasks],
                      "test": len(test), "excluded": dict(excluded), "invalid": invalid,
                      "retain_pool": payload["retain_pool"]}, indent=2))


def get_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        value = model
        for part in path.split("."):
            value = getattr(value, part, None)
        if value is not None:
            return value
    raise ValueError("Unsupported decoder layout; use CodeLlama/Llama/Qwen2 or GPT-2")


class Engine:
    def __init__(self, model, tokenizer, layer, max_length):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.layer_index = layer
        self.layer = get_layers(model)[layer]
        self.max_length = max_length
        self.device = model.get_input_embeddings().weight.device

    def ids(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("Text tokenizes to an empty sequence")
        return ids

    def prompt_ids(self, text, reserve=0):
        room = self.max_length - reserve
        if room < 1:
            raise ContextWindowError("Completion consumes the context window; increase --max-length")
        return self.ids(text)[-room:]

    def pair_ids(self, row):
        good, bad = self.ids(row["replacement"]), self.ids(row["deprecated"])
        # Both alternatives see exactly the same left-truncated prompt.
        prompt = self.prompt_ids(row["prompt"], max(len(good), len(bad)))
        return prompt, good, bad

    def forward(self, ids):
        x = torch.tensor([ids], device=self.device)
        return self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)

    @torch.no_grad()
    def hidden(self, ids):
        captured = []

        def capture(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[0, -1].detach().float().cpu())

        handle = self.layer.register_forward_hook(capture)
        try:
            self.forward(ids)
        finally:
            handle.remove()
        return captured[0]

    @contextmanager
    def steering(self, bank, prompt_length):
        """Compute gates before seeing targets; share that gate over cached decoding."""
        if not bank:
            yield
            return
        delta = None
        first = True

        def hook(module, inputs, output):
            nonlocal delta, first
            h = output[0] if isinstance(output, tuple) else output
            if first:
                if h.shape[1] < prompt_length:
                    raise ValueError("Initial forward does not contain the full prompt")
                base = h[:, prompt_length - 1, :].float()
                delta = torch.zeros_like(base)
                for item in bank:
                    state = {k: item[k].to(h.device) for k in ("mean", "scale", "weight", "bias", "vector")}
                    gate = torch.sigmoid(((base - state["mean"]) / state["scale"]) @ state["weight"] + state["bias"])
                    delta = delta + item["strength"] * gate.unsqueeze(-1) * state["vector"]
                start = prompt_length - 1
                first = False
            else:
                start = 0
            changed = h.clone()
            changed[:, start:, :] = changed[:, start:, :] + delta.unsqueeze(1).to(h.dtype)
            return (changed,) + output[1:] if isinstance(output, tuple) else changed

        handle = self.layer.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def score(self, prompt, completion, bank):
        with self.steering(bank, len(prompt)):
            logits = self.forward(prompt + completion).logits[0, len(prompt) - 1:-1].float()
        target = torch.tensor(completion, device=logits.device)
        values = F.log_softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
        return {"mean": values.mean().item(), "sum": values.sum().item(), "tokens": len(completion)}

    @torch.no_grad()
    def generate(self, prompt, bank, max_new_tokens):
        ids = self.prompt_ids(prompt, max_new_tokens)
        x = torch.tensor([ids], device=self.device)
        with self.steering(bank, len(ids)):
            output = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                                         max_new_tokens=max_new_tokens, do_sample=False,
                                         use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)


def load_engine(args, metadata=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if metadata and args.model != metadata["model"]:
        raise ValueError("Use the same --model as the steering checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                               device_map=args.device, attn_implementation="eager")
    layer = metadata["layer"] if metadata else args.layer
    if layer == -1:
        layer = len(get_layers(model)) // 2
    if not 0 <= layer < len(get_layers(model)):
        raise ValueError("--layer is a zero-based decoder block index")
    max_length = metadata["max_length"] if metadata else args.max_length
    if max_length > getattr(model.config, "max_position_embeddings", max_length):
        raise ValueError("--max-length exceeds this model's context window")
    return Engine(model, tokenizer, layer, max_length)


def sample(rows, limit, seed):
    if not limit or len(rows) <= limit:
        return list(rows)
    return random.Random(seed).sample(rows, limit)


def features(engine, rows):
    result = {name: [] for name in ("prompt", "replacement", "deprecated", "retain")}
    skipped = []
    for i, row in enumerate(rows):
        try:
            prompt, good, bad = engine.pair_ids(row)
        except ContextWindowError:
            skipped.append(row["id"])
            continue
        result["prompt"].append(engine.hidden(prompt))
        result["replacement"].append(engine.hidden(prompt + good))
        result["deprecated"].append(engine.hidden(prompt + bad))
        clean = engine.prompt_ids(row["retain"])
        result["retain"].append(engine.hidden(clean[:max(1, len(clean) // 2)]))
        if (i + 1) % 25 == 0:
            print(f"  extracted {i + 1}/{len(rows)}", flush=True)
    if not result["prompt"]:
        raise ValueError("No pairs fit the context window; increase --max-length")
    tensors = {key: torch.stack(value) for key, value in result.items()}
    tensors["skipped_context_ids"] = skipped
    if skipped:
        print(f"  skipped {len(skipped)} oversized completions; IDs recorded in checkpoint", flush=True)
    return tensors


def fit_gate(feat, args):
    rep_mean = feat["replacement"].mean(0)
    vector = rep_mean - feat["deprecated"].mean(0)
    if not torch.isfinite(vector).all() or vector.norm() < 1e-8:
        raise ValueError("Nonfinite or zero steering direction")
    positives = torch.cat([feat["prompt"], feat["deprecated"]])
    negatives = torch.cat([feat["replacement"], feat["retain"]])
    x = torch.cat([positives, negatives])
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))])
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)
    z = (x - mean) / scale
    gate = torch.nn.Linear(x.shape[-1], 1)
    torch.nn.init.zeros_(gate.weight)
    torch.nn.init.zeros_(gate.bias)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.gate_steps):
        logits = gate(z).squeeze(-1)
        probabilities = logits.sigmoid()
        moved = positives + args.strength * probabilities[:len(positives), None] * vector
        cosine = (1 - F.cosine_similarity(moved, rep_mean.expand_as(moved))).mean()
        loss = F.binary_cross_entropy_with_logits(logits, labels) + args.cosine_weight * cosine
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {"vector": vector, "replacement_mean": rep_mean, "mean": mean, "scale": scale,
            "weight": gate.weight.detach().squeeze(0), "bias": gate.bias.detach().squeeze(0),
            "strength": args.strength, "gate_loss": loss.item()}


def gate_metrics(item, feat):
    def probability(x):
        return (((x - item["mean"]) / item["scale"]) @ item["weight"] + item["bias"]).sigmoid()
    return {"prompt_true_positive_rate": (probability(feat["prompt"]) >= .5).float().mean().item(),
            "replacement_false_positive_rate": (probability(feat["replacement"]) >= .5).float().mean().item(),
            "retain_false_positive_rate": (probability(feat["retain"]) >= .5).float().mean().item()}


def save_checkpoint(path, metadata, bank):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"version": 1, "metadata": metadata, "bank": bank}, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1:
        raise ValueError("Unsupported checkpoint version")
    return checkpoint


def train(args):
    torch.manual_seed(args.seed)
    data = read_json(args.data)
    dataset_hash = fingerprint(json.dumps(data, sort_keys=True))
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    engine = load_engine(args, checkpoint["metadata"] if checkpoint else None)
    metadata = {"model": args.model, "layer": engine.layer_index, "max_length": engine.max_length,
                "dataset_hash": dataset_hash, "task_order": [t["name"] for t in data["tasks"]],
                "seed": args.seed, "method": "contrastive-mean+sigmoid-gate+cosine"}
    bank = checkpoint["bank"] if checkpoint else []
    if checkpoint and checkpoint["metadata"] != metadata:
        raise ValueError("Resume requires identical model, prepared data, task order and seed")
    if [b["task"] for b in bank] != metadata["task_order"][:len(bank)]:
        raise ValueError("Checkpoint tasks are not a prefix of the requested task order")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not bank:
        if (output / "step_000.pt").exists():
            raise ValueError("Output already contains a run; use --resume or another --output")
        save_checkpoint(output / "step_000.pt", metadata, [])
    for index, task in enumerate(data["tasks"]):
        if index < len(bank):
            continue
        if args.stop_after and index >= args.stop_after:
            break
        print(f"Step {index + 1}: {task['name']}", flush=True)
        rows = sample(task["train"], args.max_samples, args.seed + index)
        # Extraction always uses M0, without any previous steering hooks installed.
        feat = features(engine, rows)
        item = fit_gate(feat, args)
        item["task"] = task["name"]
        item["train_samples"] = len(feat["prompt"])
        item["train_skipped_context_ids"] = feat["skipped_context_ids"]
        val = sample(task["validation"], args.validation_samples, args.seed + index)
        validation_features = features(engine, val)
        item["validation"] = gate_metrics(item, validation_features)
        item["validation_samples"] = len(validation_features["prompt"])
        item["validation_skipped_context_ids"] = validation_features["skipped_context_ids"]
        item["training"] = {key: getattr(args, key) for key in
                            ("gate_steps", "lr", "weight_decay", "cosine_weight", "strength")}
        bank.append(item)
        path = output / f"step_{len(bank):03d}.pt"
        if path.exists():
            raise ValueError(f"Refusing to overwrite {path}; use another output directory")
        save_checkpoint(path, metadata, bank)
        print(json.dumps({"checkpoint": str(path), "gate_validation": item["validation"]}), flush=True)


def api_hits(text, row):
    # Static identifier matching only; generated code is never executed.
    calls = set(re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?=\s*\()", text))
    aliases = row.get("alias dict") or {}
    canonical = {aliases.get(call, call) for call in calls}
    old = bool(canonical.intersection(row["deprecated api"]))
    new = row.get("replacement api") in canonical
    return old, new


def evaluate_rows(engine, rows, bank, max_new_tokens):
    totals = defaultdict(float)
    details = []
    for row in rows:
        record = {"id": row["id"]}
        pair = None
        if row["replacement"] and row["deprecated"]:
            try:
                pair = engine.pair_ids(row)
            except ContextWindowError as error:
                record["pair_skipped"] = str(error)
                totals["oversized_pairs"] += 1
        if pair is not None:
            prompt, good, bad = pair
            positive, negative = engine.score(prompt, good, bank), engine.score(prompt, bad, bank)
            totals["pair_samples"] += 1
            totals["replacement_mean_logp"] += positive["mean"]
            totals["alternative_mean_logp"] += negative["mean"]
            totals["replacement_preference_rate"] += positive["mean"] > negative["mean"]
            record.update(replacement=positive, alternative=negative)
        retain_ids = engine.prompt_ids(row["retain"])
        split = max(1, len(retain_ids) // 2)
        if split == len(retain_ids):
            raise ValueError("Retain text must contain at least two tokens")
        retain = engine.score(retain_ids[:split], retain_ids[split:], bank)
        totals["retain_logp_sum"] += retain["sum"]
        totals["retain_tokens"] += retain["tokens"]
        record["retain"] = retain
        if max_new_tokens:
            generated = engine.generate(row["prompt"], bank, max_new_tokens)
            old, new = api_hits(generated, row)
            totals["deprecated_call_rate"] += old
            totals["replacement_call_rate"] += new
            record.update(generated=generated, deprecated_call=old, replacement_call=new)
        details.append(record)
    metrics = {}
    for key in ("replacement_mean_logp", "alternative_mean_logp", "replacement_preference_rate"):
        metrics[key] = totals[key] / totals["pair_samples"] if totals["pair_samples"] else None
    if max_new_tokens:
        for key in ("deprecated_call_rate", "replacement_call_rate"):
            metrics[key] = totals[key] / len(rows)
    metrics["pair_samples"] = int(totals["pair_samples"])
    metrics["oversized_pairs"] = int(totals["oversized_pairs"])
    metrics["retain_nll"] = -totals["retain_logp_sum"] / totals["retain_tokens"]
    metrics["retain_perplexity"] = math.exp(min(metrics["retain_nll"], 80))
    metrics["samples"] = len(rows)
    return metrics, details


def evaluate(args):
    data = read_json(args.data)
    paths = sorted(Path(args.checkpoints).glob("step_*.pt"))
    if not paths:
        raise ValueError("No step_*.pt checkpoints found")
    initial = load_checkpoint(paths[0])
    metadata = initial["metadata"]
    if metadata["dataset_hash"] != fingerprint(json.dumps(data, sort_keys=True)):
        raise ValueError("Evaluation must use the prepared data recorded during training")
    engine = load_engine(args, metadata)
    groups = defaultdict(list)
    if args.split == "test":
        source = data["test"]
    else:
        source = [r for t in data["tasks"] for r in t["validation"]]
    for row in source:
        groups[(row["task"], row["category"] or "unknown")].append(row)
    groups = {key: sample(value, args.max_samples, args.seed) for key, value in sorted(groups.items())}
    report = {"split": args.split, "metadata": metadata, "stages": []}
    baseline = {}
    for path in paths:
        state = load_checkpoint(path)
        if state["metadata"] != metadata:
            raise ValueError("Mixed experiments in checkpoint directory")
        bank = state["bank"]
        stage = {"checkpoint": path.name, "learned_tasks": [b["task"] for b in bank], "groups": []}
        for (task, category), rows in groups.items():
            print(f"{path.name}: {task}/{category}, n={len(rows)}", flush=True)
            metrics, details = evaluate_rows(engine, rows, bank, args.max_new_tokens)
            key = (task, category)
            if not bank:
                baseline[key] = metrics
            if key in baseline:
                metrics["delta_vs_base"] = {k: metrics[k] - baseline[key][k] for k in
                                            ("replacement_mean_logp", "alternative_mean_logp", "retain_nll")
                                            if metrics[k] is not None and baseline[key][k] is not None}
            stage["groups"].append({"task": task, "category": category,
                                    "seen": task in stage["learned_tasks"], "metrics": metrics,
                                    "examples": details})
        report["stages"].append(stage)
        write_json(args.output, report)


def generate(args):
    state = load_checkpoint(args.checkpoint)
    engine = load_engine(args, state["metadata"])
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    print(engine.generate(prompt, state["bank"], args.max_new_tokens))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--forget", default="data/D_forget.json")
    prep.add_argument("--test", default="data/D_test.json")
    prep.add_argument("--output", default="data/prepared.json")
    prep.add_argument("--task-by", choices=["library", "api"], default="library")
    prep.add_argument("--task-order", help="Comma-separated tasks; default: alphabetical")
    prep.add_argument("--val-fraction", type=float, default=.1)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=prepare)
    for name, function in (("train", train), ("evaluate", evaluate), ("generate", generate)):
        command = commands.add_parser(name)
        command.add_argument("--model", default="codellama/CodeLlama-7b-hf")
        command.add_argument("--device", default="auto")
        command.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
        command.add_argument("--layer", type=int, default=-1)
        command.add_argument("--max-length", type=int, default=1024)
        command.set_defaults(func=function)
        if name == "generate":
            command.add_argument("--checkpoint", required=True)
            prompt = command.add_mutually_exclusive_group(required=True)
            prompt.add_argument("--prompt")
            prompt.add_argument("--prompt-file")
            command.add_argument("--max-new-tokens", type=int, default=128)
            continue
        command.add_argument("--data", default="data/prepared.json")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--max-samples", type=int, default=256 if name == "train" else 32,
                             help="Samples per task (evaluation: per task/category); 0 means all")
        if name == "train":
            command.add_argument("--output", default="checkpoints")
            command.add_argument("--resume")
            command.add_argument("--stop-after", type=int, default=0,
                                 help="Stop at this total task count; 0 completes the schedule")
            command.add_argument("--validation-samples", type=int, default=32)
            command.add_argument("--gate-steps", type=int, default=300)
            command.add_argument("--lr", type=float, default=.01)
            command.add_argument("--weight-decay", type=float, default=.01)
            command.add_argument("--cosine-weight", type=float, default=.1)
            command.add_argument("--strength", type=float, default=1.)
        else:
            command.add_argument("--checkpoints", default="checkpoints")
            command.add_argument("--split", choices=["validation", "test"], default="validation")
            command.add_argument("--output", default="results/validation.json")
            command.add_argument("--max-new-tokens", type=int, default=0,
                                 help="0 skips generation; positive values also measure API calls")
    return root


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args()
    for key in ("max_samples", "validation_samples", "max_new_tokens", "cosine_weight", "strength", "stop_after"):
        if hasattr(args, key) and getattr(args, key) < 0:
            raise ValueError(f"--{key.replace('_', '-')} must be nonnegative")
    if hasattr(args, "gate_steps") and args.gate_steps < 1:
        raise ValueError("--gate-steps must be positive")
    if hasattr(args, "val_fraction") and not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between zero and one")
    args.func(args)


if __name__ == "__main__":
    main()
