# Continual API Unlearning with MLLMEraser Steering

Adaptation of MLLMEraser for the code/API task in `Continual Unlearning Task MML.pdf`.
The base causal language model is frozen. Each task adds a contrastive direction
and a small linear sigmoid gate. This is an experimental implementation, not a
claim that unlearning or knowledge preservation has already been demonstrated.

## Layout

```text
project/
  data/               # Original JSON data and generated prepared.json
  requirements.txt
  .gitignore
  .gitattributes
  algo.py             # Preparation, extraction, gates, inference, evaluation
  run_script.sh
  README.md
  tests/test_algo.py  # Offline correctness and integration checks
```

`checkpoints/` and `results/` are created at runtime and excluded from Git.

## Install

Python 3.10-3.12 is recommended. Create a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1` instead of `source`.
Install the appropriate PyTorch CUDA wheel for your machine if needed.
The default model is `codellama/CodeLlama-7b-hf`; pass `--model` to use your own
M0 checkpoint. Loading 7B weights in FP16 needs roughly 14 GB just for weights,
plus activations and generation cache. `--device auto` allows Accelerate's
placement; CPU-only runs should use `--device cpu --dtype float32`.

## Data and Splits

Input files are `data/D_forget.json` and `data/D_test.json` from the supplied
[Google Drive folder](https://drive.google.com/drive/folders/1IQ9hSeuDFWcQaTuoHYrmN8f2IOYBc3gr).

| Dataset field | Meaning here |
| --- | --- |
| `probing input` | Code context x, before the full completion line |
| `y_pos` | Replacement completion, desired |
| `y_neg` | Unwanted alternative; deprecated only for appropriate outdated rows |
| `retain` | Candidate clean code for the gate and utility evaluation |
| `library` | Default continual task grouping |
| `deprecated api`, `replacement api`, `alias dict` | Static API-call evaluation |

The PDF reverses `y+`/`y-` names in one paragraph. The implementation consistently
uses the dataset's `y_pos = replacement`, `y_neg = unwanted` convention. It does
not use `probing input new`, which can already end inside the API expression and
therefore does not align with these full-line completion targets. Whitespace in
completion strings is preserved. Prompt and completion tokens are encoded
separately, so the conditional token boundary is explicit and reproducible.

```bash
python algo.py prepare
```

Preparation validates fields and records rejected IDs/reasons in `invalid`.
Only complete `outdated` pairs train the erasure direction. It removes duplicate
pairs and test-overlapping contexts/functions before splitting. Shared prompts
and functions stay in the same split, using whitespace-normalized hashes.
It constructs a disjoint retain pool from supplied retain snippets, excluding
test code and outdated source functions, with separate train/validation retain partitions.
Retain candidates containing a task's deprecated API calls are skipped. This is
an exact normalized-text overlap check, not semantic clone detection.

The supplied D_test contains only `up-to-dated` records. It is a retention/utility
test, not a held-out forgetting benchmark. Rows with missing completions still
contribute retain and optional generation metrics; `pair_samples` gives the
number actually used for pair scoring. A separate unseen outdated test set is
needed for a final forgetting claim. Outdated validation from D_forget is useful
for development, but should not be presented as an untouched final test after
tuning on it.

Default task order is alphabetical by library, not an inferred real timeline.
Choose an explicit order before preparation:

```bash
python algo.py prepare --task-order numpy,pandas,pytorch,scipy,seaborn,sklearn,tensorflow,transformers
```

For a finer sequence, `--task-by api` groups by library plus deprecated API list.
Prepared data records task names/order, exclusions and retain-pool sizes.

For the downloaded Drive files, the default split contains 7,604 training pairs,
840 validation pairs and 17,173 valid test rows across eight libraries. The raw
files have 10,248 and 17,179 rows respectively. Eight records have blank contexts
(two forget, six test); 311 outdated pairs have a blank target, 253 overlap test
contexts/functions and 77 are duplicate pairs. The 1,161 up-to-dated forget-file
rows are not positive forgetting examples. Train/validation retain pools contain
1,047/116 snippets. The test has 542 valid rows without a complete target pair.

## Train and Continue

```bash
python algo.py train --model codellama/CodeLlama-7b-hf
```

Defaults: one middle decoder block, 256 training examples/task, 32 validation
examples/task, strength 1, 300 gate optimizer steps, cosine weight 0.1.
`--max-samples 0` uses all training examples; `--validation-samples 0` uses all
validation examples. Sampling is deterministic. Layer indices are zero-based
decoder block outputs, before the model's final normalization. `--layer 16`
selects a specific block. All tasks share this layer.

```bash
python algo.py train --stop-after 1
python algo.py train --resume checkpoints/step_001.pt
```

`step_000.pt` is the unsteered baseline; later files contain the cumulative bank.
Resume verifies model name, data fingerprint, layer, context length, task order
and seed, and preserves older vectors/gates. Prepare the whole planned task
sequence first; changing the prepared dataset requires a new experiment.
Each checkpoint records gate hyperparameters and validation gate statistics.
Inputs are left-truncated, preserving complete target alternatives. Pairs whose
targets alone exceed `--max-length` are skipped with IDs/counts recorded in the
checkpoint or evaluation report; increase the context limit to include them.

For each task k:

```text
v_k = mean(h(x, replacement)) - mean(h(x, deprecated))
g_k(h) = sigmoid(w_k @ ((h - mean_k) / scale_k) + b_k)
h' = h + sum_k strength_k * g_k(h_prompt) * v_k
loss_gate = BCE(g_k, labels) + cosine_weight * (1 - cos(h_steered, mean_replacement))
```

Positive gate examples are outdated prompt states and deprecated completion
states. Negative examples are replacement completion states and clean code
prefix states. The cosine term acts only on positive examples. Direction and
features always come from M0; only the linear gate is optimized, on CPU.

At inference the gate sees the final prompt token before any target tokens.
All gates see the same unmodified activation. Their deltas are summed and
applied at the prompt boundary and subsequent generated tokens, using the KV
cache. Each request recomputes its own gates. Prefixes before the boundary are
unchanged. Teacher-forced scoring uses this same rule, so gates cannot inspect
the reference answer. A one-token prompt is supported.

## Evaluate

Tune layer, strength and cosine weight on validation, in separate output folders:

```bash
python algo.py evaluate --split validation --max-new-tokens 64 --output results/validation.json
python algo.py evaluate --split test --max-new-tokens 64 --output results/test.json
```

Evaluation compares every available checkpoint, including M0, on the same
deterministic sample for every task/category, including future tasks. Defaults
use 32 examples/group; `--max-samples 0` evaluates all. `--max-new-tokens 0`
skips generation and computes teacher-forced metrics only.

Metrics include mean completion log probabilities, replacement preference rate,
token-weighted retain NLL/perplexity, differences from M0, and optional generated
deprecated/replacement call rates. Completion preference is length-normalized;
raw sequence log probabilities and token counts are also stored per example.
For up-to-dated rows, `alternative_mean_logp` is NOT a deprecated-API metric.
Static call rates resolve aliases present in each record but do not execute code,
check correctness, or perform full Python name resolution. Calls in comments or
strings and unresolved aliases can affect this heuristic.

Reports contain the full per-stage/per-task matrix and per-example results.
Compare an old task at its learning stage versus the final stage to measure loss
of prior forgetting; inspect future tasks and retain NLL to measure interference.
Frozen weights guarantee reversible intervention, not absence of behavioral
forgetting. Additive gates can overlap and accumulate; no hard API-ban guarantee
or proof of no catastrophic forgetting is claimed.

Generate with a cumulative checkpoint:

```bash
python algo.py generate --checkpoint checkpoints/step_008.pt --prompt-file data/prompt.txt
```

On Bash/WSL, `bash run_script.sh` runs prepare, train and validation. Any arguments
are forwarded directly, e.g. `bash run_script.sh train --stop-after 1`.
Set `MODEL` to change the no-argument pipeline's model, and `PYTHON` to select
the interpreter. On PowerShell run the equivalent `python algo.py ...` commands.

## Changes Relative to the Original

Source: supplied `MLLMEraser-B41D.zip`, especially `Extractor.py`,
`qwen_steering.py` and `null_space_util.py`.
[Paper](https://arxiv.org/abs/2510.04217) and
[author-linked anonymous repository](https://anonymous.4open.science/r/MLLMEraser-B41D/README.md).

| Original mechanism | Adaptation for this task |
| --- | --- |
| Multimodal contrastive activations | Code-context plus deprecated/replacement pairs |
| Mean contrastive erasure direction | Same difference-of-means construction |
| Input-aware linear steering matrix/null-space fit | Sigmoid linear gate and cosine term specified in the task PDF |
| Custom Qwen-VL decoder subclass | Output hook on a standard causal-LM decoder block |
| Prefill steering broadcast | Prompt-boundary steering continued during decoding, with a fixed prompt-only gate |
| One forgetting setup | Cumulative immutable vector/gate bank and stagewise evaluation |

These are deliberate method changes, not a reproduction of the paper's reported
numbers. Vision processing, adversarial images, LoRA and training baselines are
omitted because the provided task is text-only and freezes M0. The supplied ZIP
imports missing `lib`/`data_process` modules and hard-codes author-local paths;
the self-contained implementation replaces those dependencies. The PDF's later
handwritten suggestion of an additional output-probability loss is not added:
the implemented training objective is its explicit BCE plus cosine gate design,
and probability changes are measured in evaluation.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests initialize a tiny random Llama locally, without downloading model weights.
They cover frozen weights, zero-strength equivalence, cached versus teacher-forced
consistency, additive task ordering, hook cleanup, gate learning, data separation,
checkpoint resume, evaluation and generation. Passing these checks verifies
pipeline mechanics only; run your M0 experiment to establish effectiveness.

An additional local smoke run used the actual prepared Drive data with a random
tiny Llama: two training examples and one validation example per task, five gate
steps, all eight continual stages, and all 72 baseline/stage-by-task validation
evaluations. Its artifacts are under `results/offline_smoke/`; they are only
debugging artifacts and must not be used as research results or deployed gates.
