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

The canonical source is
[tummitum/Data-Collection](https://huggingface.co/datasets/tummitum/Data-Collection).
The pipeline pins revision `07a1ca0083ab8b0a71c18a43195330cf495f475a` by default.
Download the split for the base model before preparing data:

```bash
python algo.py fetch-data --family codellama
python algo.py prepare
```

Files live in `data/codellama/`: `D_forget.json`, `D_test.json`,
`D_test_U_dep.json`, and a generated `source.json` recording revision, row counts
and SHA-256 checksums. `prepared.json` is generated in the same directory.
Downloads use `huggingface_hub`, already in requirements.txt. No manual SCP or
Drive download is needed on the server. Avoid loading the entire HF repository
as one dataset: files have different schemas and belong to different models.

The benchmark defines D_forget as outdated examples plus selected up-to-dated
examples where the corresponding model still emits deprecated APIs. Both kinds
are forgetting examples. D_test has two subsets: `U_dep` (known deprecated-API
errors by that model) and `U_nondep` (the remaining updated examples). U_nondep
is computed as the multiset complement of the supplied U_dep file; there is no
need to download its redundant large JSON file.

CodeLlama raw counts: D_forget = 10,396; D_test = 17,031; U_dep = 1,310;
U_nondep = 15,721. These supersede the earlier Drive data and its counts.

Available families are `codellama`, `codegen`, `deepseek`, and `starcoder`.
For another family, use its own paths and matching M0 checkpoint:

```bash
python algo.py fetch-data --family deepseek
python algo.py prepare --forget data/deepseek/D_forget.json --test data/deepseek/D_test.json --output data/deepseek/prepared.json
python algo.py train --model /path/to/deepseek-model --data data/deepseek/prepared.json --output checkpoints/deepseek_hf
```

Recognizable model-family mismatches are rejected. For a local model path without
a recognizable family name, selecting the correct checkpoint remains the user's
responsibility. Use `fetch-data --revision COMMIT` to explicitly change revision.

| Dataset field | Meaning here |
| --- | --- |
| `probing input` | Code context x, before the full completion line |
| `y_pos` | Replacement completion, desired |
| `y_neg` | Unwanted completion in the enriched forgetting examples |
| `retain` | Candidate clean code for the gate and utility evaluation |
| `library` | Task grouping; inferred from the canonical API root when absent |
| `deprecated api`, `replacement api`, `alias dict` | Static API-call evaluation |

The PDF reverses `y+`/`y-` names in one paragraph. The implementation consistently
uses the dataset's `y_pos = replacement`, `y_neg = unwanted` convention. It does
not use `probing input new`, which can already end inside the API expression and
therefore does not align with these full-line completion targets. Whitespace in
completion strings is preserved. Prompt and completion tokens are encoded
separately, so the conditional token boundary is explicit and reproducible.

Preparation validates fields and records rejected IDs/reasons in `invalid`.
Complete pairs from both categories in D_forget train the erasure direction.
The `up-to-dated` category is not a negative-gate label by itself. Preparation removes duplicate
pairs and test-overlapping contexts/functions before splitting. Shared prompts
and functions stay in the same split, using whitespace-normalized hashes.
It constructs retain pools from each split's supplied retain snippets and its
replacement-completed contexts (`prompt + y_pos`). Test code and the other
split's source functions/completed contexts are excluded. Shared retain candidates
are removed from the validation pool, keeping the pools disjoint.
Retain candidates containing a task's deprecated API calls are skipped. This is
an exact normalized-text overlap check, not semantic clone detection.

D_test is not required to contain `id`, `library`, `retain`, `y_pos` or `y_neg`.
Every example keeps its source row index; library names are inferred from API
roots (`torch` maps to `pytorch`). Missing pair targets stay missing and are not
fabricated. Optional probability diagnostics use the supplied `function` as the
retain text when `retain` is absent and report zero pair samples where no targets
exist. This does not affect generated API counts, which need no target completion.
Retain training snippets are a proxy constructed from the available benchmark,
not an independent general-code benchmark. Evaluate independent code tasks as
well before claiming preservation of broad coding capability.

Default task order is alphabetical by library, not an inferred real timeline.
Choose an explicit order before preparation:

```bash
python algo.py prepare --task-order numpy,pandas,pytorch,scipy,seaborn,sklearn,tensorflow,transformers
```

For a finer sequence, `--task-by api` groups by library plus deprecated API list.
Prepared data records task names/order, exclusions and retain-pool sizes.

For the pinned CodeLlama data, generation evaluation includes 10,395 forget and
17,026 test examples; one forget and five test rows have blank prompts. Training
preparation additionally excludes 286 context/function overlaps, 59 duplicate
pairs and 11 incomplete pairs, yielding 9,037 training and 1,002 validation pairs.
These filters affect training only: raw generation evaluation keeps all valid
examples and both categories. All exclusions are recorded.

**Migration from Drive:** prepare and train a new run. Old gates were fitted on
different data with a different retain policy and cannot be resumed into this
experiment. Defaults now use `checkpoints/codellama_hf/`, leaving old checkpoints
and raw files untouched. Checkpoints record the HF source manifest; evaluation
rejects a manifest mismatch instead of silently mixing model families/revisions.

## Train and Continue

### Local 4-bit NumPy Trial

For a pretrained-model experiment on a small NVIDIA GPU, install the optional
4-bit dependency in the project environment:

```bash
python -m pip install -r requirements-4bit.txt
python algo.py trial
```

This loads `codellama/CodeLlama-7b-hf` with NF4 double quantization, freezes the
base weights, uses 100 NumPy training examples, and evaluates the SAME 50 separate
validation examples before and after fitting the gate. All rows come from the
prepared HF data; no random model is substituted. The base model is loaded once.
Defaults use CUDA, FP16 computation, SDPA attention, a 512-token context window,
and 64 generated tokens. Quantization reduces GPU weight memory, but the first
run still downloads full original model weights and can need substantial disk
space and host RAM. An 8 GB GPU is not a guarantee that every context/model fits.
Model downloads are cached under `.cache/huggingface/hub` in the project by
default (`--cache-dir` overrides this), so later runs reuse the downloaded weights.

Outputs under `results/numpy_4bit_trial/`:

- `baseline.json`: written before gate training, so inspect whether M0 can emit API calls.
- `step_000.pt`, `step_001.pt`: baseline and learned steering checkpoint.
- `comparison.json`: paired predictions, counts, outcome transitions, and retain-NLL change.

Look for `deprecated->correct_rep`, not just `deprecated->mismatch`. Also inspect
`correct_rep->deprecated`, `correct_rep->mismatch`, and `delta_retain_nll` (a large
positive value signals degradation). Gate validation accuracy alone is not an
unlearning result. Sample counts are capped at available rows without duplication.

PowerShell in this workspace:

If the earlier model download stopped, run this from `project/` to resume the
pinned CodeLlama download in `.cache/models/CodeLlama-7b-hf`, verify its weight
checksums, and then run the 100/50 NumPy trial using those local files:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_local_trial.ps1
```

The runner records its phase (`downloading`, `evaluating`, `complete`, or `failed`)
in `.cache/local_trial/status.json`. Only `complete` means the comparison is ready.
An interrupted download reuses completed parts. If a trial has already written
`baseline.json`, preserve it and select a fresh `-Output results/numpy_4bit_trial_2`.
Keep the machine awake while downloading and evaluating. This runner performs
the single-task validation trial; it does not run the full eight-task benchmark.

For the standard Hugging Face cache route:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py algo.py trial
$r = Get-Content results/numpy_4bit_trial/comparison.json -Raw | ConvertFrom-Json
$r.baseline.summary
$r.steered.summary
$r.transitions
$r.delta_retain_nll
```

To compare layer/strength settings, rerun with a NEW output directory and the
same seed, e.g. `--layer 12 --strength 0.5 --output results/numpy_l12_s05`.
Tune on validation only, then evaluate on the independent test split. Existing
output directories are protected from accidental overwrite. No automatic
hyperparameter search or output-probability loss is introduced in this change.

Other model commands also accept `--quantization 4bit`; loading a gate checkpoint
requires matching quantization and recorded compute dtype. Never reuse gates
from the random-model smoke check as CodeLlama gates. This trial remains a
single-task development check, not a full continual-unlearning experiment.

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
python algo.py train --resume checkpoints/codellama_hf/step_001.pt
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

Positive gate examples are D_forget prompt states and unwanted completion
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

### Full local 4-bit run (PowerShell)

After downloading CodeLlama, run `powershell -NoProfile -ExecutionPolicy Bypass -File
.\run_full_dataset.ps1` from `project/` (as one command). This trains all eight
tasks with all 9,037 prepared training pairs and all 1,002 validation pairs,
subject to the recorded context-length exclusions. It uses the tested local
4-bit model, a 512-token context, and the default gate hyperparameters.
It then evaluates `step_008.pt` on every valid raw D_forget and D_test row,
generating up to 64 tokens per row. This evaluates the final cumulative bank;
it does not evaluate every intermediate checkpoint.

Checkpoints go to `checkpoints/codellama_full_4bit/` and the report to
`results/codellama_full_4bit/api_counts.json`. Check `status.json` in that results
directory: only `complete` means both datasets finished. A report written while
evaluation is running may contain just D_forget. The runner refuses to overwrite
an existing training run. A full run can take many hours or days on a laptop;
keep it powered and awake. For a background launch with redirected logs, monitor:

```powershell
Get-Content results/codellama_full_4bit/stdout.log -Tail 10 -Wait
```

### Generated API Counts on Both Raw Datasets

The main evaluation now generates code for every valid row in the original
`D_forget.json` and `D_test.json`, including duplicates, mixed categories, training
examples, and rows without `y_pos`/`y_neg`/`retain`. It does not substitute the
prepared validation split for the full forget dataset. See the migration note
above when replacing the old Drive dataset.

```bash
python algo.py evaluate-api --checkpoint checkpoints/codellama_hf/step_008.pt
```

Omit `--checkpoint` to evaluate every stage, including `step_000.pt` (M0).
Default output: `results/api_counts.json`. Default generation is greedy, 64 new
tokens, with `--max-samples 0` meaning the entire raw dataset. A positive sample
limit selects that many rows per dataset, not per library.

| Count | Definition |
| --- | --- |
| `no_dep_count` | No targeted deprecated API call in the generated continuation |
| `correct_rep_count` | Target replacement call present, with no targeted deprecated call |
| `mismatch_count` | Neither targeted deprecated nor target replacement call present |
| `deprecated_count` | At least one targeted deprecated call, even if replacement also appears |
| `both_dep_and_rep_count` | Both call types present; a subset of deprecated_count |

`no_dep_count = correct_rep_count + mismatch_count`.
`evaluated = deprecated_count + correct_rep_count + mismatch_count`.
These are sample counts, not numbers of call occurrences. Each count also has a
rate using `evaluated` as denominator. Empty output is mismatch, not correct rep.
Correct rep means API-name correctness, not verified arguments or functional
correctness. A truncated `np.prod(` counts as a replacement call-name prefix.

Matching uses Python tokenization of generated text only, excluding comments,
string literals and function/class definitions. It resolves exact aliases from
each row's `alias dict`, without executing code. It does not resolve arbitrary
runtime bindings or infer new aliases. On tokenization errors, already-tokenized
calls are used and the error is recorded per example for audit. Calls already in
the prompt are not counted. Prompt input remains `probing input`.

The JSON includes per-dataset totals, per-library/category counts, every generated
continuation and its classification, file fingerprints, and skipped source row
indices/reasons. Blank prompts or missing API labels are skipped and explicitly
reported, never counted as successful forgetting. Missing completion targets do
not prevent generation evaluation. Full D_forget metrics include training data
and are not an independent generalization estimate.
For D_test, the report also contains `subsets.U_dep` and `subsets.U_nondep` with
the same counts/rates, without running generation twice. Subset membership is
matched using original benchmark fields, not the optional IDs/enriched fields.
Whole-dataset and subgroup counts use evaluated examples, with skipped rows
reported separately.

### Optional Probability Diagnostics

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
For the raw HF D_test, pair metrics are unavailable because y_pos/y_neg are absent.
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
python algo.py generate --checkpoint checkpoints/codellama_hf/step_008.pt --prompt-file data/prompt.txt
```

On Bash/WSL, `bash run_script.sh` downloads the pinned dataset, runs prepare, train and full generated API-count
evaluation on both raw datasets at every checkpoint. This can be expensive: nine
checkpoints require nine generations per valid sample across both datasets.
Any arguments
are forwarded directly, e.g. `bash run_script.sh train --stop-after 1`.
Set `FAMILY` and `MODEL` together to change the no-argument pipeline's dataset
family and model. Results go to `results/${FAMILY}_api_counts.json`.
Set `PYTHON` to select
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

Earlier Drive smoke artifacts under `results/offline_smoke/` are obsolete for
the canonical HF experiment and must not be used as research results or gates.
