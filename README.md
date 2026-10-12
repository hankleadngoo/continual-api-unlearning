# Continual API Unlearning

An experimental code-model pipeline for reducing deprecated API completions through learned activation steering.

## Implemented workflow

1. Prepare matching forget/test data and a code model.
2. Train a steering gate from paired examples.
3. Generate baseline and steered completions on the same evaluation sample.
4. Report deprecated API, replacement, and mismatch counts.

The current default compares steering OFF with fixed strength `t=0.5` on at most 500 valid test rows in total. Generation is batched; reports restore sample order. The default server configuration uses DeepSeek-Coder 1.3B Instruct and a next-line completion prompt.

## Setup and run

Run from the source repository root on a Linux server with Conda; a compatible GPU is recommended for model execution.

```bash
bash setup_conda.sh
TRAIN_SAMPLES=20 EVAL_SAMPLES=20 GATE_STEPS=10 bash run_script.sh
```

For the default experiment:

```bash
bash run_script.sh
```

Do not reuse a CodeLlama gate with DeepSeek. Dataset and model downloads require network access and suitable disk space. See the original operational notes for environment overrides and Slurm settings.

## Repository layout

```text
algo.py               Training and evaluation CLI
setup_conda.sh        Environment setup
run_script.sh         Server pipeline entry point
run_slurm.sh          Cluster submission
examples/             Existing examples
tests/                Existing tests
```

## Evidence and limitations

Publish a small, completed `comparison.json` with model, dataset revision, sample size, and hardware details after checking it for private data. A report should contain `complete: true` before being presented as a finished evaluation. Results on a capped sample do not establish performance on the full dataset. Historical strength-search instructions in the existing README need consolidation before release.

## Engineering focus

Model instrumentation, batched generation, experiment configuration, and transparent comparison of baseline and intervention.

## Portfolio evidence

This README describes the inspected local implementation. Full model training and production integrations have not been rerun for this portfolio release. No benchmark improvement or production reliability claim is made without a reproducible report.

For team or reproduction work, document your specific changes in `docs/CONTRIBUTIONS.md` before publishing. Preserve upstream attribution and applicable licenses. Include only synthetic or redistribution-approved examples.

## Operational notes

See [docs/OPERATIONS.md](docs/OPERATIONS.md) for preserved source-project notes. Historical options may differ from the current entry points.

## Validation

See [docs/VALIDATION.md](docs/VALIDATION.md) for checks performed and their limitations.
