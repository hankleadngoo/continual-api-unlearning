# Canonical Data

Source: https://huggingface.co/datasets/tummitum/Data-Collection

```bash
python algo.py fetch-data --family codellama
python algo.py prepare
```

Default revision: `07a1ca0083ab8b0a71c18a43195330cf495f475a`.
Families: codellama, codegen, deepseek, starcoder. Never mix their split files.

```text
data/
  codellama/
    D_forget.json      # 10,396 raw records
    D_test.json        # 17,031 raw records; no enriched target/retain fields
    D_test_U_dep.json  # 1,310 raw records; U_nondep is its complement in D_test
    source.json        # Resolved commit, family, counts, SHA-256 checksums
    prepared.json      # Generated train/validation and normalized test data
```

Data and cache files are excluded from Git at every directory depth. Raw files
remain unchanged. D_forget contains outdated examples AND model-specific
up-to-dated examples on which the base model still used deprecated APIs.

Legacy Drive JSON files at the data root, if present, are not used by default.
They are not overwritten by fetch-data. Train new gates after migrating; the
full workflow and evaluation definitions are documented in the project README.
