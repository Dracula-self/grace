# Data Preparation

## Datasets

This pipeline supports:

- `devign` from `DetectVul/devign`
- `reveal` from `claudios/ReVeal`
- `bigvul` from `bstee615/bigvul`

The normalized schema is:

- `id`
- `func`
- `target`
- `project`
- `commit_id`
- `split`
- `source_dataset`

## Commands

Download raw splits:

```bash
python scripts/download_datasets.py --dataset all
```

Normalize the raw files:

```bash
python scripts/prepare_dataset.py --dataset all
```

Output layout:

```text
data/
  raw/<dataset>/<split>.jsonl
  processed/<dataset>/<split>.jsonl
```

## Field Mapping

The mapping rules live in [configs/datasets.yaml](/home/dlts/GRACE/configs/datasets.yaml).

The normalizer checks candidate field names in order, so small upstream schema changes can be absorbed without patching Python code.

