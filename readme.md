# GRACE Local Pipeline

This repository repackages the open-source code for the paper "GRACE: Empowering LLM-based Software Vulnerability Detection with Graph Structure and In-context Learning" into a reproducible local pipeline.

The original repository depended on:

- expired Google Drive dataset links
- a remote GPT-4 API
- hard-coded local file paths

This version replaces that setup with:

- Hugging Face dataset download scripts for `Devign`, `ReVeal`, and `BigVul`
- Joern installation and graph extraction scripts
- local model download and serving scripts
- an evaluation entrypoint that supports both OpenAI-compatible servers and direct `transformers` inference

## Repository Layout

- `scripts/download_datasets.py`: fetch datasets from Hugging Face
- `scripts/prepare_dataset.py`: normalize all datasets to one schema
- `scripts/run_joern_extract.py`: build node / edge text with Joern
- `scripts/build_retrieval_index.py`: build the CodeT5 + FAISS retrieval index
- `scripts/generate_examples.py`: create in-context examples
- `scripts/build_grace_json.py`: merge graph and example features into the final JSON
- `scripts/run_eval.py`: run `basep` or full `grace` inference
- `scripts/install_joern.sh`: install Joern from the official release page
- `scripts/download_models.sh`: download CodeT5 and local LLM weights
- `scripts/serve_openai_compatible.sh`: run a local OpenAI-compatible service with vLLM

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Download and normalize datasets:

```bash
python scripts/download_datasets.py --dataset all
python scripts/prepare_dataset.py --dataset all
```

Install Joern:

```bash
bash scripts/install_joern.sh
export JOERN_HOME=$PWD/tools/joern/2.0.42
```

Extract graph text:

```bash
python scripts/run_joern_extract.py --dataset devign --split train --joern-home "$JOERN_HOME" --workers 4
python scripts/run_joern_extract.py --dataset devign --split test --joern-home "$JOERN_HOME" --workers 4
```

Download models:

```bash
bash scripts/download_models.sh --with-awq
```

Build retrieval index and examples:

```bash
python scripts/build_retrieval_index.py --dataset devign
python scripts/generate_examples.py --dataset devign --split test
python scripts/build_grace_json.py --dataset devign --split test
```

Run full GRACE through an OpenAI-compatible local server:

```bash
MODEL_PATH=models/local-awq bash scripts/serve_openai_compatible.sh
python scripts/run_eval.py --dataset devign --mode grace --max-samples 10
```

Run direct `transformers` inference:

1. Edit `configs/inference.local.yaml`
2. Set `provider: transformers`
3. Set `model_name_or_path` to the downloaded local model path

Then run:

```bash
python scripts/run_eval.py --dataset devign --mode grace --max-samples 10
```

## Compatibility Wrappers

The historical entrypoints still exist:

- `python llmpre.py`
- `python basep.py`
- `python genexample.py`

They now delegate to the new parameterized pipeline. By default they use:

- dataset: `devign`
- config: `configs/inference.local.yaml`

You can override them with:

```bash
export GRACE_DATASET=bigvul
export GRACE_CONFIG=configs/inference.local.yaml
```

## Additional Documentation

- [Data Preparation](docs/data_preparation.md)
- [Joern Setup](docs/joern_setup.md)
- [Local LLM Setup](docs/local_llm_setup.md)
- [Run Pipeline](docs/run_pipeline.md)

## Reference Sources

- Devign: https://huggingface.co/datasets/DetectVul/devign
- ReVeal: https://huggingface.co/datasets/claudios/ReVeal
- BigVul: https://huggingface.co/datasets/bstee615/bigvul
- CodeT5-base: https://huggingface.co/Salesforce/codet5-base
- Joern installation: https://docs.joern.io/installation/
- Joern export: https://docs.joern.io/export/
