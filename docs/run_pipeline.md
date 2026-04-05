# Run Pipeline

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Download Data

```bash
python scripts/download_datasets.py --dataset devign
python scripts/prepare_dataset.py --dataset devign
```

## 3. Install Joern

```bash
bash scripts/install_joern.sh
export JOERN_HOME=$PWD/tools/joern/2.0.42
```

## 4. Generate CPG Text

```bash
python scripts/run_joern_extract.py --dataset devign --split train --joern-home "$JOERN_HOME" --workers 4
python scripts/run_joern_extract.py --dataset devign --split test --joern-home "$JOERN_HOME" --workers 4
```

## 5. Download Models

```bash
bash scripts/download_models.sh --with-awq
```

## 6. Build Retrieval Assets

```bash
python scripts/build_retrieval_index.py --dataset devign
python scripts/generate_examples.py --dataset devign --split test
```

## 7. Merge Final Evaluation JSON

```bash
python scripts/build_grace_json.py --dataset devign --split test
```

## 8. Run Smoke Test

OpenAI-compatible service:

```bash
MODEL_PATH=models/local-awq bash scripts/serve_openai_compatible.sh
python scripts/run_eval.py --dataset devign --mode grace --max-samples 10
```

Direct `transformers` route:

```bash
python scripts/run_eval.py --dataset devign --mode grace --max-samples 10
```

## 9. Run Baseline Prompt

```bash
python scripts/run_eval.py --dataset devign --mode basep --max-samples 10
```

## 10. Scale to Platform

On the 48G GPU platform:

- switch to `Qwen2.5-Coder-14B-Instruct` or the AWQ variant
- increase Joern worker count carefully
- remove `--max-samples`
- run each dataset separately to simplify failure recovery

