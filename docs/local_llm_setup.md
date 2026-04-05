# Local LLM Setup

## Route A: OpenAI-Compatible Local Service

This route keeps the original GRACE calling pattern close to the paper code.

Recommended for:

- local smoke tests
- quick backend swapping
- 48G platform runs with vLLM

Install:

```bash
pip install vllm
```

Serve the local model:

```bash
MODEL_PATH=models/local-awq bash scripts/serve_openai_compatible.sh
```

Use the default config:

```yaml
inference:
  provider: openai_compatible
  model_name: models/local-awq
  base_url: http://127.0.0.1:8000/v1
```

## Route B: Direct Transformers Inference

Recommended for:

- environments where you do not want an extra serving process
- controlled single-GPU runs

Edit [configs/inference.local.yaml](/home/dlts/GRACE/configs/inference.local.yaml) to:

```yaml
inference:
  provider: transformers
  model_name_or_path: models/local-awq
  device_map: auto
  torch_dtype: auto
```

## Model Recommendations

For 2070S local debugging:

- `Qwen/Qwen2.5-Coder-3B-Instruct-AWQ`

For the 48G platform:

- `Qwen/Qwen2.5-Coder-14B-Instruct`
- `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`

Always download CodeT5 as well:

```bash
bash scripts/download_models.sh --with-awq
```
