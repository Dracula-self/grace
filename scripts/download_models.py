from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_PRESETS = {
    "codet5": "Salesforce/codet5-base",
    "local-awq": "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ",
    "local-full": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "platform-awq": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
    "platform-full": "Qwen/Qwen2.5-Coder-14B-Instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model assets for GRACE.")
    parser.add_argument("--models", nargs="+", default=["codet5", "local-awq"])
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import snapshot_download

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in args.models:
        repo_id = MODEL_PRESETS.get(item, item)
        local_dir = output_dir / item.replace("/", "_")
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
        print(f"[model] {repo_id} -> {local_dir}")


if __name__ == "__main__":
    main()
