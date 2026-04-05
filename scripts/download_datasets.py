from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    default_config = REPO_ROOT / "configs" / "datasets.yaml"
    parser = argparse.ArgumentParser(description="Download GRACE datasets from Hugging Face.")
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    from grace.config import ensure_dir, load_yaml

    args = parse_args()
    config = load_yaml(args.config)
    selected = (
        config["datasets"].keys()
        if args.dataset == "all"
        else [args.dataset]
    )

    from datasets import load_dataset

    for name in selected:
        spec = config["datasets"][name]
        dataset = load_dataset(
            spec["hf_id"],
            cache_dir=args.cache_dir,
        )
        target_dir = ensure_dir(Path(args.output_dir) / name)
        for split_name, split_data in dataset.items():
            output_path = target_dir / f"{split_name}.jsonl"
            with output_path.open("w", encoding="utf-8") as handle:
                for row in split_data:
                    handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            print(f"[downloaded] {name}:{split_name} -> {output_path}")


if __name__ == "__main__":
    main()
