from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    default_config = REPO_ROOT / "configs" / "datasets.yaml"
    parser = argparse.ArgumentParser(description="Normalize vulnerability datasets for GRACE.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def pick_field(record: dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    for field in candidates:
        if field in record and record[field] not in (None, ""):
            return record[field]
    return default


def coerce_target(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "vulnerable", "yes"}:
        return 1
    if text in {"0", "false", "non-vulnerable", "non_vulnerable", "clean", "no"}:
        return 0
    return None


def normalize_record(dataset_name: str, split: str, index: int, record: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    func = pick_field(record, spec["field_candidates"]["func"])
    target = coerce_target(pick_field(record, spec["field_candidates"]["target"]))
    if not func or target is None:
        return None
    return {
        "id": str(pick_field(record, spec["field_candidates"]["id"], f"{dataset_name}-{split}-{index}")),
        "func": str(func),
        "target": target,
        "project": pick_field(record, spec["field_candidates"].get("project", []), ""),
        "commit_id": pick_field(record, spec["field_candidates"].get("commit_id", []), ""),
        "split": split,
        "source_dataset": dataset_name,
    }


def main() -> None:
    from grace.config import ensure_dir, load_yaml
    from grace.io import read_jsonl, write_jsonl

    args = parse_args()
    config = load_yaml(args.config)
    selected = (
        config["datasets"].keys()
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset_name in selected:
        spec = config["datasets"][dataset_name]
        input_dir = Path(args.input_dir) / dataset_name
        output_dir = ensure_dir(Path(args.output_dir) / dataset_name)
        for split in spec.get("splits", ["train", "validation", "test"]):
            input_path = input_dir / f"{split}.jsonl"
            if not input_path.exists():
                continue
            rows = read_jsonl(input_path)
            normalized = []
            skipped = 0
            for index, row in enumerate(rows):
                item = normalize_record(dataset_name, split, index, row, spec)
                if item is None:
                    skipped += 1
                    continue
                normalized.append(item)
            output_path = output_dir / f"{split}.jsonl"
            write_jsonl(output_path, normalized)
            print(f"[prepared] {dataset_name}:{split} -> {output_path} ({len(normalized)} kept, {skipped} skipped)")


if __name__ == "__main__":
    main()
