from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grace.io import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge normalized data, graph data, and examples into GRACE JSON.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--records-file", default=None)
    parser.add_argument("--graph-file", default=None)
    parser.add_argument("--examples-file", default=None)
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_file = Path(args.records_file or f"data/processed/{args.dataset}/{args.split}.jsonl")
    graph_file = Path(args.graph_file or f"data/cpg/{args.dataset}/{args.split}.jsonl")
    examples_file = Path(args.examples_file or f"data/examples/{args.dataset}/{args.split}.jsonl")
    output_file = Path(args.output_file or f"data/processed/{args.dataset}/{args.split}_processed.json")

    records = {row["id"]: row for row in read_jsonl(records_file)}
    graph_map = {row["id"]: row for row in read_jsonl(graph_file)} if graph_file.exists() else {}
    example_map = {row["id"]: row for row in read_jsonl(examples_file)} if examples_file.exists() else {}

    merged = []
    for record_id, row in records.items():
        graph = graph_map.get(record_id, {})
        example = example_map.get(record_id, {})
        merged.append(
            {
                **row,
                "node": graph.get("node", ""),
                "edge": graph.get("edge", ""),
                "ast": graph.get("ast", ""),
                "example": example.get("example", ""),
                "example_target": example.get("example_target"),
                "example_id": example.get("example_id", ""),
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_file, merged)
    print(f"[merge] wrote {len(merged)} records to {output_file}")


if __name__ == "__main__":
    main()
