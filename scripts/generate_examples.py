from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate in-context examples for GRACE.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--train-graph-file", default=None)
    parser.add_argument("--test-graph-file", default=None)
    parser.add_argument("--model-name-or-path", default="Salesforce/codet5-base")
    parser.add_argument("--index-dir", default="artifacts/retrieval")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="data/examples")
    return parser.parse_args()


def load_graph_map(path: Path) -> dict[str, dict]:
    from grace.io import read_jsonl

    if not path.exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def main() -> None:
    args = parse_args()
    from grace.io import read_jsonl, write_jsonl
    from grace.retrieval import encode_texts, load_index, mixed_similarity, transform_and_normalize

    test_file = Path(args.test_file or f"data/processed/{args.dataset}/{args.split}.jsonl")
    train_graph_file = Path(args.train_graph_file or f"data/cpg/{args.dataset}/train.jsonl")
    test_graph_file = Path(args.test_graph_file or f"data/cpg/{args.dataset}/{args.split}.jsonl")

    test_records = read_jsonl(test_file)
    train_graph_map = load_graph_map(train_graph_file)
    test_graph_map = load_graph_map(test_graph_file)

    index, train_records, mean, kernel = load_index(Path(args.index_dir) / args.dataset)
    train_map = {row["id"]: row for row in train_records}

    query_vectors = encode_texts(
        [row["func"] for row in test_records],
        args.model_name_or_path,
        args.batch_size,
        args.device,
    )
    query_vectors = transform_and_normalize(query_vectors, mean, kernel)
    _, candidate_indices = index.search(query_vectors, args.top_k)

    example_rows = []
    for test_row, candidates in zip(test_records, candidate_indices):
        test_graph = test_graph_map.get(test_row["id"], {})
        test_ast = test_graph.get("ast", "")
        best_score = -1.0
        best_train = None
        for idx in candidates.tolist():
            train_row = train_records[idx]
            train_graph = train_graph_map.get(train_row["id"], {})
            score = mixed_similarity(
                test_row["func"],
                train_row["func"],
                test_ast,
                train_graph.get("ast", ""),
            )
            if score > best_score:
                best_score = score
                best_train = train_row

        if best_train is None:
            best_train = train_records[int(candidates[0])]
            best_score = 0.0

        example_rows.append(
            {
                "id": test_row["id"],
                "example_id": best_train["id"],
                "example": best_train["func"],
                "example_target": best_train["target"],
                "score": best_score,
            }
        )

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}.jsonl"
    write_jsonl(output_path, example_rows)
    print(f"[examples] wrote {len(example_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
