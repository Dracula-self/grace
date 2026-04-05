from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CodeT5 + FAISS retrieval index for GRACE.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--model-name-or-path", default="Salesforce/codet5-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="artifacts/retrieval")
    parser.add_argument("--whitening-dim", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from grace.io import read_jsonl
    from grace.retrieval import build_faiss_index, encode_texts, fit_whitening, save_index, transform_and_normalize

    train_file = Path(args.train_file or f"data/processed/{args.dataset}/train.jsonl")
    records = read_jsonl(train_file)
    texts = [row["func"] for row in records]
    vectors = encode_texts(texts, args.model_name_or_path, args.batch_size, args.device)
    mean, kernel = fit_whitening(vectors, args.whitening_dim)
    normalized = transform_and_normalize(vectors, mean, kernel)
    index = build_faiss_index(normalized)
    output_dir = Path(args.output_dir) / args.dataset
    save_index(output_dir, index, records, mean, kernel)
    print(f"[retrieval] index saved to {output_dir}")


if __name__ == "__main__":
    main()
