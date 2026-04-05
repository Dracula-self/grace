from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRACE evaluation with local or OpenAI-compatible models.")
    parser.add_argument("--config", default="configs/inference.local.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=["basep", "grace"], default="grace")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def calculate_metrics(predictions: list[int], targets: list[int]) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for pred, target in zip(predictions, targets):
        if pred == target == 1:
            tp += 1
        elif pred == target == 0:
            tn += 1
        elif pred == 1 and target == 0:
            fp += 1
        elif pred == 0 and target == 1:
            fn += 1

    accuracy = (tp + tn) / len(predictions) if predictions else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    from grace.config import load_yaml
    from grace.inference import build_provider
    from grace.io import load_json
    from grace.prompting import build_prompt, normalize_prediction

    args = parse_args()
    config = load_yaml(args.config)
    provider = build_provider(config["inference"])
    input_file = Path(args.input_file or f"data/processed/{args.dataset}/test_processed.json")
    rows = load_json(input_file)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {input_file}")
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    output_dir = Path(args.output_dir) / args.dataset / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=output_dir / "metrics.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    prediction_rows = []
    parsed_predictions: list[int] = []
    targets: list[int] = []
    for row in rows:
        prompt = build_prompt(row, args.mode)
        raw_prediction = provider.generate(prompt)
        parsed = normalize_prediction(raw_prediction)
        parsed_int = parsed if parsed is not None else 2
        prediction_rows.append(
            {
                "id": row["id"],
                "prediction_raw": raw_prediction,
                "prediction": parsed_int,
                "ground_truth": row["target"],
            }
        )
        if parsed is not None:
            parsed_predictions.append(parsed)
            targets.append(int(row["target"]))

    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "prediction_raw", "prediction", "ground_truth"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    metrics = calculate_metrics(parsed_predictions, targets)
    for key, value in metrics.items():
        logging.info("%s: %.6f", key, value)
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
