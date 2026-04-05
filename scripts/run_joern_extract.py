from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grace.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Joern node/edge text for GRACE.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/cpg")
    parser.add_argument("--joern-home", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def resolve_binary(joern_home: str, name: str) -> str:
    if joern_home:
        candidate = Path(joern_home) / name
        if candidate.exists():
            return str(candidate)
    return name


def pick_first(row: dict[str, str], keys: list[str], default: str = "") -> str:
    for key in keys:
        if key in row and row[key]:
            return row[key]
    return default


def parse_csv_graph(graph_dir: Path) -> tuple[str, str, str]:
    node_candidates = [
        graph_dir / "parsed" / "nodes.csv",
        graph_dir / "nodes.csv",
    ]
    edge_candidates = [
        graph_dir / "parsed" / "edges.csv",
        graph_dir / "edges.csv",
    ]
    node_path = next((path for path in node_candidates if path.exists()), None)
    edge_path = next((path for path in edge_candidates if path.exists()), None)

    if node_path is None or edge_path is None:
        csv_files = list(graph_dir.rglob("*.csv"))
        node_path = next((path for path in csv_files if path.name == "nodes.csv"), node_path)
        edge_path = next((path for path in csv_files if path.name == "edges.csv"), edge_path)
    if node_path is None or edge_path is None:
        raise FileNotFoundError(f"Could not locate nodes.csv / edges.csv under {graph_dir}")

    nodes_text: list[str] = []
    ast_text: list[str] = []
    with node_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node_id = pick_first(row, ["id", "key", ":ID", "ID"])
            label = pick_first(row, ["label", "_label", "type", "TYPE"])
            code = pick_first(row, ["code", "CODE", "name", "NAME", "FULL_NAME"])
            if label:
                ast_text.append(label)
            nodes_text.append(f"{node_id}:{label}:{code}")

    edges_text: list[str] = []
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            src = pick_first(row, ["start", "start_id", ":START_ID", "OUT_NODE"])
            dst = pick_first(row, ["end", "end_id", ":END_ID", "IN_NODE"])
            label = pick_first(row, ["label", "_label", "type", "TYPE"])
            edges_text.append(f"{src}->{dst}:{label}")

    return " ".join(ast_text), "\n".join(nodes_text), "\n".join(edges_text)


def run_joern(record: dict, joern_home: str) -> dict:
    source_suffix = ".c"
    parse_bin = resolve_binary(joern_home, "joern-parse")
    with tempfile.TemporaryDirectory(prefix="grace_joern_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        src_dir = tmpdir_path / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        source_file = src_dir / f"{record['id']}{source_suffix}"
        source_file.write_text(record["func"], encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("JAVA_OPTS", "-Xms1G -Xmx8G")

        commands = [
            [parse_bin, str(src_dir)],
        ]
        last_error = ""
        for command in commands:
            try:
                subprocess.run(
                    command,
                    cwd=tmpdir,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                ast_text, node_text, edge_text = parse_csv_graph(tmpdir_path)
                return {
                    "id": record["id"],
                    "ast": ast_text,
                    "node": node_text,
                    "edge": edge_text,
                    "graph_error": "",
                }
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                last_error = str(exc)
        return {
            "id": record["id"],
            "ast": "",
            "node": "",
            "edge": "",
            "graph_error": last_error or "joern extraction failed",
        }


def process_split(records: list[dict], output_path: Path, joern_home: str, workers: int) -> None:
    if workers <= 1:
        results = [run_joern(record, joern_home) for record in records]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(lambda item: run_joern(item, joern_home), records))
    write_jsonl(output_path, results)
    print(f"[joern] wrote {len(results)} records to {output_path}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir) / args.dataset
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]
    for split in splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        records = read_jsonl(input_path)
        if args.max_samples > 0:
            records = records[: args.max_samples]
        output_path = output_dir / f"{split}.jsonl"
        process_split(records, output_path, args.joern_home, args.workers)


if __name__ == "__main__":
    main()
