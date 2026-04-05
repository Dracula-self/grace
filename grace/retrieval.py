from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def mean_pool(hidden_states, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(
    texts: Iterable[str],
    model_name_or_path: str,
    batch_size: int = 8,
    device: str = "auto",
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    outputs: list[np.ndarray] = []
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) < batch_size:
            continue
        outputs.append(_encode_batch(batch, tokenizer, model, device))
        batch = []
    if batch:
        outputs.append(_encode_batch(batch, tokenizer, model, device))
    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _encode_batch(batch, tokenizer, model, device):
    import torch

    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
        pooled = mean_pool(hidden, encoded["attention_mask"])
    return pooled.detach().cpu().numpy().astype(np.float32)


def fit_whitening(vectors: np.ndarray, output_dim: int = 256):
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    cov = np.cov(centered, rowvar=False)
    u, s, _ = np.linalg.svd(cov)
    dim = min(output_dim, u.shape[1])
    kernel = np.dot(u[:, :dim], np.diag(1.0 / np.sqrt(s[:dim] + 1e-12)))
    return mean.astype(np.float32), kernel.astype(np.float32)


def transform_and_normalize(vectors: np.ndarray, mean: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    transformed = np.dot(vectors - mean, kernel)
    norm = np.linalg.norm(transformed, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (transformed / norm).astype(np.float32)


def build_faiss_index(vectors: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index


def save_index(output_dir: str | Path, index, train_records: list[dict], mean: np.ndarray, kernel: np.ndarray) -> None:
    import faiss

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    np.save(output_dir / "mean.npy", mean)
    np.save(output_dir / "kernel.npy", kernel)
    with (output_dir / "train_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in train_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_index(index_dir: str | Path):
    import faiss

    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    mean = np.load(index_dir / "mean.npy")
    kernel = np.load(index_dir / "kernel.npy")
    train_records: list[dict] = []
    with (index_dir / "train_records.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            train_records.append(json.loads(line))
    return index, train_records, mean, kernel


def lexical_similarity(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens and not b_tokens:
        return 1.0
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


def syntactic_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.split(), b=b.split()).ratio()


def mixed_similarity(code_a: str, code_b: str, ast_a: str, ast_b: str, lexical_weight: float = 0.7) -> float:
    lexical = lexical_similarity(code_a, code_b)
    syntactic = syntactic_similarity(ast_a, ast_b)
    return lexical_weight * lexical + (1.0 - lexical_weight) * syntactic

