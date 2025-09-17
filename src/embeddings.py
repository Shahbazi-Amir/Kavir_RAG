# src/embeddings.py
"""
Embedding + Indexing pipeline for local RAG.
- Reads JSONL chunk files from an input dir
- Generates embeddings with multilingual-e5-small (SentenceTransformers)
- L2-normalizes vectors (safety check even if model normalizes)
- Builds a FAISS Inner-Product index (cosine via normalized vectors)
- Writes index + metadata; makes timestamped backups if outputs exist
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Iterator, List, Dict

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# --------- IO & Utils ---------
def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def backup_if_exists(path: Path) -> None:
    """If file exists, rename it with a .bak-<timestamp> suffix."""
    if path.exists():
        bak = path.with_suffix(path.suffix + f".bak-{timestamp()}")
        path.replace(bak)


def iter_jsonl(dir_path: Path) -> Iterator[Dict]:
    """
    Yield records from all *.jsonl files in dir_path.
    Prefer files that look like 'chunks*.jsonl' to avoid reading manifests.
    """
    jsonl_files: List[Path] = []

    # First, any chunks*.jsonl
    jsonl_files += sorted(dir_path.glob("chunks*.jsonl"))
    # Fallback to all .jsonl if none matched
    if not jsonl_files:
        jsonl_files = sorted(dir_path.glob("*.jsonl"))

    for fp in jsonl_files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield rec


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


# --------- Embedding ---------
def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts in batches; returns (N, D) float32 array."""
    embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # model-level normalization
        )
        # Safety: ensure float32 and normalized
        emb = emb.astype("float32", copy=False)
        embs.append(emb)
    X = np.vstack(embs) if embs else np.zeros((0, 384), dtype="float32")
    # Extra safety normalization
    X = l2_normalize(X).astype("float32", copy=False)
    return X


# --------- Main ---------
def main(input_dir: str, output_dir: str, batch_size: int = 32) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔹 Loading embedding model: intfloat/multilingual-e5-small")
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    print(f"🔹 Reading chunks from: {in_dir}")
    chunks: List[Dict] = [rec for rec in iter_jsonl(in_dir) if "text" in rec and rec["text"]]
    texts: List[str] = [c["text"] for c in chunks]
    print(f"   → Loaded {len(chunks)} chunks")

    if not chunks:
        print("⚠️ No chunks found. Aborting.")
        return

    print("🔹 Computing embeddings...")
    X = embed_texts(model, texts, batch_size=batch_size)
    dim = X.shape[1]
    print(f"   → Shape: {X.shape}")

    print("🔹 Building FAISS IP index (cosine via normalized vectors)...")
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    assert index.ntotal == X.shape[0]

    # Paths
    idx_path = out_dir / "faiss.index"
    meta_path = out_dir / "meta.jsonl"

    # Backups if exist
    backup_if_exists(idx_path)
    backup_if_exists(meta_path)

    print(f"🔹 Writing index → {idx_path}")
    faiss.write_index(index, str(idx_path))

    print(f"🔹 Writing metadata → {meta_path}")
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("✅ Done.")
    print(f"   Index size: {index.ntotal} vectors | dim={dim}")
    print(f"   Files written: {idx_path.name}, {meta_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/chunks", help="Path to chunked JSONL files")
    parser.add_argument("--out", default="data/index", help="Output folder for FAISS index and metadata")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for embeddings")
    args = parser.parse_args()
    main(args.input, args.out, args.batch)
