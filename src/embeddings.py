# src/embeddings.py
"""
Embedding + Indexing pipeline (NumPy backend).

Flow:
- Read chunk JSONL files
- Generate embeddings with SentenceTransformers
- L2-normalize vectors
- Store vectors as .npy
- Store metadata as meta.jsonl

NOTE:
- This is a CPU-safe, FAISS-free backend
- Designed to be replaced later by FAISS/HNSW if needed
"""

import json
import time
import argparse
from pathlib import Path
from typing import Iterator, List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


# --------- Utils ---------
def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def backup_if_exists(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(path.suffix + f".bak-{timestamp()}")
        path.replace(bak)


def iter_jsonl(dir_path: Path) -> Iterator[Dict]:
    jsonl_files = sorted(dir_path.glob("chunks*.jsonl"))
    if not jsonl_files:
        jsonl_files = sorted(dir_path.glob("*.jsonl"))

    for fp in jsonl_files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


# --------- Embedding ---------
def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        e = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embs.append(e.astype("float32", copy=False))

    X = np.vstack(embs)
    return l2_normalize(X).astype("float32", copy=False)


# --------- Main ---------
def main(input_dir: str, output_dir: str, batch_size: int) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔹 Loading embedding model (CPU-safe)")
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    print(f"🔹 Reading chunks from {in_dir}")
    chunks = [c for c in iter_jsonl(in_dir) if c.get("text")]
    texts = [c["text"] for c in chunks]

    if not texts:
        print("⚠️ No chunks found")
        return

    print(f"   → {len(texts)} chunks")

    print("🔹 Computing embeddings")
    X = embed_texts(model, texts, batch_size=batch_size)
    print(f"   → Embedding shape: {X.shape}")

    vec_path = out_dir / "vectors.npy"
    meta_path = out_dir / "meta.jsonl"

    backup_if_exists(vec_path)
    backup_if_exists(meta_path)

    print(f"🔹 Writing vectors → {vec_path}")
    np.save(vec_path, X)

    print(f"🔹 Writing metadata → {meta_path}")
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("✅ Index build finished (NumPy backend)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/chunks")
    parser.add_argument("--out", default="data/index")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    main(args.input, args.out, args.batch)
