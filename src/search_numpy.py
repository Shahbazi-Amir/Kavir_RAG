# src/search_numpy.py
# English comments inside code.

import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


def load_meta(meta_path: Path) -> List[Dict]:
    meta = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    q = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return q.astype("float32")


def main(index_dir: str, query: str, k: int) -> None:
    index_dir = Path(index_dir)
    vec_path = index_dir / "vectors.npy"
    meta_path = index_dir / "meta.jsonl"

    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError("vectors.npy or meta.jsonl not found")

    print("🔹 Loading vectors & metadata")
    X = np.load(vec_path)               # shape: (N, D)
    meta = load_meta(meta_path)

    if len(meta) != X.shape[0]:
        print("⚠️ meta size != vectors count")

    print("🔹 Loading embedding model")
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    print("🔹 Encoding query")
    q = encode_query(model, query)       # shape: (1, D)

    print(f"🔹 Computing cosine similarity (Top-{k})")
    # Since embeddings are normalized, cosine = dot product
    scores = np.dot(X, q.T).squeeze()    # shape: (N,)
    top_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for rank, idx in enumerate(top_idx, 1):
        m = meta[idx]
        preview = (m.get("text", "") or "").replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:240] + "…"
        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "id": m.get("id"),
            "doc_id": m.get("doc_id"),
            "chunk_idx": m.get("chunk_idx"),
            "text_preview": preview,
        })

    print("✅ Results:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index", help="Directory with vectors.npy and meta.jsonl")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--k", type=int, default=3, help="Top-k results")
    args = parser.parse_args()

    main(args.index, args.query, args.k)
