# src/search.py
"""
Simple retrieval over FAISS index using multilingual-e5-small.
- Loads faiss.index and meta.jsonl
- Encodes a query (L2-normalized) and searches Top-k by cosine similarity (via Inner Product)
"""

import os
import json
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_meta(meta_path):
    """Load metadata JSONL into a list aligned with vectors order."""
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def ensure_exists(path, desc):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} not found at: {path}")


def encode_query(model, text: str) -> np.ndarray:
    """Encode and L2-normalize a single query string to shape (1, dim)."""
    q = model.encode([text], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return q.astype("float32")


def main(index_path, meta_path, query, k):
    ensure_exists(index_path, "FAISS index")
    ensure_exists(meta_path, "Metadata JSONL")

    print("🔹 Loading FAISS & metadata...")
    index = faiss.read_index(index_path)
    meta = load_meta(meta_path)
    if len(meta) != index.ntotal:
        print(f"⚠️ meta count ({len(meta)}) != index vectors ({index.ntotal})")

    print("🔹 Loading embedding model...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    print("🔹 Encoding query...")
    q = encode_query(model, query)

    print(f"🔹 Searching top-{k} ...")
    scores, ids = index.search(q, k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results = []
    for rank, (idx, s) in enumerate(zip(ids, scores), 1):
        if idx == -1:
            continue
        m = meta[idx]
        # keep preview short
        text_preview = (m.get("text", "") or "").replace("\n", " ")
        if len(text_preview) > 240:
            text_preview = text_preview[:240] + "…"
        results.append({
            "rank": rank,
            "score": float(s),
            "id": m.get("id"),
            "doc_id": m.get("doc_id"),
            "chunk_idx": m.get("chunk_idx"),
            "text_preview": text_preview
        })

    print("✅ Results:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index/faiss.index", help="Path to FAISS index")
    parser.add_argument("--meta",  default="data/index/meta.jsonl",   help="Path to metadata JSONL")
    parser.add_argument("--k", type=int, default=3,                   help="Top-k results")
    parser.add_argument("--query", required=True,                      help="Search query text")
    args = parser.parse_args()
    main(args.index, args.meta, args.query, args.k)
