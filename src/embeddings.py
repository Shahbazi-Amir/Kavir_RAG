"""
Embedding + Indexing pipeline for RAG (local).
Reads chunked JSONL files, generates embeddings with multilingual-e5-small,
normalizes vectors, builds FAISS index, and saves index + metadata.
"""

import os
import json
import argparse
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_chunks(input_dir):
    """Yield chunk records from all JSONL files in input_dir."""
    for fname in os.listdir(input_dir):
        if fname.endswith(".jsonl"):
            with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)


def embed_chunks(model, texts, batch_size=32):
    """Encode list of texts into embeddings with batching."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    return np.vstack(embeddings)


def main(input_dir, output_dir, batch_size=32):
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("🔹 Loading embedding model...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    # Load chunks
    print("🔹 Reading chunks...")
    chunks = list(load_chunks(input_dir))
    texts = [c["text"] for c in chunks]

    print(f"Loaded {len(chunks)} chunks")

    # Compute embeddings
    print("🔹 Computing embeddings...")
    X = embed_chunks(model, texts, batch_size=batch_size)

    # Build FAISS index (cosine similarity via Inner Product)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    # Save index
    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))

    # Save metadata
    with open(os.path.join(output_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("✅ Done!")
    print(f"Index size: {index.ntotal} vectors of dim {dim}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/chunks", help="Path to chunked JSONL files")
    parser.add_argument("--out", default="data/index", help="Output folder for FAISS index and metadata")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for embeddings")
    args = parser.parse_args()

    main(args.input, args.out, args.batch)
