# src/rag_cli.py

import argparse
from typing import List, Dict

from src.search_numpy import main as search_main
from src.generate_local import build_prompt, generate_answer


def run_rag(query: str, k: int) -> None:
    # Step 1: run search and capture results
    from src.search_numpy import load_meta
    import numpy as np
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    index_dir = Path("data/index")
    vectors = np.load(index_dir / "vectors.npy")
    meta = load_meta(index_dir / "meta.jsonl")

    model = SentenceTransformer("intfloat/multilingual-e5-small")
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    scores = (vectors @ q_vec.T).squeeze()
    top_idx = scores.argsort()[-k:][::-1]

    contexts: List[Dict] = [meta[i] for i in top_idx]

    # Step 2: build prompt
    prompt = build_prompt(query, contexts)

    # Step 3: generate answer
    answer = generate_answer(prompt)

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    run_rag(args.query, args.k)
