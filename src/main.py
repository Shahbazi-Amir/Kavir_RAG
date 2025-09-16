# src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Local RAG", version="0.1.0")

# ---------- Config ----------
INDEX_PATH = os.environ.get("RAG_FAISS_INDEX", "data/index/faiss.index")
META_PATH  = os.environ.get("RAG_META_PATH",  "data/index/meta.jsonl")
MODEL_NAME = os.environ.get("RAG_EMBED_MODEL","intfloat/multilingual-e5-small")

# ---------- Globals (loaded once) ----------
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_meta: List[dict] = []

# ---------- Schemas ----------
class QueryIn(BaseModel):
    query: str
    k: int = 3

class Hit(BaseModel):
    rank: int
    score: float
    id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_idx: Optional[int] = None
    text_preview: str

class SearchOut(BaseModel):
    results: List[Hit]

# ---------- Utils ----------
def _load_meta(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def _encode_query(text: str) -> np.ndarray:
    # L2-normalized embeddings (cosine via Inner Product)
    q = _model.encode([text], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return q.astype("float32")

# ---------- Lifecycle ----------
@app.on_event("startup")
def on_startup():
    global _model, _index, _meta
    # Load model
    _model = SentenceTransformer(MODEL_NAME)

    # Load index
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"FAISS index not found: {INDEX_PATH}")
    _index = faiss.read_index(INDEX_PATH)

    # Load meta
    if not os.path.exists(META_PATH):
        raise RuntimeError(f"Meta JSONL not found: {META_PATH}")
    _meta = _load_meta(META_PATH)

    if _index.ntotal != len(_meta):
        # Not fatal, but warn
        print(f"[WARN] index vectors ({_index.ntotal}) != meta rows ({len(_meta)})")

    print(f"[READY] model={MODEL_NAME}  vectors={_index.ntotal}")

# ---------- Routes ----------
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/query", response_model=SearchOut)
def query(body: QueryIn):
    if not body.query.strip():
        raise HTTPException(400, "Empty query")

    q = _encode_query(body.query)
    k = max(1, min(body.k, 20))
    scores, ids = _index.search(q, k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[Hit] = []
    for rank, (idx, s) in enumerate(zip(ids, scores), 1):
        if idx == -1:  # no hit
            continue
        m = _meta[idx] if 0 <= idx < len(_meta) else {}
        text = (m.get("text", "") or "").replace("\n", " ")
        if len(text) > 240:
            text = text[:240] + "…"
        results.append(Hit(
            rank=rank,
            score=float(s),
            id=m.get("id"),
            doc_id=m.get("doc_id"),
            chunk_idx=m.get("chunk_idx"),
            text_preview=text
        ))
    return SearchOut(results=results)
