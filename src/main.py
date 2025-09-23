# src/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os, json, subprocess
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

class ReindexIn(BaseModel):
    input_dir: str = Field(default="data/chunks", description="Path to chunked JSONL dir")
    output_dir: str = Field(default="data/index", description="Output dir for FAISS+meta")
    batch: int = Field(default=32, ge=1, le=1024, description="Batch size for embeddings")

class ReindexOut(BaseModel):
    status: str
    input_dir: str
    output_dir: str
    batch: int
    stdout_tail: str

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

@app.post("/reindex", response_model=ReindexOut)
def reindex(body: ReindexIn):
    # Run embeddings script as a subprocess inside this container
    # embeddings.py خودش قبل از نوشتن، بکاپ می‌سازد.
    cmd = [
        "python", "-u", "/app/src/embeddings.py",
        "--input", f"/app/{body.input_dir}" if not body.input_dir.startswith("/") else body.input_dir,
        "--out",   f"/app/{body.output_dir}" if not body.output_dir.startswith("/") else body.output_dir,
        "--batch", str(body.batch),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        err = (e.stdout or "") + "\n" + (e.stderr or "")
        raise HTTPException(500, f"Reindex failed:\n{err.strip()}")

    # Reload index + meta into memory so /query بلافاصله با ایندکس جدید کار کند
    global _index, _meta
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
    if os.path.exists(META_PATH):
        _meta = _load_meta(META_PATH)

    out_text = proc.stdout.strip().splitlines()
    tail = "\n".join(out_text[-10:]) if out_text else ""  # آخرین ۱۰ خط لاگ

    return ReindexOut(
        status="ok",
        input_dir=body.input_dir,
        output_dir=body.output_dir,
        batch=body.batch,
        stdout_tail=tail,
    )
