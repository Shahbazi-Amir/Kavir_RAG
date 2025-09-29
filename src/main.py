# main.py — Full overwrite with RAG + /chat integration (CPU-only llama.cpp, cleaned)
# Notes (EN):
# - Keeps existing RAG: /ping, /query, /reindex, /upload
# - Adds /chat with persistent llama.cpp session via pexpect
# - Cleans output so only model’s answer after "Assistant:" is returned

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, json, subprocess, uuid, asyncio, pexpect, re
from pathlib import Path
from datetime import datetime
import requests


import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.loaders import load_any_bytes
from src.chunking import chunk_text_smart, choose_chunk_params

app = FastAPI(title="Local RAG", version="0.4.1")

# ---------- Config ----------
INDEX_PATH = os.environ.get("RAG_FAISS_INDEX", "data/index/faiss.index")
META_PATH  = os.environ.get("RAG_META_PATH",  "data/index/meta.jsonl")
MODEL_NAME = os.environ.get("RAG_EMBED_MODEL","intfloat/multilingual-e5-small")
RAW_DIR    = "data/raw"
CHUNKS_DIR = "data/chunks"
CHUNKS_PATH = f"{CHUNKS_DIR}/chunks.jsonl"
MANIFEST_PATH = f"{CHUNKS_DIR}/manifest.jsonl"

LLAMA_BIN   = os.getenv("LLAMA_BIN", "llama.cpp/build/bin/llama-cli")
MODEL_PATH  = os.getenv("LLAMA_MODEL", "data/models/qwen2-7b-instruct-q4_k_m.gguf")
LLAMA_ARGS = [
    LLAMA_BIN,
    "--model", MODEL_PATH,
    "-c", "768",
    "-n", "48",
    "-t", "4",
    "-b", "128",
    "-ngl", "0",
    "--no-warmup"
]

MAX_CTX_DOC_CHARS = int(os.getenv("CHAT_CTX_MAX_CHARS", "1600"))

# ---------- Globals ----------
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_meta: List[dict] = []
_llama_lock = asyncio.Lock()
_llama_sess: Optional[pexpect.spawnbase.SpawnBase] = None

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
    input_dir: str = Field(default="data/chunks")
    output_dir: str = Field(default="data/index")
    batch: int = Field(default=32, ge=1, le=1024)

class ReindexOut(BaseModel):
    status: str
    input_dir: str
    output_dir: str
    batch: int
    stdout_tail: str

class UploadOut(BaseModel):
    status: str
    doc_id: str
    saved_path: str
    size_bytes: int
    n_chunks: int
    reindexed: bool = False
    reindex_log_tail: Optional[str] = None

class ChatIn(BaseModel):
    prompt: str
    system: Optional[str] = None
    max_new_tokens: Optional[int] = None
    use_rag: bool = True
    rag_k: int = 3
    language: Optional[str] = None

# ---------- Utils ----------
def _load_meta(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _encode_query(text: str) -> np.ndarray:
    q = _model.encode([text], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return q.astype("float32")

def _ensure_dirs() -> None:
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(INDEX_PATH)).mkdir(parents=True, exist_ok=True)

def _slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name or "file")

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _run_reindex(input_dir="data/chunks", output_dir="data/index", batch=32) -> str:
    cmd = [
        "python", "-u", "/app/src/embeddings.py",
        "--input", f"/app/{input_dir}" if not input_dir.startswith("/") else input_dir,
        "--out",   f"/app/{output_dir}" if not output_dir.startswith("/") else output_dir,
        "--batch", str(batch),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout or "") + "\n" + (proc.stderr or ""))
    return proc.stdout.strip()

def _reload_index_and_meta() -> None:
    global _index, _meta
    _index = faiss.read_index(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
    _meta = _load_meta(META_PATH)

# ---------- llama helpers ----------
def _spawn_llama(max_new_tokens: Optional[int] = None):
    global _llama_sess
    args = LLAMA_ARGS[:]
    if max_new_tokens:
        for i,a in enumerate(args):
            if a == "-n" and i+1 < len(args):
                args[i+1] = str(max_new_tokens); break
    if _llama_sess is None or not _llama_sess.isalive():
        _llama_sess = pexpect.spawn(" ".join(args), encoding="utf-8", timeout=120)
        _llama_sess.expect("interactive mode")

def _format_chat(system: Optional[str], user: str, ctx_text: Optional[str], language: Optional[str]) -> str:
    # Qwen2 chat template (from GGUF metadata)
    sysmsg = (system or "You are a helpful assistant.").strip()
    lang_hint = f" Respond in {language}." if language else ""
    ctx_block = f"\nContext:\n{ctx_text}\n" if ctx_text else ""

    return (
        f"<|im_start|>system\n"
        f"{sysmsg}{lang_hint}{ctx_block}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _llama_ask(prompt_text: str, max_new_tokens: int = 48) -> str:
    args = [
        LLAMA_BIN, "--model", MODEL_PATH,
        "-c", "768", "-n", str(max_new_tokens), "-t", "4", "-b", "128",
        "-ngl", "0", "--no-warmup",
        "-p", prompt_text
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli failed:\n{proc.stderr}")
    out = proc.stdout.strip()
    # keep only last non-empty lines (model answer)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[-1] if lines else out







def _build_rag_context(query_text: str, k: int) -> str:
    if _index is None or not _meta:
        return ""
    q = _encode_query(query_text)
    scores, ids = _index.search(q, min(max(1, k), 10))
    ids = ids[0].tolist()
    pieces, budget = [], MAX_CTX_DOC_CHARS
    for idx in ids:
        if idx == -1 or idx >= len(_meta): 
            continue
        t = (_meta[idx].get("text") or "").strip()
        if not t:
            continue
        take = t[: min(len(t), budget)]
        pieces.append(take)
        budget -= len(take)
        if budget <= 0:
            break
    return "\n---\n".join(pieces).strip()

# ---------- Lifecycle ----------
@app.on_event("startup")
def on_startup():
    global _model, _index, _meta
    _ensure_dirs()
    _model = SentenceTransformer(MODEL_NAME)
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
    _meta = _load_meta(META_PATH)

# ---------- Routes ----------
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/query", response_model=SearchOut)
def query(body: QueryIn):
    if _index is None or not _meta:
        raise HTTPException(503, "Index not ready. Reindex first.")
    if not body.query.strip():
        raise HTTPException(400, "Empty query")
    q = _encode_query(body.query)
    scores, ids = _index.search(q, min(max(1, body.k), 20))
    results = []
    for rank, (idx, s) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), 1):
        if idx == -1: continue
        m = _meta[idx] if 0 <= idx < len(_meta) else {}
        text = (m.get("text", "") or "").replace("\n", " ")
        if len(text) > 240: text = text[:240] + "…"
        results.append(Hit(rank=rank, score=float(s),
            id=m.get("id"), doc_id=m.get("doc_id"),
            chunk_idx=m.get("chunk_idx"), text_preview=text))
    return SearchOut(results=results)

@app.post("/reindex", response_model=ReindexOut)
def reindex(body: ReindexIn):
    try:
        out = _run_reindex(body.input_dir, body.output_dir, body.batch)
    except Exception as e:
        raise HTTPException(500, f"Reindex failed:\n{e}")
    _reload_index_and_meta()
    tail = "\n".join(out.strip().splitlines()[-10:])
    return ReindexOut(status="ok", input_dir=body.input_dir, output_dir=body.output_dir, batch=body.batch, stdout_tail=tail)

@app.post("/upload", response_model=UploadOut)
async def upload_any(file: UploadFile = File(...), reindex: bool = Query(False), ocr: bool = Query(False)):
    _ensure_dirs()
    original = _slugify(file.filename or "upload")
    ext = Path(original).suffix.lower()
    if ext not in (".txt", ".pdf", ".docx", ".csv"):
        raise HTTPException(400, "Only .txt, .pdf, .docx, .csv are accepted")
    doc_id = str(uuid.uuid4())
    unique_name = f"{Path(original).stem}__{doc_id}{ext}"
    raw_path = Path(RAW_DIR) / unique_name
    data = await file.read()
    raw_path.write_bytes(data)
    text = load_any_bytes(data, ext=ext.lstrip("."), ocr=ocr)
    if not text.strip():
        raise HTTPException(400, "No extractable text from file")
    size, overlap = choose_chunk_params(ext.lstrip("."), text)
    chunks = chunk_text_smart(text, ext.lstrip("."))
    created_at = _now_iso()
    _append_jsonl(MANIFEST_PATH, {
        "doc_id": doc_id, "path": str(raw_path), "filename": original,
        "ext": ext.lstrip("."), "size_bytes": raw_path.stat().st_size,
        "n_chunks": len(chunks), "created_at": created_at,
        "source_type": ext.lstrip("."), "chunk_size": size, "chunk_overlap": overlap,
    })
    for i,t in enumerate(chunks):
        _append_jsonl(CHUNKS_PATH, {
            "id": f"{doc_id}:{i:04d}", "doc_id": doc_id, "chunk_idx": i,
            "text": t, "path": str(raw_path), "created_at": created_at,
            "source_type": ext.lstrip("."),
        })
    did_reindex, tail = False, None
    if reindex:
        try:
            out = _run_reindex(CHUNKS_DIR, os.path.dirname(INDEX_PATH), 32)
            _reload_index_and_meta(); did_reindex = True
            tail = "\n".join(out.strip().splitlines()[-10:])
        except Exception as e:
            raise HTTPException(500, f"Upload OK but reindex failed:\n{e}")
    return UploadOut(status="ok", doc_id=doc_id, saved_path=str(raw_path),
        size_bytes=raw_path.stat().st_size, n_chunks=len(chunks),
        reindexed=did_reindex, reindex_log_tail=tail)

@app.post("/chat")
async def chat(body: ChatIn):
    if not body.prompt.strip():
        raise HTTPException(400, "Empty prompt")

    ctx = _build_rag_context(body.prompt, body.rag_k) if body.use_rag else ""

    # آماده‌سازی prompt
    prompt = _format_chat(body.system, body.prompt, ctx, body.language)

    try:
        resp = requests.post(
            "http://localhost:8080/completion",
            json={
                "prompt": prompt,
                "n_predict": body.max_new_tokens or 48,
                "temperature": 0.7,
                "stop": ["<|im_end|>", "</s>"]
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("content", "").strip()
        return {"answer": answer, "used_rag": body.use_rag, "rag_chars": len(ctx)}
    except Exception as e:
        raise HTTPException(500, f"chat failed: {e}")

