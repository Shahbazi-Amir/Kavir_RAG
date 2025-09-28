# main.py — Full overwrite with RAG + /chat integration (CPU-only llama.cpp)
# Notes (EN):
# - Keeps your existing RAG: /ping, /query, /reindex, /upload
# - Adds /chat with persistent llama.cpp session via pexpect
# - Optional RAG context in /chat using top-k chunks (configurable)
# - Tuned for MacBook Pro 2015: -ngl 0, no warmup, short outputs by default

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, json, subprocess, uuid, asyncio, pexpect
from pathlib import Path
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.loaders import load_any_bytes
from src.chunking import chunk_text_smart, choose_chunk_params  # NEW

app = FastAPI(title="Local RAG", version="0.4.0")

# ---------- Config ----------
INDEX_PATH = os.environ.get("RAG_FAISS_INDEX", "data/index/faiss.index")
META_PATH  = os.environ.get("RAG_META_PATH",  "data/index/meta.jsonl")
MODEL_NAME = os.environ.get("RAG_EMBED_MODEL","intfloat/multilingual-e5-small")
RAW_DIR    = "data/raw"
CHUNKS_DIR = "data/chunks"
CHUNKS_PATH = f"{CHUNKS_DIR}/chunks.jsonl"
MANIFEST_PATH = f"{CHUNKS_DIR}/manifest.jsonl"

# llama.cpp config (CPU-only for MBP2015)
LLAMA_BIN   = os.getenv("LLAMA_BIN", "llama.cpp/build/bin/llama-cli")
MODEL_PATH  = os.getenv("LLAMA_MODEL", "data/models/qwen2-7b-instruct-q4_k_m.gguf")
LLAMA_ARGS  = [
    LLAMA_BIN, "--model", MODEL_PATH,
    "-c", "768", "-n", "48", "-t", "4", "-b", "128",
    "-ngl", "0", "--no-warmup", "--interactive-first", "--color", "never"
]
MAX_CTX_DOC_CHARS = int(os.getenv("CHAT_CTX_MAX_CHARS", "1600"))  # cap context size

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
    use_rag: bool = True          # attach RAG context
    rag_k: int = 3                # how many chunks to attach
    language: Optional[str] = None  # force response language if desired (e.g., "fa")

# ---------- Utils ----------
def _load_meta(path: str) -> List[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _encode_query(text: str) -> np.ndarray:
    q = _model.encode([text], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return q.astype("float32")

def _ensure_dirs() -> None:
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(INDEX_PATH)).mkdir(parents=True, exist_ok=True)

def _slugify(name: str) -> str:
    keep = []
    for ch in name or "":
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    s = "".join(keep) or "file"
    while s.startswith("."):
        s = "_" + s[1:]
    return s

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _run_reindex(input_dir: str = "data/chunks", output_dir: str = "data/index", batch: int = 32) -> str:
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
    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)
    else:
        _index = None
    _meta = _load_meta(META_PATH)

# llama helpers
def _spawn_llama(max_new_tokens: Optional[int] = None):
    global _llama_sess
    args = LLAMA_ARGS[:]
    if max_new_tokens:
        # override -n on demand
        for i,a in enumerate(args):
            if a == "-n" and i+1 < len(args):
                args[i+1] = str(max_new_tokens); break
    if _llama_sess is None or not _llama_sess.isalive():
        _llama_sess = pexpect.spawn(" ".join(args), encoding="utf-8", timeout=120)
        _llama_sess.expect("interactive mode")

def _format_chat(system: Optional[str], user: str, ctx_text: Optional[str], language: Optional[str]) -> str:
    # Minimal template: system + optional RAG context + user instruction
    today = datetime.utcnow().date().isoformat()
    sysmsg = (system or f"You are concise. Keep answers short. Date: {today}").strip()
    lang_hint = f" Respond in {language}." if language else ""
    ctx_block = f"\n\n<<CONTEXT>>\n{ctx_text}\n<</CONTEXT>>" if ctx_text else ""
    return f"<<SYS>>\n{sysmsg}{lang_hint}\n<</SYS>>{ctx_block}\n\nUser: {user}\nAssistant:"

def _llama_ask(prompt_text: str) -> str:
    _llama_sess.sendline(prompt_text)
    _llama_sess.expect("Assistant:")
    # Read until model yields control; simple heuristics for pause
    _llama_sess.expect(["\nUser:", "interactive mode", pexpect.TIMEOUT, pexpect.EOF], timeout=30)
    return (_llama_sess.before or "").strip()

def _build_rag_context(query_text: str, k: int) -> str:
    if _index is None or not _meta:
        return ""
    q = _encode_query(query_text)
    k = max(1, min(k, 10))
    scores, ids = _index.search(q, k)
    ids = ids[0].tolist()
    pieces = []
    budget = MAX_CTX_DOC_CHARS
    for idx in ids:
        if idx == -1: 
            continue
        if 0 <= idx < len(_meta):
            t = (_meta[idx].get("text") or "").strip()
            if not t:
                continue
            # append with simple budget cap
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
    k = max(1, min(body.k, 20))
    scores, ids = _index.search(q, k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[Hit] = []
    for rank, (idx, s) in enumerate(zip(ids, scores), 1):
        if idx == -1:
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
    try:
        out = _run_reindex(body.input_dir, body.output_dir, body.batch)
    except Exception as e:
        raise HTTPException(500, f"Reindex failed:\n{e}")
    _reload_index_and_meta()
    tail = "\n".join(out.strip().splitlines()[-10:])
    return ReindexOut(status="ok", input_dir=body.input_dir, output_dir=body.output_dir, batch=body.batch, stdout_tail=tail)

# ------------ /upload (txt/pdf/docx/csv) -------------
@app.post("/upload", response_model=UploadOut)
async def upload_any(
    file: UploadFile = File(...),
    reindex: bool = Query(default=False),
    ocr: bool = Query(default=False, description="Use OCR fallback for PDFs"),
):
    _ensure_dirs()

    original = _slugify(file.filename or "upload")
    ext = Path(original).suffix.lower()
    if ext not in (".txt", ".pdf", ".docx", ".csv"):
        raise HTTPException(400, "Only .txt, .pdf, .docx, .csv are accepted")

    doc_id = str(uuid.uuid4())
    unique_name = f"{Path(original).stem}__{doc_id}{ext}"
    raw_path = Path(RAW_DIR) / unique_name
    data = await file.read()
    with open(raw_path, "wb") as f:
        f.write(data)
    size_bytes = raw_path.stat().st_size

    text = load_any_bytes(data, ext=ext.lstrip("."), ocr=ocr)
    if not text.strip():
        raise HTTPException(400, "No extractable text from file")

    size, overlap = choose_chunk_params(ext.lstrip("."), text)
    chunks = chunk_text_smart(text, ext.lstrip("."))

    created_at = _now_iso()

    manifest_row = {
        "doc_id": doc_id,
        "path": str(raw_path),
        "filename": original,
        "ext": ext.lstrip("."),
        "size_bytes": size_bytes,
        "n_chunks": len(chunks),
        "created_at": created_at,
        "source_type": ext.lstrip("."),
        "chunk_size": size,
        "chunk_overlap": overlap,
    }
    _append_jsonl(MANIFEST_PATH, manifest_row)

    for i, t in enumerate(chunks):
        row = {
            "id": f"{doc_id}:{i:04d}",
            "doc_id": doc_id,
            "chunk_idx": i,
            "text": t,
            "path": str(raw_path),
            "created_at": created_at,
            "source_type": ext.lstrip("."),
        }
        _append_jsonl(CHUNKS_PATH, row)

    did_reindex = False
    tail = None
    if reindex:
        try:
            out = _run_reindex(input_dir=CHUNKS_DIR, output_dir=os.path.dirname(INDEX_PATH), batch=32)
            _reload_index_and_meta()
            did_reindex = True
            tail = "\n".join(out.strip().splitlines()[-10:])
        except Exception as e:
            raise HTTPException(500, f"Upload OK but reindex failed:\n{e}")

    return UploadOut(
        status="ok",
        doc_id=doc_id,
        saved_path=str(raw_path),
        size_bytes=size_bytes,
        n_chunks=len(chunks),
        reindexed=did_reindex,
        reindex_log_tail=tail,
    )

# ------------ NEW: /chat -------------
@app.post("/chat")
async def chat(body: ChatIn):
    if not body.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    # Optional RAG context
    ctx = _build_rag_context(body.prompt, body.rag_k) if body.use_rag else ""
    async with _llama_lock:
        try:
            _spawn_llama(body.max_new_tokens)
            prompt = _format_chat(body.system, body.prompt, ctx, body.language)
            answer = _llama_ask(prompt)
            return {"answer": answer, "used_rag": body.use_rag, "rag_chars": len(ctx)}
        except Exception as e:
            raise HTTPException(500, f"chat failed: {e}")
