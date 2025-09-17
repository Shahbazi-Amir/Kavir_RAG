"""
ingestion.py
-------------
Load documents (PDF, DOCX, TXT, CSV) and chunk text into overlapping windows.
CLI usage:
    python src/ingestion.py --path data/raw --out data/chunks --chunk-size 800 --overlap 100
"""

import os
import sys
import json
import argparse
import hashlib
from typing import List, Dict, Tuple

import pandas as pd
import fitz  # pymupdf
import docx


# -------------------- Utilities --------------------

ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".csv"}

def _file_meta(path: str) -> Dict:
    """Return basic file metadata."""
    st = os.stat(path)
    return {
        "source_path": os.path.abspath(path),
        "source_name": os.path.basename(path),
        "size_bytes": st.st_size,
    }

def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


# -------------------- Loaders --------------------

def load_pdf(path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    texts = []
    with fitz.open(path) as doc:
        for page in doc:
            texts.append(page.get_text("text"))
    return "\n".join(texts)

def load_docx(path: str) -> str:
    """Extract text from a DOCX using python-docx."""
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

def load_txt(path: str) -> str:
    """Read TXT with reasonable encoding fallbacks (no chardet dependency)."""
    # Try utf-8 first; if it fails, fallback to common encodings.
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue
    # Last-resort: ignore errors
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_csv(path: str) -> str:
    """
    Read CSV into DataFrame and serialize back to CSV text.
    Using sep=None with engine='python' to auto-detect delimiter.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    return df.to_csv(index=False)

def load_file(path: str) -> Tuple[str, Dict]:
    """Load file by extension and return (text, metadata)."""
    meta = _file_meta(path)
    ext = _ext(path)
    if ext == ".pdf":
        text = load_pdf(path)
        meta["filetype"] = "pdf"
    elif ext == ".docx":
        text = load_docx(path)
        meta["filetype"] = "docx"
    elif ext == ".txt":
        text = load_txt(path)
        meta["filetype"] = "txt"
    elif ext == ".csv":
        text = load_csv(path)
        meta["filetype"] = "csv"
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    meta["text_hash"] = _hash_text(text)
    meta["num_chars"] = len(text)
    return text, meta


# -------------------- Chunker --------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
    """Split text into overlapping character windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    n = len(text)
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append({
            "text": chunk,
            "start": start,
            "end": end,
            "index": idx
        })
        idx += 1
        start += step

    return chunks


# -------------------- Persist --------------------

def save_chunks_jsonl(chunks: List[Dict], meta: Dict, out_dir: str) -> str:
    """
    Save chunks as JSONL with metadata per line.
    Returns the output file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(meta["source_name"])[0]
    # unique name based on content hash to avoid collisions
    out_path = os.path.join(out_dir, f"{base}__{meta['text_hash'][:12]}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            row = {
                "text": c["text"],
                "chunk_index": c["index"],
                "char_range": [c["start"], c["end"]],
                "metadata": meta,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_path


# -------------------- Runner --------------------

def iter_input_paths(path: str):
    """Yield file paths from a single file or recursively from a directory."""
    if os.path.isfile(path):
        yield path
    else:
        for root, _, files in os.walk(path):
            for name in files:
                p = os.path.join(root, name)
                if _ext(p) in ALLOWED_EXTS:
                    yield p

def main():
    parser = argparse.ArgumentParser(description="Ingest and chunk files.")
    parser.add_argument("--path", required=True, help="File or directory to ingest (e.g., data/raw or data/raw/sample.txt)")
    parser.add_argument("--out", default="data/chunks", help="Output directory for JSONL chunks")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size (characters)")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap (characters)")
    parser.add_argument("--preview", action="store_true", help="Print a short preview for sanity check")
    args = parser.parse_args()

    total_files = 0
    total_chunks = 0

    for fp in iter_input_paths(args.path):
        total_files += 1
        text, meta = load_file(fp)
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        out_file = save_chunks_jsonl(chunks, meta, args.out)
        total_chunks += len(chunks)

        print(f"✅ Ingested: {fp}")
        print(f"   chars={meta['num_chars']}  chunks={len(chunks)}  -> {out_file}")
        if args.preview and chunks:
            preview = chunks[0]["text"][:200].replace("\n", " ")
            print(f"   preview: {preview} ...")

    print(f"\n🏁 Done. files={total_files}, chunks={total_chunks}, out_dir={args.out}")

if __name__ == "__main__":
    main()
