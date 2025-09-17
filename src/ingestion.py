# src/ingestion.py

import os
import uuid
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Union

from schema_defs import DocumentManifest, TextChunk, sha1_of_text, validate_chunks

# ---------- Paths (robust to where you run from) ----------
ROOT = Path(__file__).resolve().parents[1]   # expected: /app
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"

# ---------- utils ----------
def file_checksum(path: Union[str, Path]) -> str:
    """Return sha1 checksum of file content."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into chunks safely (no infinite loops on short texts)."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if not (0 <= overlap < size):
        raise ValueError("overlap must be >= 0 and < size")

    chunks: List[str] = []
    n = len(text)
    if n == 0:
        return chunks

    step = max(1, size - overlap)  # ensure progress
    start = 0
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks

# ---------- ingestion ----------
def ingest_file(path: Union[str, Path], out_dir: Path = CHUNKS_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fpath = Path(path)
    ext = fpath.suffix.lower().lstrip(".")
    stat = fpath.stat()
    doc_id = str(uuid.uuid4())
    checksum = file_checksum(fpath)

    # load raw text (txt for now; extend later for pdf/docx)
    if ext == "txt":
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        mime = "text/plain"
        page_start = None
        page_end = None
    else:
        raise NotImplementedError(f"Extension {ext} not supported yet")

    # chunking
    size = 800
    overlap = 150
    raw_chunks = chunk_text(text, size=size, overlap=overlap)

    text_chunks: List[TextChunk] = []
    step = max(1, size - overlap)
    for idx, chunk in enumerate(raw_chunks):
        ch_id = f"{doc_id}:{idx:04d}"
        start_char = idx * step
        end_char = start_char + len(chunk)
        text_chunks.append(
            TextChunk(
                id=ch_id,
                doc_id=doc_id,
                chunk_idx=idx,
                text=chunk,
                n_chars=len(chunk),
                start_char=start_char,
                end_char=end_char,
                overlap=overlap,
                source={
                    "path": str(fpath),
                    "page_start": page_start,
                    "page_end": page_end,
                    "mime": mime,
                },
                checksum=sha1_of_text(chunk),
            )
        )

    validate_chunks(text_chunks)

    # write chunks.jsonl
    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("a", encoding="utf-8") as f:
        for ch in text_chunks:
            f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")

    # write manifest.jsonl
    manifest_path = out_dir / "manifest.jsonl"
    manifest = DocumentManifest(
        doc_id=doc_id,
        path=str(fpath),
        ext=ext,
        size_bytes=stat.st_size,
        mtime=datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
        checksum=checksum,
        num_chunks=len(text_chunks),
        pages=None,
        ingested_at=datetime.utcnow().isoformat(),
    )
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest.__dict__, ensure_ascii=False) + "\n")

    print(f"Ingested {fpath} → {len(text_chunks)} chunks")
    print(f"Wrote: {chunks_path} , {manifest_path}")

# ---------- demo run ----------
if __name__ == "__main__":
    demo_file = RAW_DIR / "demo.txt"   # => /app/data/raw/demo.txt
    if demo_file.exists():
        ingest_file(demo_file)
    else:
        print("No demo file found at", demo_file)
