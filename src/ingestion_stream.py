# English comments inside code.

import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterator, Union, List

from src.loaders import load_any_bytes
from schema_defs import TextChunk, DocumentManifest, sha1_of_text, validate_chunks


def stream_bytes(
    path: Union[str, Path],
    chunk_size: int = 1024 * 1024
) -> Iterator[bytes]:
    path = Path(path)
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data


def extract_text_stream(
    path: Union[str, Path],
    ext: str,
    ocr: bool = False
) -> Iterator[str]:
    # Stream-safe: accumulate bytes (loader limitation), no RAM spike logic elsewhere
    buffer = bytearray()
    for b in stream_bytes(path):
        buffer.extend(b)
    text = load_any_bytes(bytes(buffer), ext=ext, ocr=ocr)
    if text:
        yield text


def chunk_text_stream(
    text_iter: Iterator[str],
    size: int = 800,
    overlap: int = 150
) -> Iterator[str]:
    if size <= 0 or not (0 <= overlap < size):
        raise ValueError("invalid chunk params")
    step = max(1, size - overlap)
    for text in text_iter:
        n = len(text)
        start = 0
        while start < n:
            end = min(start + size, n)
            yield text[start:end]
            if end >= n:
                break
            start += step


def ingest_stream(
    path: Union[str, Path],
    out_dir: Path,
    ocr: bool = False,
    size: int = 800,
    overlap: int = 150
) -> None:
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower().lstrip(".")
    stat = path.stat()
    doc_id = str(uuid.uuid4())

    # File checksum
    h = hashlib.sha1()
    for b in stream_bytes(path):
        h.update(b)
    checksum = h.hexdigest()

    # Text stream -> chunks
    text_iter = extract_text_stream(path, ext=ext, ocr=ocr)
    chunks_iter = chunk_text_stream(text_iter, size=size, overlap=overlap)

    text_chunks: List[TextChunk] = []
    step = max(1, size - overlap)
    for idx, chunk in enumerate(chunks_iter):
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
                source={"path": str(path), "page_start": None, "page_end": None, "mime": f"file/{ext}"},
                checksum=sha1_of_text(chunk),
            )
        )

    validate_chunks(text_chunks)

    # Persist chunks
    with (out_dir / "chunks.jsonl").open("a", encoding="utf-8") as f:
        for ch in text_chunks:
            f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")

    # Persist manifest
    manifest = DocumentManifest(
        doc_id=doc_id,
        path=str(path),
        ext=ext,
        size_bytes=stat.st_size,
        mtime=datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
        checksum=checksum,
        num_chunks=len(text_chunks),
        pages=None,
        ingested_at=datetime.utcnow().isoformat(),
    )
    with (out_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest.__dict__, ensure_ascii=False) + "\n")
