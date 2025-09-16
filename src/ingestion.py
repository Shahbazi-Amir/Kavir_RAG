"""
ingestion.py
-------------
Module for loading documents (PDF, DOCX, TXT, CSV) and chunking text.
"""

import os
import chardet
import pandas as pd
import fitz  # pymupdf
import docx
from typing import List, Dict


# ---------- Loaders ----------

def load_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return "\n".join(text)


def load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def load_txt(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
        enc = chardet.detect(raw)["encoding"]
    return raw.decode(enc or "utf-8")


def load_csv(path: str, sep: str = ",") -> str:
    df = pd.read_csv(path, sep=sep)
    return df.to_csv(index=False)


def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext == ".txt":
        return load_txt(path)
    elif ext == ".csv":
        return load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------- Chunker ----------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append({"text": chunk, "start": start, "end": end})
        start += chunk_size - overlap

    return chunks


# ---------- Example Runner ----------

if __name__ == "__main__":
    sample_path = "data/sample.txt"  # change this for testing
    raw_text = load_file(sample_path)
    chunks = chunk_text(raw_text)

    print(f"Loaded {len(raw_text)} characters")
    print(f"Created {len(chunks)} chunks")
    print(chunks[0])
