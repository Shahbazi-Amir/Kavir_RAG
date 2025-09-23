# English comments inside code.

import re
from typing import List, Tuple

_SENT_END = re.compile(r"([.!؟?]+)(\s+)")

def choose_chunk_params(ext: str, text: str) -> Tuple[int, int]:
    ext = (ext or "").lstrip(".").lower()
    n = len(text)
    punct_ratio = (text.count(".") + text.count("؟") + text.count("?")) / max(1, n)

    if ext in ("pdf", "docx"):
        size, overlap = 600, 150
        if n > 100_000:  # very long
            size, overlap = 900, 150
    elif ext in ("csv",):
        size, overlap = 700, 120
    else:  # txt / default
        size, overlap = 800, 200

    # If sentence density is very low (tables/lists), use smaller overlap
    if punct_ratio < 0.002:
        overlap = max(80, int(overlap * 0.6))
    return size, overlap

def _hard_chunks(text: str, size: int, overlap: int) -> List[str]:
    chunks, n = [], len(text)
    if n == 0:
        return chunks
    step = max(1, size - overlap)
    s = 0
    while s < n:
        e = min(n, s + size)
        chunk = text[s:e]
        # try to end on sentence boundary within the last 80 chars
        tail = chunk[-80:]
        m = list(_SENT_END.finditer(tail))
        if m:
            cut = e - (80 - m[-1].end())
            chunk = text[s:cut]
            e = cut
        chunks.append(chunk)
        if e >= n:
            break
        s = e - overlap
        if s < 0:
            s = 0
    return chunks

def chunk_text_smart(text: str, ext: str) -> List[str]:
    size, overlap = choose_chunk_params(ext, text)
    return _hard_chunks(text, size, overlap)
