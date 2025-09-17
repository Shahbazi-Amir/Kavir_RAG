# src/schema_defs.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import hashlib

# ---------- Document Manifest ----------
@dataclass
class DocumentManifest:
    doc_id: str              # unique id for the document (uuid or hash)
    path: str                # file path
    ext: str                 # file extension, e.g., pdf, docx
    size_bytes: int          # file size in bytes
    mtime: str               # last modified time (ISO string)
    checksum: str            # sha1 checksum of file
    num_chunks: int          # number of chunks generated
    pages: Optional[int]     # number of pages if applicable (pdf/docx)
    ingested_at: str         # ingestion timestamp (ISO string)

# ---------- Text Chunk ----------
@dataclass
class TextChunk:
    id: str                  # unique chunk id: doc_id:chunk_idx
    doc_id: str              # reference to parent document
    chunk_idx: int           # sequential index
    text: str                # chunk text
    n_chars: int             # number of characters in chunk
    start_char: int          # start offset in document
    end_char: int            # end offset in document
    overlap: int             # overlap with previous chunk
    source: Dict[str, Any]   # metadata (page_start, page_end, path, mime)
    checksum: str            # checksum of chunk text

# ---------- Utility Functions ----------
def sha1_of_text(text: str) -> str:
    """Return sha1 checksum of given text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def validate_chunks(chunks: List[TextChunk]) -> None:
    """Run basic validations on a list of chunks."""
    seen_ids = set()
    for ch in chunks:
        assert ch.id not in seen_ids, f"Duplicate chunk id found: {ch.id}"
        seen_ids.add(ch.id)
        assert len(ch.text) > 0, f"Empty text in chunk {ch.id}"
        assert ch.start_char < ch.end_char, f"Invalid offsets in chunk {ch.id}"
    print(f"Validated {len(chunks)} chunks successfully.")

# ---------- Demo (if run directly) ----------
if __name__ == "__main__":
    # Example chunk just for testing schema
    demo_chunk = TextChunk(
        id="doc123:0001",
        doc_id="doc123",
        chunk_idx=1,
        text="This is a test chunk.",
        n_chars=22,
        start_char=0,
        end_char=22,
        overlap=0,
        source={"path": "data/raw/demo.txt", "page_start": None, "page_end": None, "mime": "text/plain"},
        checksum=sha1_of_text("This is a test chunk.")
    )
    validate_chunks([demo_chunk])
    print("Demo run finished.")
