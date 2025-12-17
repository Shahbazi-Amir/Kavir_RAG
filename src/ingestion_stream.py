# English comments inside code.
from pathlib import Path
from typing import Iterator, Union

def stream_bytes(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield file bytes in fixed-size batches."""
    pass

def extract_text_stream(path: Union[str, Path], ext: str, ocr: bool = False) -> Iterator[str]:
    """Yield text segments without loading full file."""
    pass

def chunk_text_stream(text_iter: Iterator[str], size: int, overlap: int):
    """Yield chunks incrementally with overlap."""
    pass

def ingest_stream(path: Union[str, Path], out_dir: Path):
    """High-level streaming ingestion pipeline."""
    pass
