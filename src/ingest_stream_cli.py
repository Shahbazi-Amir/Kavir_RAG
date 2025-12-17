# English comments inside code.

import argparse
from pathlib import Path

# IMPORTANT: absolute import via src package
from src.ingestion_stream import ingest_stream


def main():
    parser = argparse.ArgumentParser(description="Stream-based ingestion CLI")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (PDF / TXT / MD)"
    )
    parser.add_argument(
        "--out",
        default="data/chunks",
        help="Output directory for chunks"
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR if supported"
    )
    args = parser.parse_args()

    ingest_stream(
        path=Path(args.input),
        out_dir=Path(args.out),
        ocr=args.ocr,
    )


if __name__ == "__main__":
    main()
