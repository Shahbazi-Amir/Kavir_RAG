# English comments inside code.

import subprocess
import sys


def run(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    # EDIT THESE
    input_file = "data/raw/sample.pdf"    
    chunks_dir = "data/chunks"
    index_dir = "data/index"
    query = "your question here"

    # 1) Ingest
    run(
        f"python -m src.ingest_stream_cli "
        f"--input \"{input_file}\" "
        f"--out \"{chunks_dir}\""
    )

    # 2) Embedding + Index
    run(
        f"python -m src.embeddings "
        f"--input \"{chunks_dir}\" "
        f"--out \"{index_dir}\""
    )

    # 3) Search
    run(
        f"python -m src.search "
        f"--index \"{index_dir}/faiss.index\" "
        f"--meta \"{index_dir}/meta.jsonl\" "
        f"--query \"{query}\""
    )


if __name__ == "__main__":
    main()
