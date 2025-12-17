## RAG Pipeline Overview

1. Ingestion
   - Input: PDF / TXT / DOCX
   - Output: data/chunks/chunks.jsonl
   - Responsibility:
     - Read files
     - Extract text
     - Chunk text with overlap
     - Persist chunk metadata

2. Embedding + Index (Current: NumPy backend)
   - Input: chunk JSONL files
   - Output:
     - data/index/vectors.npy
     - data/index/meta.jsonl
   - Responsibility:
     - Convert text chunks to dense vectors
     - Normalize embeddings
     - Store vectors and metadata
   - Notes:
     - CPU-safe
     - No FAISS dependency
     - Designed to be replaceable by FAISS/HNSW

3. Retrieval (Next step)
   - Load vectors.npy
   - Encode query
   - Cosine similarity via NumPy
   - Top-k selection

4. Generation (Later)
   - Retrieved chunks → prompt
   - Local LLM
