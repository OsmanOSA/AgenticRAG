"""
ingest.py — Ingestion one-shot : extraction → chunking → embedding → Qdrant.
À lancer une seule fois (ou après modification des PDFs).

    python ingest.py
"""
import gc
from pathlib import Path
from typing import List

from src.data.data_ingestion import PdfIngestion
from src.data.chunker import Chunker
from src.core.utils import typed_chunks_to_documents
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.entity.artifact_entity import TableChunk


def main() -> None:

    # ── 1. Extraction ────────────────────────────────────────────────────
    print("\n=== 1. EXTRACTION ===")
    all_chunks = PdfIngestion.load_typed(vlm_model="")

    texts  = PdfIngestion.text_chunks(all_chunks)
    tables = PdfIngestion.table_chunks(all_chunks)

    print(f"  Textes   : {len(texts)}")
    print(f"  Tableaux : {len(tables)}")
    print(f"  Images   : {len(PdfIngestion.image_chunks(all_chunks))}")

   # ── 2. Chunking sémantique ───────────────────────────────────────────
    print("\n=== 2. CHUNKING ===")
    docs   = typed_chunks_to_documents(texts)
    chunks = Chunker.chunk(documents=docs, strategy="semantic")
    print(f"  Chunks sémantiques : {len(chunks)}")

    # ── 3. Embedding ─────────────────────────────────────────────────────
    print("\n=== 3. EMBEDDING ===")
    embd = Embedder()

    print(f"  Textes  ({len(chunks)} chunks)…")
    chunks = embd.embed(chunks=chunks)
    gc.collect()

    print(f"  Tableaux ({len(tables)} chunks)…")
    tables = embd.embed(chunks=tables)
    gc.collect()

    # ── 4. Indexation Qdrant ─────────────────────────────────────────────
    print("\n=== 4. INDEXATION QDRANT ===")
    store = VectorStore()
    store.delete_collection()
    store.create_collection()
    store.upsert(chunks + tables)
    print(f"  {store.count()} points indexés dans '{store.collection}'")
    print("\nIngestion terminée. Lance app.py pour interroger le RAG.")


if __name__ == "__main__":
    main()
