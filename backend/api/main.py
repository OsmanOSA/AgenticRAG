import gc
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.schemas import (
    QueryRequest, QueryResponse, SourceItem,
    StatusResponse, IngestResponse, StatsResponse, DocumentStats,
)
from backend.api.dependencies import (
    get_embedder, get_store, get_semantic_search,
    get_keyword_search, get_reranker, get_llm,
)
from src.core.logging import logging
from src.core.exception import AgenticRagException

app = FastAPI(
    title="NovaRAG API",
    description="Agentic RAG — retrieval hybride + Gemma 4",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/status", response_model=StatusResponse)
def status():
    """État de la collection Qdrant."""
    store = get_store()
    count = store.count()
    return StatusResponse(
        collection=store.collection,
        point_count=count,
        ready=count > 0,
    )


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Répond à une question via le pipeline RAG complet."""

    try:
        
        sem_search = get_semantic_search()
        kw_search  = get_keyword_search()
        reranker   = get_reranker()
        llm        = get_llm()

        sem_results = sem_search.search(req.query, k=req.k)
        kw_results  = kw_search.search(req.query, k=req.k)
        context     = reranker.fuse(sem_results, kw_results, query=req.query, top_k=req.top_k)

        answer  = llm.generate(query=req.query, context_chunks=context)

        sources = [
            SourceItem(
                content=r["content"],
                type=r.get("type", "Text"),
                source=r.get("source"),
                page_number=r.get("page_number"),
                rerank_score=r["rerank_score"],
            )
            for r in context
        ]

        logging.info(f"Query OK : '{req.query[:60]}'")
        return QueryResponse(answer=answer, sources=sources, query=req.query)

    except AgenticRagException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest", response_model=IngestResponse)
def ingest():
    """Déclenche l'ingestion complète des PDFs (bloquant)."""
    try:
        from src.data.data_ingestion import PdfIngestion
        from src.data.chunker import Chunker
        from src.core.utils import typed_chunks_to_documents
        from src.indexing.embedder import Embedder
        from src.indexing.vector_store import VectorStore
        from backend.api.dependencies import (
            get_embedder, get_store, get_semantic_search,
            get_keyword_search, get_reranker,
        )

        all_chunks = PdfIngestion.load_typed(vlm_model="")
        texts      = PdfIngestion.text_chunks(all_chunks)
        tables     = PdfIngestion.table_chunks(all_chunks)
        images     = PdfIngestion.image_chunks(all_chunks)

        # Fallback content pour les images sans description VLM
        for img in images:
            if not img.content.strip():
                page = img.metadata.get("page_number", "?")
                img.content = (
                    f"Image présente page {page} du document {img.doc_id}. "
                    f"Contenu visuel non décrit (VLM désactivé)."
                )

        docs   = typed_chunks_to_documents(texts)
        chunks = Chunker.chunk(documents=docs, strategy="semantic")

        embd = get_embedder()
        chunks = embd.embed(chunks=chunks);  gc.collect()
        tables = embd.embed(chunks=tables);  gc.collect()
        images = embd.embed(chunks=images);  gc.collect()

        store = get_store()
        store.delete_collection()
        store.create_collection()
        store.upsert(chunks + tables + images)

        # Invalide les caches pour forcer la reconstruction au prochain appel
        get_semantic_search.cache_clear()
        get_keyword_search.cache_clear()
        get_reranker.cache_clear()

        total = store.count()
        logging.info(f"Ingestion OK : {total} points indexés ({len(chunks)} texte, {len(tables)} tableaux, {len(images)} images)")
        return IngestResponse(
            status="ok",
            text_chunks=len(chunks),
            table_chunks=len(tables),
            image_chunks=len(images),
            total_indexed=total,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
def stats():
    """Statistiques détaillées par document et par type de chunk."""
    from collections import defaultdict

    store = get_store()
    all_points = store.scroll_all()

    total_chars = 0
    text_chunks = 0
    table_chunks = 0
    image_chunks = 0

    doc_stats: dict = defaultdict(lambda: {
        "text": 0, "table": 0, "image": 0,
        "pages": set(), "chars": 0,
    })

    for p in all_points:
        src   = p.get("source") or "unknown"
        ctype = p.get("type", "Text")
        chars = len(p.get("content", ""))
        total_chars += chars
        doc_stats[src]["chars"] += chars

        if p.get("page_number") is not None:
            doc_stats[src]["pages"].add(p["page_number"])

        if ctype == "Table":
            table_chunks += 1
            doc_stats[src]["table"] += 1
        elif ctype == "Image":
            image_chunks += 1
            doc_stats[src]["image"] += 1
        else:
            text_chunks += 1
            doc_stats[src]["text"] += 1

    documents = [
        DocumentStats(
            filename=src.split("\\")[-1].split("/")[-1],
            path=src,
            text_chunks=v["text"],
            table_chunks=v["table"],
            image_chunks=v["image"],
            page_count=len(v["pages"]),
            estimated_tokens=v["chars"] // 4,
        )
        for src, v in sorted(doc_stats.items())
    ]

    return StatsResponse(
        total_chunks=len(all_points),
        text_chunks=text_chunks,
        table_chunks=table_chunks,
        image_chunks=image_chunks,
        document_count=len(doc_stats),
        total_chars=total_chars,
        estimated_tokens=total_chars // 4,
        documents=documents,
    )


@app.get("/api/health")
def health():
    return {"status": "ok"}
