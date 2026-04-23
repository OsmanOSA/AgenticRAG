"""
Dépendances FastAPI partagées entre les routes.
Chaque ressource est instanciée une seule fois (singleton via lru_cache).
"""
from functools import lru_cache
from types import SimpleNamespace

from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.retrieval.semantic_search import SemanticSearch
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.reranker import Reranker
from src.generation.llm_client import LLMClient


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()


@lru_cache(maxsize=1)
def get_semantic_search() -> SemanticSearch:
    return SemanticSearch(get_embedder(), get_store())


@lru_cache(maxsize=1)
def get_keyword_search() -> KeywordSearch:
    store    = get_store()
    payloads = store.scroll_all()
    chunks   = [
        SimpleNamespace(
            content=p["content"],
            doc_id=p["doc_id"],
            type=p["type"],
            metadata={
                "source":      p["source"],
                "page_number": p["page_number"],
                "chunk_index": p["chunk_index"],
            },
        )
        for p in payloads
        if p["content"]
    ]
    ks = KeywordSearch()
    ks.build(chunks=chunks)
    return ks


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
