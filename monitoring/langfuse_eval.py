"""
Langfuse v4 — tracing production RAG.

Variables d'environnement :
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST  ou  LANGFUSE_BASE_URL  (défaut : https://cloud.langfuse.com)

Hiérarchie des spans créés par @observe() :
    run_rag_pipeline  (trace racine)
      ├── run_retrieval   (span)
      └── generate_answer (span generation)
"""
import os
import sys
from typing import List

from dotenv import load_dotenv

load_dotenv()

# Compatibilité LANGFUSE_BASE_URL → LANGFUSE_HOST (v4 lit LANGFUSE_HOST)
if not os.getenv("LANGFUSE_HOST") and os.getenv("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]

from langfuse import observe, get_client, Langfuse, propagate_attributes
from monitoring.llm_as_judge import RAGJudge

from src.core.logging import logging
from src.core.exception import AgenticRagException
from backend.api.dependencies import (get_semantic_search,
    get_keyword_search, get_reranker, get_llm)



# ── Pipeline RAG tracé ────────────────────────────────────────────────────────

@observe()
def run_retrieval(query: str, k: int = 20, top_k: int = 5) -> list:
    """Hybrid retrieval (sémantique + BM25 + RRF) avec span Langfuse."""
    
    try:

        sem_search = get_semantic_search()
        kw_search  = get_keyword_search()
        reranker   = get_reranker()

        sem_results = sem_search.search(query, k=k)
        kw_results  = kw_search.search(query, k=k)
        context     = reranker.fuse(sem_results, kw_results, query=query, top_k=top_k)

        # Trim : on ne logue que les métadonnées, pas le contenu complet des chunks
        context_meta = [
            {
                "source":       c.get("source", "").split("\\")[-1].split("/")[-1],
                "page":         c.get("page_number"),
                "type":         c.get("type", "Text"),
                "rerank_score": round(c["rerank_score"], 5),
            }
            for c in context
        ]

        get_client().update_current_span(
            input={"query": query, "k": k, "top_k": top_k},
            output={
                "semantic_count": len(sem_results),
                "keyword_count": len(kw_results),
                "fused_count": len(context),
                "top_rerank_score": round(context[0]["rerank_score"], 5) if context else 0,
                "chunks": context_meta})
        
        return context

    except Exception as e:
        raise AgenticRagException(e, sys)


@observe()
def generate_answer(query: str, context: List[dict]) -> str:
    """Génération LLM avec span Langfuse (modèle, tokens, latence)."""
    
    try:

        llm = get_llm()
        answer, usage = llm.generate(query=query, context_chunks=context)

        get_client().update_current_generation(
            input=[{"role": "user", "content": query}],
            output=answer,
            model=llm.model,
            usage_details=usage,
            metadata={"context_chunks": len(context),
                       "answer_length": len(answer)})
        return answer

    except Exception as e:
        raise AgenticRagException(e, sys)


@observe(name="rag-query")
def run_rag_pipeline(query: str, k: int = 20, top_k: int = 5, session_id: str | None = None) -> dict:
    """
    Trace racine du pipeline RAG complet.
    Retourne {"answer": str, "context": list}.
    """
   
    try:

        if session_id:
            propagate_attributes(session_id=session_id)

        context = run_retrieval(query, k=k, top_k=top_k)
        answer  = generate_answer(query, context)
        trace_id = get_client().get_current_trace_id()
        result = RAGJudge.evaluate(question=query, 
                                   context_chunks=context, 
                                   answer=answer, 
                                   trace_id=trace_id)
        
        return {"answer": answer, "context": context}

    except AgenticRagException:
        get_client().update_current_span(metadata={"error": True})
        raise


# ── Score humain / automatique ────────────────────────────────────────────────

def score_trace(trace_id: str, 
                name: str, 
                value: float, 
                comment: str | None = None) -> None:
    """Ajoute un score sur une trace existante (feedback utilisateur, métrique auto…).

    Parameters
    ----------
    trace_id : ID retourné par get_client().get_current_trace_id() dans run_rag_pipeline
    name     : ex. "user-feedback", "faithfulness", "relevance"
    value    : flottant normalisé (0.0 à 1.0)
    comment  : annotation textuelle optionnelle
    """
    
    try:

        client = Langfuse()
        client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
        client.flush()
   
    except Exception as exc:
        logging.warning(f"Langfuse score échoué (non bloquant) : {exc}")
