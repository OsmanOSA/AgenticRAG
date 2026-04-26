import sys

from typing import List

from src.core.logging import logging
from src.core.exception import AgenticRagException


class Reranker:
    """Fusion sémantique + BM25 par Reciprocal Rank Fusion (RRF).

    RRF combine les rangs des deux listes sans biais de format :
    textes narratifs et tableaux markdown sont traités de façon équitable.
    score_RRF = Σ  1 / (k + rang_i),  k=60 (constante standard).
    """

    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf
        logging.info(f"Reranker RRF initialisé (k={k_rrf})")

    def fuse(
        self,
        semantic_results: List[dict],
        keyword_results: List[dict],
        query: str,
        top_k: int = 10,
    ) -> List[dict]:
        """Fusionne et re-classe par RRF.

        Parameters
        ----------
        semantic_results : List[dict]
            Résultats SemanticSearch triés par score décroissant.
        keyword_results : List[dict]
            Résultats KeywordSearch triés par score décroissant.
        query : str
            Non utilisé par RRF, conservé pour compatibilité d'interface.
        top_k : int
            Nombre de résultats finaux.

        Returns
        -------
        List[dict]
            Résultats fusionnés avec ``rerank_score`` (score RRF).
        """
        try:
            rrf_scores: dict[str, float] = {}
            registry:   dict[str, dict]  = {}

            for rank, result in enumerate(semantic_results):
                key = self._key(result)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.k_rrf + rank + 1)
                registry.setdefault(key, result)

            for rank, result in enumerate(keyword_results):
                key = self._key(result)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.k_rrf + rank + 1)
                registry.setdefault(key, result)

            ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            results = [
                {
                    "content":      registry[key]["content"],
                    "rerank_score": round(score, 6),
                    "doc_id":       registry[key].get("doc_id"),
                    "source":       registry[key].get("source"),
                    "page_number":  registry[key].get("page_number"),
                    "chunk_index":  registry[key].get("chunk_index"),
                    "type":         registry[key].get("type"),
                }
                for key, score in ranked[:top_k]
            ]

            n_tables = sum(1 for r in results if r.get("type") == "Table")
            logging.info(
                f"Reranker RRF : {len(results)} résultat(s) "
                f"({n_tables} table(s)) depuis {len(rrf_scores)} candidat(s)"
            )
            return results

        except Exception as e:
            raise AgenticRagException(e, sys)

    @staticmethod
    def _key(result: dict) -> str:
        
        doc_id      = result.get("doc_id", "")
        chunk_index = result.get("chunk_index")
        
        if chunk_index is not None:
            return f"{doc_id}::{chunk_index}"
        
        return result.get("content", "")[:120]
