import sys

from typing import List
from rank_bm25 import BM25Okapi

from src.core.logging import logging
from src.core.exception import AgenticRagException
from src.entity.artifact_entity import TextChunk, TableChunk


class KeywordSearch:
    """Recherche par mots-clés BM25 sur TextChunks et TableChunks.

    L'index est construit en mémoire à l'initialisation.
    Appeler ``build()`` après chaque mise à jour des chunks.
    """

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: List[TextChunk | TableChunk] = []

    def build(self, 
              chunks: List[TextChunk | TableChunk]
              ) -> None:
        """Construit l'index BM25 à partir des chunks.

        Parameters
        ----------
        chunks : List[TextChunk | TableChunk]
            Chunks à indexer — texte et tableaux acceptés.
        """
        try:

            self._chunks = chunks
            tokenized = [c.content.lower().split() for c in chunks]
            self._index = BM25Okapi(tokenized)
            logging.info(f"BM25 : index construit sur {len(chunks)} chunk(s)")
        
        except Exception as e:
            raise AgenticRagException(e, sys)

    def search(self, 
               query: str, 
               k: int = 5) -> List[dict]:
        """Recherche les k chunks les plus pertinents par BM25.

        Parameters
        ----------
        query : str
            Requête utilisateur.
        k : int, optional
            Nombre de résultats (défaut : 5).

        Returns
        -------
        List[dict]
            Résultats triés par score décroissant avec
            ``content``, ``score``, ``doc_id``, ``source``, ``page_number``.
        """
        try:
            
            if self._index is None:
                raise ValueError("Index BM25 non construit — appelez build() d'abord.")

            tokens = query.lower().split()
            scores = self._index.get_scores(tokens)

            ranked = sorted(
                zip(scores, self._chunks),
                key=lambda x: x[0],
                reverse=True,
            )

            results = [
                {
                    "content":     chunk.content,
                    "score":       float(score),
                    "doc_id":      chunk.doc_id,
                    "source":      chunk.metadata.get("source"),
                    "page_number": chunk.metadata.get("page_number"),
                    "chunk_index": chunk.metadata.get("chunk_index"),
                    "type":        chunk.type,
                }
                for score, chunk in ranked[:k]
                if score > 0
            ]

            logging.info(f"BM25 search : '{query[:60]}' → {len(results)} résultat(s)")
            return results

        except Exception as e:
            raise AgenticRagException(e, sys)
