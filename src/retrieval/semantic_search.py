import sys

from typing import List, Optional

from src.core.logging import logging
from src.core.exception import AgenticRagException
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore


class SemanticSearch:
    """Recherche sémantique dense via Qdrant.

    Encode la requête avec l'Embedder et retourne les chunks
    les plus proches dans le vector store.
    """

    def __init__(self, embedder: Embedder, 
                 vector_store: VectorStore):
        
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        k: int = 10,
        doc_id: Optional[str] = None) -> List[dict]:
        """Recherche les chunks les plus pertinents pour une requête.

        Parameters
        ----------
        query : str
            Question ou requête utilisateur.
        k : int, optional
            Nombre de résultats à retourner (défaut : 10).
        doc_id : str, optional
            Restreindre la recherche à un document source.

        Returns
        -------
        List[dict]
            Résultats triés par score décroissant, chacun contenant
            ``content``, ``score``, ``doc_id``, ``source``, ``page_number``.
        """
        
        try:

            query_vector = self.embedder.embed_query(query)
            results = self.vector_store.search(query_vector, k=k, doc_id=doc_id)
            
            logging.info(f"SemanticSearch : '{query[:60]}...' → {len(results)} résultat(s)")
            return results

        except Exception as e:
            raise AgenticRagException(e, sys)
