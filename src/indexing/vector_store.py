import sys

from typing import List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.core.logging import logging
from src.core.exception import AgenticRagException
from src.core.config import QDRANT_URL, QDRANT_COLLECTION, QDRANT_VECTOR_SIZE
from src.entity.artifact_entity import TextChunk, TableChunk


class VectorStore:
    """Interface Qdrant pour l'indexation et la recherche de TextChunks.

    Méthodes
    --------
    create_collection()   Crée la collection si elle n'existe pas.
    upsert(chunks)        Indexe une liste de TextChunks embeddés.
    search(vector, k)     Recherche les k voisins les plus proches.
    delete_collection()   Supprime la collection.
    """

    def __init__(self, url: str = QDRANT_URL, 
                 collection: str = QDRANT_COLLECTION):
        
        try:

            self.client = QdrantClient(url=url)
            self.collection = collection
            self.create_collection()

        except Exception as e:
            raise AgenticRagException(e, sys)

    def create_collection(self) -> None:
        """Crée la collection Qdrant si elle n'existe pas encore."""

        try:

            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection not in existing:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=QDRANT_VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )

                logging.info(f"Collection Qdrant créée : '{self.collection}'")

            else:

                logging.info(f"Collection Qdrant existante : '{self.collection}'")

        except Exception as e:
            raise AgenticRagException(e, sys)

    def upsert(self, chunks: List[TextChunk | TableChunk]) -> None:
        """Indexe les chunks dans Qdrant.

        Parameters
        ----------
        chunks : List[TextChunk | TableChunk]
            Chunks avec ``metadata["embedding"]`` renseigné par ``Embedder``.
        """

        try:

            points: List = []

            for chunk in chunks:
                vector = chunk.metadata.get("embedding")
                if vector is None:

                    logging.warning(f"Chunk {chunk.id} sans embedding — ignoré.")
                    
                    continue

                payload = {
                    "id":          chunk.id,
                    "doc_id":      chunk.doc_id,
                    "content":     chunk.content,
                    "type":        chunk.type,
                    "page_number": chunk.metadata.get("page_number"),
                    "source":      chunk.metadata.get("source"),
                    "chunk_index": chunk.metadata.get("chunk_index"),
                }
                
                points.append(PointStruct(id=chunk.id, vector=vector, payload=payload))

            self.client.upsert(collection_name=self.collection, points=points)
            logging.info(f"Qdrant : {len(points)} chunk(s) indexés dans '{self.collection}'")

        except Exception as e:
            raise AgenticRagException(e, sys)

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        doc_id: Optional[str] = None) -> List[dict]:
        """Recherche les k chunks les plus proches du vecteur requête.

        Parameters
        ----------
        query_vector : List[float]
            Vecteur de la requête (produit par ``Embedder.embed_query``).
        k : int, optional
            Nombre de résultats (défaut : 5).
        doc_id : str, optional
            Filtre sur un document source spécifique.

        Returns
        -------
        List[dict]
            Résultats avec ``content``, ``score``, ``doc_id``, ``page_number``.
        """
        
        try:

            query_filter = None

            if doc_id:
                query_filter = Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )

            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )

            results = [
                {
                    "content":     h.payload.get("content"),
                    "score":       h.score,
                    "doc_id":      h.payload.get("doc_id"),
                    "source":      h.payload.get("source"),
                    "page_number": h.payload.get("page_number"),
                    "chunk_index": h.payload.get("chunk_index"),
                    "type":        h.payload.get("type"),
                }
                for h in hits
            ]
            logging.info(f"Qdrant search : {len(results)} résultat(s) pour k={k}")
            return results

        except Exception as e:
            raise AgenticRagException(e, sys)

    def count(self) -> int:
        """Retourne le nombre de points dans la collection."""
        try:
            return self.client.count(collection_name=self.collection).count
        except Exception:
            return 0

    def scroll_all(self) -> List[dict]:
        """Récupère tous les points de la collection (pour reconstruire BM25).

        Returns
        -------
        List[dict]
            Chaque dict contient ``content``, ``doc_id``, ``type``,
            ``source``, ``page_number``, ``chunk_index``.
        """
        try:
            results = []
            offset  = None

            while True:
                records, next_offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for r in records:
                    results.append({
                        "content":     r.payload.get("content", ""),
                        "doc_id":      r.payload.get("doc_id", ""),
                        "type":        r.payload.get("type", "Text"),
                        "source":      r.payload.get("source"),
                        "page_number": r.payload.get("page_number"),
                        "chunk_index": r.payload.get("chunk_index"),
                    })
                if next_offset is None:
                    break
                offset = next_offset

            logging.info(f"scroll_all : {len(results)} points récupérés")
            return results

        except Exception as e:
            raise AgenticRagException(e, sys)

    def delete_collection(self) -> None:
        """Supprime la collection Qdrant."""

        try:

            self.client.delete_collection(self.collection)
            logging.info(f"Collection supprimée : '{self.collection}'")
            
        except Exception as e:
            raise AgenticRagException(e, sys)
