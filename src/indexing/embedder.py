import sys

from typing import List, Union

from langchain_core.documents import Document  # type: ignore

from src.core.logging import logging
from src.core.exception import AgenticRagException
from src.core.config import get_embed_model
from src.entity.artifact_entity import TextChunk

Embeddable = Union[List[TextChunk], List[Document]]


class Embedder:
    """Génère et attache les embeddings aux TextChunks.

    Utilise le modèle HuggingFace configuré dans ``config.py``
    """

    def __init__(self):
        self._model = get_embed_model()

    def embed(self, chunks: Embeddable) -> Embeddable:
        """Génère les embeddings et les attache aux chunks in-place.

        Parameters
        ----------
        chunks : List[TextChunk]
            Chunks à encoder.

        Returns
        -------
        List[TextChunk]
            Les mêmes chunks avec ``metadata["embedding"]`` renseigné.
        """
        try:

            if not chunks:
                logging.info("Chunks vide")
                return chunks

            texts = [
                c.page_content if isinstance(c, Document) else c.content
                for c in chunks
            ]

            batch_size = 8
            total_batches = -(-len(texts) // batch_size)
            vectors: list = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start: start + batch_size]
                vectors.extend(self._model.embed_documents(batch))
                logging.info(f"Embedder : batch {start // batch_size + 1}/{total_batches} encodé ({len(batch)} chunks)")

            for chunk, vector in zip(chunks, vectors):
                chunk.metadata["embedding"] = vector

            logging.info(f"Embedder : {len(chunks)} chunk(s) encodés (dim={len(vectors[0])})")
            return chunks

        except Exception as e:
            raise AgenticRagException(e, sys)

    def embed_query(self, query: str) -> List[float]:
        """Encode une requête utilisateur en vecteur.

        Parameters
        ----------
        query : str
            Texte de la requête.

        Returns
        -------
        List[float]
            Vecteur d'embedding.
        """
        try:

            return self._model.embed_query(query)
        
        except Exception as e:
            raise AgenticRagException(e, sys)
