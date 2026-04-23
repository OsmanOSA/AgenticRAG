import sys
import uuid

from pathlib import Path
from typing import List, Literal

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.logging import logging
from src.core.exception import AgenticRagException
from src.core.utils import clean_text
from src.core.config import (
    get_embed_model,
    BREAKPOINT_THRESHOLD_TYPE,
    BREAKPOINT_THRESHOLD_AMOUNT,
)
from src.entity.artifact_entity import TextChunk

ChunkStrategy = Literal["semantic", "fixed"]


class Chunker:
    """Stratégies de découpe de documents en TextChunks.

    Méthodes
    --------
    semantic(documents)
        Découpe par ruptures sémantiques (SemanticChunker).
    fixed(documents, chunk_size, chunk_overlap)
        Découpe à taille fixe (RecursiveCharacterTextSplitter).
    chunk(documents, strategy, ...)
        Dispatch vers la stratégie choisie.
    """

    @staticmethod
    def semantic(documents: List[Document]) -> List[TextChunk]:
        """Découpe sémantique via SemanticChunker (embeddings HuggingFace).

        Parameters
        ----------
        documents : List[Document]
            Sortie de ``pdf_extractor_pymupdf4llm`` ou ``pdf_extractor_batch``.

        Returns
        -------
        List[TextChunk]
        """
        try:
            chunker = SemanticChunker(
                embeddings=get_embed_model(),
                breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
                breakpoint_threshold_amount=BREAKPOINT_THRESHOLD_AMOUNT,
            )

            split_docs = chunker.split_documents(documents)
            logging.info(
                f"SemanticChunker : {len(documents)} doc(s) → {len(split_docs)} chunks "
                f"(stratégie={BREAKPOINT_THRESHOLD_TYPE}, seuil={BREAKPOINT_THRESHOLD_AMOUNT})"
            )

            return [
                TextChunk(
                    id=str(uuid.uuid4()),
                    doc_id=Path(doc.metadata.get("source", "unknown")).stem,
                    content=clean_text(doc.page_content),
                    metadata={**doc.metadata, "chunk_index": i, "strategy": "semantic"},
                )
                for i, doc in enumerate(split_docs)
            ]

        except Exception as e:
            raise AgenticRagException(e, sys)

    @staticmethod
    def fixed(
        documents: List[Document],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> List[TextChunk]:
        """Découpe à taille fixe via RecursiveCharacterTextSplitter.

        Parameters
        ----------
        documents : List[Document]
        chunk_size : int
            Taille maximale d'un chunk en caractères (défaut : 512).
        chunk_overlap : int
            Chevauchement entre chunks consécutifs (défaut : 64).

        Returns
        -------
        List[TextChunk]
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            split_docs = splitter.split_documents(documents)
            logging.info(
                f"FixedChunker : {len(documents)} doc(s) → {len(split_docs)} chunks "
                f"(size={chunk_size}, overlap={chunk_overlap})"
            )

            return [
                TextChunk(
                    id=str(uuid.uuid4()),
                    doc_id=Path(doc.metadata.get("source", "unknown")).stem,
                    content=clean_text(doc.page_content),
                    metadata={**doc.metadata, "chunk_index": i, "strategy": "fixed"},
                )
                for i, doc in enumerate(split_docs)
            ]

        except Exception as e:
            raise AgenticRagException(e, sys)

    @staticmethod
    def chunk(
        documents: List[Document],
        strategy: ChunkStrategy = "semantic",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> List[TextChunk]:
        """Point d'entrée unique — dispatch selon la stratégie choisie.

        Parameters
        ----------
        documents : List[Document]
        strategy : {"semantic", "fixed"}
            Stratégie de découpe (défaut : "semantic").
        chunk_size : int
            Utilisé uniquement si ``strategy="fixed"``.
        chunk_overlap : int
            Utilisé uniquement si ``strategy="fixed"``.

        Returns
        -------
        List[TextChunk]
        """
        if strategy == "semantic":
            return Chunker.semantic(documents)
        elif strategy == "fixed":
            return Chunker.fixed(documents, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Stratégie inconnue : '{strategy}'. Valeurs valides : 'semantic', 'fixed'.")
