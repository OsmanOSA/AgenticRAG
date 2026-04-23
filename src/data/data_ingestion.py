import sys

from itertools import chain
from typing import Dict, List

from src.core.logging import logging
from src.core.exception import AgenticRagException
from src.core.utils import pdf_to_typed_chunks
from src.core.config import PATH_FILE_DOCUMENTS
from src.entity.artifact_entity import DoclingChunks, TextChunk, TableChunk, ImageChunk


class PdfIngestion:

    @staticmethod
    def load_typed(vlm_model: str = "") -> Dict[str, DoclingChunks]:
        """Extrait texte, tableaux et images de tous les PDFs du dossier.

        Returns
        -------
        Dict[str, DoclingChunks]
            Clé = stem du fichier PDF, valeur = {"text", "tables", "images"}.
        """
        try:
            pdf_files = sorted(PATH_FILE_DOCUMENTS.glob("*.pdf"))
            if not pdf_files:
                logging.warning(f"Aucun PDF trouvé dans : {PATH_FILE_DOCUMENTS}")
                return {}

            results: Dict[str, DoclingChunks] = {}
            for pdf_path in pdf_files:
                results[pdf_path.stem] = pdf_to_typed_chunks(str(pdf_path), vlm_model=vlm_model)

            logging.info(f"Ingestion terminée : {len(results)} PDF(s)")
            return results

        except Exception as e:
            raise AgenticRagException(e, sys)

    @staticmethod
    def text_chunks(all_chunks: Dict[str, DoclingChunks]
                    ) -> List[TextChunk]:
        """Retourne tous les TextChunks — entrée du Chunker pour l'embedding."""
        
        return list(chain.from_iterable(c["text"] for c in all_chunks.values()))

    @staticmethod
    def table_chunks(all_chunks: Dict[str, DoclingChunks]
                     ) -> List[TableChunk]:
        """Retourne tous les TableChunks — entrée de l'index BM25/structuré."""
        
        return list(chain.from_iterable(c["tables"] for c in all_chunks.values()))

    @staticmethod
    def image_chunks(all_chunks: Dict[str, DoclingChunks]
                     ) -> List[ImageChunk]:
        """Retourne tous les ImageChunks — entrée de l'index multimodal."""
        
        return list(chain.from_iterable(c["images"] for c in all_chunks.values()))
