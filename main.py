import gc

from src.data.data_ingestion import PdfIngestion
from src.data.chunker import Chunker
from src.core.utils import typed_chunks_to_documents
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.retrieval.semantic_search import SemanticSearch
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.reranker import Reranker
from src.generation.llm_client import LLMClient


QUERY = input("Ta question : ")

# ── 1. Extraction ────────────────────────────────────────────────────────
print("\n=== 1. EXTRACTION ===")
all_chunks = PdfIngestion.load_typed(vlm_model="")

texts  = PdfIngestion.text_chunks(all_chunks)
tables = PdfIngestion.table_chunks(all_chunks)
images = PdfIngestion.image_chunks(all_chunks)

print(f"  Textes   : {len(texts)}")
print(f"  Tableaux : {len(tables)}")
print(f"  Images   : {len(images)}")
