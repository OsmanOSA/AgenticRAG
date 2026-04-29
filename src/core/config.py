import os
from pathlib import Path
from typing import Literal

# ─────────────────────────────────────────
# Chemins & extraction PDF
# ─────────────────────────────────────────
PATH_FILE_DOCUMENTS: Path = Path("datasets")

# ─────────────────────────────────────────
# VLM — Description d'images (Gemini)
# Désactivé si GEMINI_API_KEY absent
# ─────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_VLM_MODEL: str = "gemini-2.0-flash"

# ─────────────────────────────────────────
# Semantic chunking
# ─────────────────────────────────────────
NAME_EMBED_MODEL: str = "BAAI/bge-m3"

BREAKPOINT_THRESHOLD_AMOUNT: float = 95.0
BREAKPOINT_THRESHOLD_TYPE: Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
] = "percentile"

# ─────────────────────────────────────────
# Normalisation texte
# ─────────────────────────────────────────
LIGATURES: dict = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u00ad": "",    # Soft hyphen
}

# ─────────────────────────────────────────
# Qdrant — Vector store
# ─────────────────────────────────────────
QDRANT_URL: str        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = "agenticrag"
QDRANT_VECTOR_SIZE: int = 1024   # bge-m3 output dim

# ─────────────────────────────────────────
# Lazy loading du modèle d'embedding
# Le modèle (~500 Mo) n'est chargé qu'au
# premier appel à get_embed_model().
# ─────────────────────────────────────────
_embed_model = None

def get_embed_model():
    """Retourne le modèle d'embedding HuggingFace (singleton lazy-loaded)."""
    global _embed_model
    if _embed_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        _embed_model = HuggingFaceEmbeddings(model_name=NAME_EMBED_MODEL)
    return _embed_model


DEFAULT_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_LLM = "minimax/minimax-m2.5:free"



"""

Stratégies disponibles via ``breakpoint_threshold_type`` :

    - ``"percentile"``         : coupe aux X% de dissimilarité les plus forts
      (défaut 95.0 → conservatif, peu de chunks)
    - ``"standard_deviation"`` : coupe si dissimilarité > moyenne + X*σ
    - ``"interquartile"``      : coupe si dissimilarité > Q3 + X*(Q3-Q1)
    - ``"gradient"``           : coupe aux pics de variation de dissimilarité




"""