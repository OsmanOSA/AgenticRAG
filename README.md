# NovaRAG

Système de **Retrieval-Augmented Generation agentique** construit avec LangGraph et LangChain. NovaRAG extrait, indexe et interroge des documents PDF en combinant recherche sémantique, recherche par mots-clés (BM25) et recherche par graphe, avec un agent qui critique et affine ses réponses.

---

## Architecture

```
Documents PDF
     │
     ▼
┌────────────────────────────────────────┐
│           Extraction (src/core/utils)  │
│  pymupdf4llm ──► List[Document]        │
│  Docling     ──► TextChunk             │
│                  TableChunk            │
│                  ImageChunk + VLM desc │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│           Chunking                     │
│  split_chunk    (taille fixe)          │
│  semantic_chunk (ruptures sémantiques) │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│           Indexation                   │
│  Embedder ──► ChromaDB (vecteurs)      │
│           ──► BM25 index (keywords)    │
│           ──► NetworkX (graphe)        │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│           Agent LangGraph              │
│  Router ──► Semantic search            │
│         ──► Keyword search (BM25)      │
│         ──► Graph search               │
│  Reranker ──► Critique ──► Génération  │
└────────────────────────────────────────┘
                     │
                     ▼
              API FastAPI
```

---

## Structure du projet

```
NovaRAG/
├── src/
│   ├── core/
│   │   ├── config.py          # Constantes & lazy loading modèles
│   │   ├── utils.py           # Extracteurs PDF, chunkers, utilitaires texte
│   │   ├── exception.py       # AgenticRagException
│   │   └── logging.py         # Configuration des logs
│   ├── entity/
│   │   └── artifact_entity.py # TextChunk, TableChunk, ImageChunk, DoclingChunks
│   ├── data/
│   │   ├── loaders.py         # Chargement multi-format
│   │   ├── cleaner.py         # Nettoyage des données
│   │   ├── chunker.py         # Stratégies de découpe
│   │   └── data_ingestion.py  # Pipeline d'ingestion
│   ├── indexing/
│   │   ├── embedder.py        # Génération des embeddings
│   │   └── vector_store.py    # Interface ChromaDB
│   ├── retrieval/
│   │   ├── semantic_search.py # Recherche dense (embeddings)
│   │   ├── keyword_search.py  # Recherche BM25
│   │   ├── graph_search.py    # Recherche par graphe (NetworkX)
│   │   └── reranker.py        # Reranking des résultats
│   ├── agent/
│   │   ├── graph.py           # Graphe LangGraph principal
│   │   ├── router.py          # Routage des requêtes
│   │   ├── state.py           # État de l'agent
│   │   └── critique.py        # Nœud de critique/correction
│   └── generation/
│       ├── llm_client.py      # Client LLM (Claude)
│       └── prompts.py         # Templates de prompts
├── backend/
│   └── api/
│       ├── main.py            # Application FastAPI
│       ├── schemas.py         # Schémas Pydantic
│       └── dependencies.py    # Injection de dépendances
├── monitoring/
│   ├── eval_ragas.py          # Évaluation RAGAS
│   ├── benchmark_piaf.py      # Benchmark PIAF
│   └── langsmith_eval.py      # Traçabilité LangSmith
├── test/
│   ├── conftest.py
│   └── test_agent.py
├── main.py                    # Point d'entrée principal
└── requirements.txt
```

---

## Installation

### Prérequis

- Python 3.12+
- Java 21+ (requis par `opendataloader-pdf`)
- [Ollama](https://ollama.com/download) (pour les descriptions d'images VLM)

### Environnement virtuel

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate        # Linux/macOS
pip install -r requirements.txt
```

### Modèle VLM local (descriptions d'images)

```bash
ollama pull llava
ollama serve   # laisser tourner en arrière-plan
```

---

## Utilisation

### Extraction et chunking

```python
from src.core.utils import pdf_extractor_pymupdf4llm, semantic_chunk, save_result

# Extraction page par page
docs = pdf_extractor_pymupdf4llm("mon_document.pdf")

# Sauvegarde du markdown extrait
save_result(docs, name_method="extraction", source_name="mon_document")

# Chunking sémantique
chunks = semantic_chunk(docs)
```

### Extraction Docling (text + tables + images séparés)

```python
from src.core.utils import pdf_extractor_docling

result = pdf_extractor_docling("mon_document.pdf", vlm_model="llava")

text_chunks  = result["text"]    # List[TextChunk]
table_chunks = result["tables"]  # List[TableChunk]
image_chunks = result["images"]  # List[ImageChunk] avec description VLM
```

### API

```bash
uvicorn backend.api.main:app --reload --port 8000
```

---

## Types de chunks

| Type | Contenu | Usage |
|---|---|---|
| `TextChunk` | Paragraphes, titres, listes | Embedding sémantique |
| `TableChunk` | CSV des tableaux détectés | Recherche structurée |
| `ImageChunk` | Image en base64 + description VLM | Recherche multimodale |

---

## Évaluation

```bash
# RAGAS
python monitoring/eval_ragas.py

# Benchmark PIAF
python monitoring/benchmark_piaf.py

# Tests unitaires
pytest test/ --cov=src
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Orchestration agent | LangGraph |
| LLM | Claude (Anthropic) |
| Extraction PDF | pymupdf4llm, Docling |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| Recherche keyword | BM25 (rank-bm25) |
| Recherche graphe | NetworkX |
| VLM local | Ollama (llava) |
| API | FastAPI |
| Évaluation | RAGAS, LangSmith |

---

## Auteur

Osman SAID ALI
