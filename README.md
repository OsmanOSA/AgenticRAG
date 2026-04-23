# RAG Agentique

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
│  DocChunks   ──► TextChunk             │
│                  TableChunk            │
│                  ImageChunk + VLM desc │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│           Chunking                     │
│  semantic_chunk (ruptures sémantiques) │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│            Indexation Qdrant           │
│ Embedder BAAI/bge-m3 ──► TextChunks    │
│ Embedder BAAI/bge-m3 ──► TablesChunks  │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│                Retrieval               │
│             Semantic search            │
│             Keyword search (BM25)      │
│             Reranker                   │
└────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│             Génération LLM             │   
└────────────────────────────────────────┘ 
                    │
                    ▼                 
              API FastAPI
```

---

## Structure du projet

```
AgenticRAG/
├── src/
│   ├── core/
│   │   ├── config.py          # Constantes & lazy loading modèles
│   │   ├── utils.py           # Extracteurs PDF, chunkers, utilitaires texte
│   │   ├── exception.py       # AgenticRagException
│   │   └── logging.py         # Configuration des logs
│   ├── entity/
│   │   └── artifact_entity.py # TextChunk, TableChunk, ImageChunk, DocChunks
│   ├── data/
│   │   ├── chunker.py         # Stratégies de découpe
│   │   └── data_ingestion.py  # Pipeline d'ingestion
│   ├── indexing/
│   │   ├── embedder.py        # Génération des embeddings
│   │   └── vector_store.py    # Interface Qdrant
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
├── frontend/                  # Interface User
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
- Qdrant
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

### Qdrant local

```bash
docker run -p 6333:6333 qdrant/qdrant   
```
---

## Utilisation

### Extraction et chunking

```python
from src.core.utils import pdf_to_typed_chunks, semantic_chunk, save_markdown

# Extraction page par page
docs = pdf_to_typed_chunks("mon_document.pdf")

# Sauvegarde du markdown extrait
save_result(docs, name_method="extraction", source_name="mon_document")

# Chunking sémantique
chunks = semantic_chunk(docs)
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
| LLM | OpenRouter |
| Extraction PDF | pymupdf4llm |
| Embeddings | BAAI/bge-m3 |
| Vector store | Qdrant |
| Recherche keyword | BM25 (rank-bm25) |
| Recherche graphe | NetworkX |
| VLM local | Ollama (llava) |
| API | FastAPI |
| Évaluation | RAGAS, LangSmith |

---

## Auteur
Osman SAID ALI
