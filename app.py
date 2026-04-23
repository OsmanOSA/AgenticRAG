"""
app.py — Interface Streamlit NovaRAG.
Requiert que ingest.py ait été lancé au préalable (collection Qdrant peuplée).

    streamlit run app.py
"""
from types import SimpleNamespace

import streamlit as st

from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.retrieval.semantic_search import SemanticSearch
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.reranker import Reranker
from src.generation.llm_client import LLMClient


# ── Chargement des ressources (mis en cache) ─────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modèle d'embedding…")
def load_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner="Connexion à Qdrant…")
def load_store() -> VectorStore:
    store = VectorStore()
    if store.count() == 0:
        st.error("La collection Qdrant est vide. Lance `python ingest.py` d'abord.")
        st.stop()
    return store


@st.cache_resource(show_spinner="Construction de l'index BM25…")
def load_keyword_search(_store: VectorStore) -> tuple[KeywordSearch, list]:
    """Reconstruit le BM25 depuis les payloads Qdrant."""
    payloads = _store.scroll_all()
    chunks   = [
        SimpleNamespace(
            content=p["content"],
            doc_id=p["doc_id"],
            type=p["type"],
            metadata={
                "source":      p["source"],
                "page_number": p["page_number"],
                "chunk_index": p["chunk_index"],
            },
        )
        for p in payloads
        if p["content"]
    ]
    ks = KeywordSearch()
    ks.build(chunks=chunks)
    return ks, chunks


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NovaRAG", page_icon="🔍", layout="wide")
st.title("🔍 NovaRAG")
st.caption("Retrieval-Augmented Generation sur vos documents PDF")

# Initialisation des ressources
embd  = load_embedder()
store = load_store()
ks, _ = load_keyword_search(store)

sem_search = SemanticSearch(embd, store)
reranker   = Reranker()
llm        = LLMClient()

# Historique de conversation
if "history" not in st.session_state:
    st.session_state.history = []

# Affichage de l'historique
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input utilisateur
query = st.chat_input("Posez votre question…")

if query:
    # Affiche la question
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours…"):

            # Retrieval hybride
            sem_results = sem_search.search(query, k=10)
            kw_results  = ks.search(query, k=10)
            context     = reranker.fuse(sem_results, kw_results, query=query, top_k=5)

            # Génération
            answer = llm.generate(query=query, context_chunks=context)

        st.markdown(answer)

        # Sources dans un expander
        with st.expander("Sources utilisées", expanded=False):
            for i, r in enumerate(context, 1):
                badge = "📊" if r.get("type") == "Table" else "📄"
                st.markdown(
                    f"**[{i}]** {badge} `{r.get('source', '?')}` "
                    f"— page {r.get('page_number', '?')} "
                    f"— score RRF `{r['rerank_score']:.4f}`"
                )
                st.code(r["content"][:300], language="markdown")

    st.session_state.history.append({"role": "assistant", "content": answer})
