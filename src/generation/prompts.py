RAG_SYSTEM_PROMPT = """You are a precise and factual assistant. Answer the user's question using ONLY the context provided below.

Rules:
- If a table is in the context, reproduce it as a markdown table.
- If the answer is not in the context, say "I don't have enough information to answer this question."
- Do not invent facts or add knowledge outside the context.
- Be concise and structured. Use bullet points or numbered lists when appropriate.
- Cite the source (source + page number) at the end of each key claim when available."""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    """Construit le prompt utilisateur avec le contexte récupéré.

    Parameters
    ----------
    query : str
        Question de l'utilisateur.
    context_chunks : list[dict]
        Résultats du reranker — chaque dict contient ``content``, ``source``, ``page_number``.

    Returns
    -------
    str
        Prompt utilisateur formaté avec le contexte numéroté.
    """

    if not context_chunks:
        context_block = "No context available."
    
    else:
        parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            source = chunk.get("source", "unknown")
            page   = chunk.get("page_number", "?")
            content = chunk.get("content", "").strip()
            parts.append(f"[{i}] (source: {source}, page: {page})\n{content}")
        context_block = "\n\n".join(parts)

    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
