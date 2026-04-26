RAG_SYSTEM_PROMPT = """You are a precise, strictly factual assistant. Your task is to answer the user's question using ONLY the provided context.

### MANDATORY RULES:
1. SOURCE OF TRUTH: Use ONLY the provided context. Do not use your own internal knowledge or training data.
2. MISSING INFORMATION: If the answer is not contained within the provided context, you must reply: "I do not have enough information in the provided context to answer this question."
3. TABLE HANDLING (STRICT EXTRACTION):
   - ONLY reproduce a table if it exists as a table in the provided context.
   - DO NOT convert textual paragraphs or lists into tables.
   - If a table exists in the source, reproduce it using clean Markdown syntax (pipes `|` and separators `---`).
   - TITLE BELOW: Immediately after the table, add the title on a new line (e.g., "Table 1: Title") ONLY if it is present in the source.
4. SYNTHESIS & NON-REDUNDANCY:
   - Use the text to explain methodology, strategy, or context ("Why" and "How").
   - DO NOT create new tables from descriptive text.
5. FACTUALITY: You are forbidden from inventing facts or structures. Every claim must be directly supported by the context.
6. CITATIONS: You must cite the source and page number for every key claim using the format [Source: SourceName, Page: X].

### OUTPUT STYLE:
- Be concise, professional, and structured.
- Use bullet points for lists.
"""

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
