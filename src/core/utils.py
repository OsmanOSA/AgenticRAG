import sys
import re
import uuid
import base64
import yaml
import json
import fitz
import pymupdf4llm
import ollama
import tiktoken
import warnings

from pathlib import Path
from typing import List, Any, Dict, Optional

from unidecode import unidecode
from langchain_core.documents import Document  # type: ignore

from src.core.exception import AgenticRagException
from src.core.logging import logging
from src.core.config import LIGATURES
from src.entity.artifact_entity import TextChunk, TableChunk, ImageChunk, DocChunks
from src.entity.artifact_entity import JudgeResult, JudgeScore

# ─────────────────────────────────────────
# PDF — Extractor
# ─────────────────────────────────────────

def pdf_to_typed_chunks(
    file_path: str,
    vlm_model: Optional[str] = None,
) -> DocChunks:
    """Extrait texte, tableaux et images d'un PDF.

    - Texte    : ``pymupdf4llm.to_markdown(page_chunks=True)``
    - Tableaux : ``fitz.Page.find_tables()`` → CSV
    - Images   : ``fitz.Page.get_images()`` → base64

    Parameters
    ----------
    file_path : str
        Chemin vers le fichier PDF.
    vlm_model : str, optional
        Modèle Ollama pour décrire les images. ``""`` pour désactiver.

    Returns
    -------
    DocChunks
        ``{"text": List[TextChunk], "tables": List[TableChunk], "images": List[ImageChunk]}``
    """
    try:
    
        doc_id = Path(file_path).stem
        text_chunks:  List[TextChunk]  = []
        table_chunks: List[TableChunk] = []
        image_chunks: List[ImageChunk] = []

        # ── Texte + Tableaux + Images (page par page) ────────────────────────
        pdf = fitz.open(file_path)
        seen_img_hashes: set = set()

        for page_num in range(len(pdf)):
            page_md = pymupdf4llm.to_markdown(
                pdf,
                pages=[page_num],
                page_chunks=False,
                write_images=True,
                image_path="./datasets",
                show_progress=False,
            )

            # Tableaux : extraits depuis le markdown avant stripping
            md_tables = _extract_markdown_tables(page_md, page_num, doc_id, file_path)
            table_chunks.extend(md_tables)

            # Texte : markdown sans les lignes de tableau
            content = clean_text(_strip_markdown_tables(page_md))
            if content:
                text_chunks.append(TextChunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=content,
                    metadata={
                        "source": file_path,
                        "page_number": page_num,
                        "chunk_index": page_num}))

        for page_num, page in enumerate(pdf):

            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    img_bytes = pdf.extract_image(xref)["image"]
                except Exception:
                    continue

                img_hash = hash(img_bytes)
                if img_hash in seen_img_hashes:
                    continue
                seen_img_hashes.add(img_hash)

                b64 = base64.b64encode(img_bytes).decode()
                description = _describe_image_ollama(b64, vlm_model) if vlm_model else ""

                image_chunks.append(ImageChunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=description,
                    image_id=f"{doc_id}_p{page_num}_i{xref}",
                    metadata={
                        "source": file_path,
                        "page_number": page_num,
                        "b64_image": b64,
                        "description": description,
                    },
                ))

        pdf.close()
        logging.info(
            f"{doc_id} → {len(text_chunks)} textes, "
            f"{len(table_chunks)} tableaux, {len(image_chunks)} images"
        )
        
        return {"text": text_chunks, 
                "tables": table_chunks, 
                "images": image_chunks}

    except Exception as e:
        raise AgenticRagException(e, sys)


def typed_chunks_to_documents(text_chunks: List[TextChunk]) -> List[Document]:
    """Convertit une liste de TextChunks en Documents LangChain pour le Chunker."""
    
    return [
        Document(page_content=c.content, metadata=c.metadata)
        for c in text_chunks]


def _strip_markdown_tables(text: str) -> str:
    """Supprime les blocs de tableaux markdown (lignes contenant |) du texte."""

    lines = text.splitlines()
    filtered = [l for l in lines if not re.match(r"^\s*\|", l)]
    
    return "\n".join(filtered)


_CAPTION_RE = re.compile(
    r"^\s*(\*{0,2})(tableau|table|figure|fig\.?)\s*\d*\s*[:\-–]?\s*.+",
    re.IGNORECASE,
)


def _extract_markdown_tables(
    text: str,
    page_num: int,
    doc_id: str,
    source: str) -> List[TableChunk]:
    """Extrait les blocs de tableaux markdown avec leur titre (avant ou après).

    Filtre les faux tableaux (TOC, pointillés) : exige une ligne séparateur ``|---|``.
    Capture le titre le plus proche (ligne non-table avant ou après le bloc).
    """
    
    try:

        lines = text.splitlines()
        n = len(lines)
        table_chunks: List[TableChunk] = []
        t_idx = 0
        i = 0

        while i < n:
            if not re.match(r"^\s*\|", lines[i]):
                i += 1
                continue

            # Début d'un bloc table
            start = i
            while i < n and re.match(r"^\s*\|", lines[i]):
                i += 1
            end = i  # lignes[start:end] = le bloc

            block = lines[start:end]
            has_separator = any(re.match(r"^\s*\|[-| ]+\|", l) for l in block)
            if not has_separator:
                continue

            raw_content = "\n".join(block)

            # Titre : scan les lignes après le tableau, conservé seulement s'il contient "Tableau"
            title = ""
            for k in range(end, min(end + 4, n)):
                candidate = lines[k].strip()
                if not candidate or re.match(r"^\s*\|", lines[k]):
                    continue
                if "tableau" in candidate.lower():
                    title = re.sub(r":*+", "", candidate).strip()
                    break

            content = _augment_table_content(raw_content, title)
            table_chunks.append(TableChunk(
                id=str(uuid.uuid4()),
                doc_id=doc_id,
                content=content,
                table_id=f"{doc_id}_p{page_num}_t{t_idx}",
                metadata={
                    "source": source,
                    "page_number": page_num,
                    "table_index": t_idx,
                    "raw_table": raw_content,
                    "title": title,
                },
            ))
            t_idx += 1

        return table_chunks
    
    except Exception as e:
        raise AgenticRagException(e, sys)

def _augment_table_content(markdown_table: str,
                           title: str = "") -> str:
    """Préfixe le tableau avec son titre et ses noms de colonnes en langage naturel.

    Nettoie aussi les artefacts PDF : balises <br>, caractères de remplacement Unicode.

    Ex. sortie :
        
        | Type de donnée | ...
        Tableau N : Stratégie de collecte des données
        Colonnes : Type de donnée, Granularité temporelle, Source de donnée.
    """
    
    try:

        cleaned = re.sub(r"­\s*<br\s*/?>", "", markdown_table)  # trait d union conditionnel + saut -> join
        cleaned = re.sub(r"<br\s*/?>", " ", cleaned)            # <br> restants -> espace
        cleaned = re.sub(r"­", "", cleaned)                  # traits d union conditionnels restants
        cleaned = re.sub(r"�", "", cleaned)                # caractere de remplacement
        cleaned = re.sub(r"[ 	]{2,}", " ", cleaned)            # espaces horizontaux multiples

        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        if not lines:
            return markdown_table

        # Header = première ligne, ignorer la ligne séparateur (|---|) pour les noms de colonnes

        header_line = lines[0]
        cols = [c.strip().strip("*") for c in header_line.split("|")
                if c.strip() and not re.match(r"^-+$", c.strip())]
        cleaned_table = "\n".join(lines)   # reconstruire avec newlines propres
        parts: List[str] = []

        
        parts.append(cleaned_table)
        if title:
            title = re.sub(r"^\s*#{1,6}\s*", "", title).strip()
            parts.append("")
            parts.append(title)
            
        return "\n".join(parts)

    except Exception as e:
        raise AgenticRagException(e, sys)

def _describe_image_ollama(b64_image: str, model: str) -> str:
    """Envoie une image en base64 à Ollama et retourne la description."""
    
    try:
        
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Describe this image concisely for a RAG system.",
                "images": [b64_image]}])
        
        return response["message"]["content"].strip()
    
    except Exception as e:
        logging.warning(f"[VLM] ✗ Échec ({model}) : {e}")
        return ""


# ─────────────────────────────────────────
# Text Utilities
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    """Nettoie et normalise un texte brut issu d'un PDF."""
    
    try:

        for lig, replacement in LIGATURES.items():
            text = text.replace(lig, replacement)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = unidecode(text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        
        return text.strip()
    
    except Exception as e:
        raise AgenticRagException(e, sys)


def count_tokens(text: str, 
                 model: str = "gpt-4o") -> int:
    """Compte le nombre de tokens d'un texte pour un modèle donné."""
    
    try:

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    
    except Exception as e:
        raise AgenticRagException(e, sys)




# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_score(raw: str, 
                criterion: str) -> JudgeScore:
    """Extrait score + raisonnement du JSON retourné par le LLM."""
    
    try:

        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        
        if not match:
            raise ValueError("Aucun JSON trouvé dans la réponse")
        data = json.loads(match.group())
        
        return JudgeScore(
            score=max(0.0, min(1.0, float(data["score"]))),
            reasoning=str(data.get("reasoning", "")))
    
    except Exception as exc:
        logging.warning(f"Parse score '{criterion}' échoué : {exc} | raw: {raw[:120]}")
        return JudgeScore(score=0.0, reasoning=f"[parse error] {exc}")


def build_user_prompt(question: str, 
                      context_chunks: list[dict],
                        answer: str) -> str:
    
    context_text = "\n\n".join(
        f"[{i + 1}] {c['content'][:600]}"
        for i, c in enumerate(context_chunks))
    
    return (
        f"Question : {question}\n\n"
        f"Contexte :\n{context_text}\n\n"
        f"Réponse à évaluer :\n{answer}")