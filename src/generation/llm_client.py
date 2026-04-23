import sys
import os
import time

from typing import List

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

from src.core.logging import logging
from src.core.exception import AgenticRagException
from src.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from src.core.config import OPENROUTER_BASE_URL, MODEL_LLM

load_dotenv()

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 5  # secondes, doublé à chaque tentative

class LLMClient:
    """Client de génération via OpenRouter (MiniMax / tout modèle OpenRouter free).

    Utilise l'API OpenAI-compatible d'OpenRouter.
    """

    def __init__(
        self,
        model: str = MODEL_LLM,
        temperature: float = 0.2,
        max_tokens: int = 1024):
        """
        Parameters
        ----------
        model : str
            Identifiant du modèle OpenRouter (ex: ``"google/gemma-3-27b-it:free"``).
        temperature : float
            Température de génération (défaut : 0.2 pour la précision factuelle).
        max_tokens : int
            Nombre maximum de tokens générés.
        """
        
        try:

            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "Variable d'environnement OPENROUTER_API_KEY manquante."
                )

            self._client = OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
            )
            self.model       = model
            self.temperature = temperature
            self.max_tokens  = max_tokens

            logging.info(f"LLMClient initialisé — modèle : {model}")

        except Exception as e:
            raise AgenticRagException(e, sys)

    def _call(self, messages: list) -> str:
        """Appel API avec retry exponentiel sur RateLimitError (429)."""
        
        delay = _RETRY_BASE_DELAY

        for attempt in range(1, _MAX_RETRIES + 1):

            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            
            except RateLimitError:
                if attempt == _MAX_RETRIES:
                    raise
                logging.warning(
                    f"Rate limit 429 — tentative {attempt}/{_MAX_RETRIES}, "
                    f"attente {delay}s…"
                )
                time.sleep(delay)
                delay *= 2

    def generate(self, query: str, context_chunks: List[dict]) -> str:
        """Génère une réponse RAG à partir de la question et du contexte.

        Parameters
        ----------
        query : str
            Question de l'utilisateur.
        context_chunks : List[dict]
            Résultats fusionnés du Reranker (``content``, ``source``, ``page_number``…).

        Returns
        -------
        str
            Réponse générée par le modèle.
        """
        
        try:

            user_prompt = build_rag_prompt(query, context_chunks)
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ]
            answer = self._call(messages)
            logging.info(
                f"LLMClient : réponse générée ({len(answer)} chars) "
                f"— modèle {self.model}"
            )

            return answer.strip()

        except Exception as e:
            raise AgenticRagException(e, sys)

    def generate_raw(self, system: str, user: str) -> str:
        """Appel direct sans template RAG — utile pour le router / critique LangGraph.

        Parameters
        ----------
        system : str
            Prompt système.
        user : str
            Prompt utilisateur.

        Returns
        -------
        str
            Réponse brute du modèle.
        """
        
        try:

            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            return self._call(messages).strip()

        except Exception as e:
            raise AgenticRagException(e, sys)
