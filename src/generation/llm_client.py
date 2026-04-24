import sys
import os
import time

from typing import List

from openai import OpenAI, RateLimitError, InternalServerError, APIStatusError
from dotenv import load_dotenv

from src.core.logging import logging
from src.core.exception import AgenticRagException
from src.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from src.core.config import OPENROUTER_BASE_URL, MODEL_LLM

load_dotenv()

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 5  # secondes, doublé à chaque tentative


class LLMClient:
    """Client de génération via OpenRouter (API OpenAI-compatible)."""

    def __init__(
        self,
        model: str = MODEL_LLM,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "Variable d'environnement OPENROUTER_API_KEY manquante."
                )

            self._client = OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                max_retries=0,
            )
            self.model       = model
            self.temperature = temperature
            self.max_tokens  = max_tokens

            logging.info(f"LLMClient initialisé — modèle : {model}")

        except Exception as e:
            raise AgenticRagException(e, sys)

    def _call(self, messages: list) -> str:
        """Appel API avec retry exponentiel sur 429 et 503."""

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

            except RateLimitError as e:
                if attempt == _MAX_RETRIES:
                    raise
                logging.warning(
                    f"Rate limit 429 — tentative {attempt}/{_MAX_RETRIES}, "
                    f"attente {delay}s…"
                )
                time.sleep(delay)
                delay *= 2

            except (InternalServerError, APIStatusError) as e:
                status = getattr(e, "status_code", "?")
                if attempt == _MAX_RETRIES or status not in (500, 502, 503, 504):
                    raise
                logging.warning(
                    f"Erreur provider {status} — tentative {attempt}/{_MAX_RETRIES}, "
                    f"attente {delay}s…"
                )
                time.sleep(delay)
                delay *= 2

    def generate(self, query: str,
                  context_chunks: List[dict]) -> str:
        """Génère une réponse RAG à partir de la question et du contexte."""

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
        """Appel direct sans template RAG — utile pour le router / critique LangGraph."""

        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            return self._call(messages).strip()

        except Exception as e:
            raise AgenticRagException(e, sys)
