"""
LLM-as-Judge — évaluation automatique des réponses RAG.

3 dimensions scorées de 0.0 à 1.0 :
    faithfulness  — la réponse est-elle ancrée dans le contexte fourni ?
    relevance     — la réponse répond-elle à la question posée ?
    completeness  — la réponse exploite-t-elle toutes les infos pertinentes ?

Les scores sont uploadés dans Langfuse si un trace_id est fourni.

Usage
-----
>>> from monitoring.llm_as_judge import judge
>>> result = judge.evaluate(
...     question="Quelle est la stratégie de collecte de données définie ?",
...     context_chunks=context,   # liste de dicts avec 'content'
...     answer=answer,
...     trace_id=trace_id,        # optionnel — upload Langfuse
... )
>>> print(result.overall)
"""
import json
import re
from dataclasses import dataclass

from src.core.logging import logging
from backend.api.dependencies import get_llm

# ── Prompts ───────────────────────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = """\
Tu es un évaluateur expert en qualité de réponses RAG.
Évalue si la réponse est entièrement ancrée dans le contexte fourni (pas d'hallucination).

Barème :
  1.0 — toutes les affirmations sont explicitement présentes dans le contexte
  0.7 — la plupart des affirmations sont supportées, légères inférences acceptables
  0.4 — plusieurs affirmations non vérifiables dans le contexte
  0.0 — affirmations inventées absentes du contexte (hallucination)

Réponds UNIQUEMENT avec un JSON valide sans texte autour :
{"score": <float 0.0–1.0>, "reasoning": "<explication en 1-2 phrases>"}"""

_RELEVANCE_SYSTEM = """\
Tu es un évaluateur expert en qualité de réponses RAG.
Évalue si la réponse répond directement et précisément à la question posée.

Barème :
  1.0 — la réponse répond parfaitement à la question
  0.7 — la réponse répond partiellement, quelques aspects manquants
  0.4 — la réponse effleure la question sans vraiment y répondre
  0.0 — la réponse ne répond pas à la question

Réponds UNIQUEMENT avec un JSON valide sans texte autour :
{"score": <float 0.0–1.0>, "reasoning": "<explication en 1-2 phrases>"}"""

_COMPLETENESS_SYSTEM = """\
Tu es un évaluateur expert en qualité de réponses RAG.
Évalue si la réponse exploite toutes les informations pertinentes disponibles dans le contexte.

Barème :
  1.0 — la réponse utilise toutes les informations pertinentes du contexte
  0.7 — la réponse manque quelques informations mineures du contexte
  0.4 — la réponse ignore des informations importantes du contexte
  0.0 — la réponse n'exploite pas le contexte fourni

Réponds UNIQUEMENT avec un JSON valide sans texte autour :
{"score": <float 0.0–1.0>, "reasoning": "<explication en 1-2 phrases>"}"""

from src.entity.artifact_entity import JudgeResult, JudgeScore
from src.core.utils import parse_score, build_user_prompt

# ── Judge ─────────────────────────────────────────────────────────────────────

class RAGJudge:
    """
    Évalue une réponse RAG sur 3 dimensions via LLM-as-Judge.

    Chaque appel à evaluate() déclenche 3 appels LLM (un par critère).
    Les scores sont uploadés dans Langfuse si trace_id est fourni.
    """

    def _score_criterion(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         criterion: str) -> JudgeScore:
        
        try:

            llm = get_llm()
            raw = llm.generate_raw(system=system_prompt, user=user_prompt)
            return parse_score(raw, criterion)
        except Exception as exc:
            logging.warning(f"Judge '{criterion}' échoué (non bloquant) : {exc}")
            return JudgeScore(score=0.0, reasoning=str(exc))

    def evaluate(self,
        question: str,
        context_chunks: list[dict],
        answer: str,
        trace_id: str | None = None) -> JudgeResult:
        """Évalue la qualité d'une réponse RAG.

        Parameters
        ----------
        question       : question originale de l'utilisateur
        context_chunks : chunks retournés par le retrieval (clé 'content' requise)
        answer         : réponse générée par le LLM
        trace_id       : ID de trace Langfuse — si fourni, uploade les scores

        Returns
        -------
        JudgeResult avec scores faithfulness, relevance, completeness et overall
        """
        user_prompt = build_user_prompt(question, context_chunks, answer)

        faithfulness  = self._score_criterion(_FAITHFULNESS_SYSTEM, user_prompt, "faithfulness")
        relevance     = self._score_criterion(_RELEVANCE_SYSTEM,     user_prompt, "relevance")
        completeness  = self._score_criterion(_COMPLETENESS_SYSTEM,  user_prompt, "completeness")

        result = JudgeResult(
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
        )

        logging.info(
            f"LLM-as-Judge | faithfulness={faithfulness.score:.2f} "
            f"relevance={relevance.score:.2f} "
            f"completeness={completeness.score:.2f} "
            f"overall={result.overall:.2f}"
        )

        if trace_id:
            from monitoring.langfuse_eval import score_trace  # lazy — évite l'import circulaire
            score_trace(trace_id, "faithfulness",  faithfulness.score,  faithfulness.reasoning)
            score_trace(trace_id, "relevance",     relevance.score,     relevance.reasoning)
            score_trace(trace_id, "completeness",  completeness.score,  completeness.reasoning)
            score_trace(trace_id, "llm_judge_overall", result.overall)

        return result
