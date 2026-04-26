import sys 


from src.core.logging import logging
from src.core.exception import AgenticRagException
from monitoring.langfuse_eval import run_rag_pipeline

from langfuse import Langfuse
from datasets import Dataset as HFDataset
from ragas import evaluate
from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall)



def evaluate_dataset(
    qa_pairs: list[dict],
    dataset_name: str = "agenticrag-eval") -> dict:
    """Évalue un jeu de Q&R avec RAGAS et uploade les scores dans Langfuse.

    Parameters
    ----------
    qa_pairs     : liste de {"question": str, "ground_truth": str}
    dataset_name : label affiché dans Langfuse

    Returns
    -------
    dict : scores RAGAS agrégés

    Example
    -------
    >>> from monitoring.eval_ragas import evaluate_dataset
    >>> evaluate_dataset([
    ...     {"question": "Quelle est la stratégie ?", "ground_truth": "..."}
    ... ])
    """

    questions, answers, contexts, ground_truths = [], [], [], []

    logging.info(f"Évaluation RAGAS — {len(qa_pairs)} paires Q&R")

    for pair in qa_pairs:
        result = run_rag_pipeline(pair["question"])
        questions.append(pair["question"])
        answers.append(result["answer"])
        contexts.append([c["content"] for c in result["context"]])
        ground_truths.append(pair["ground_truth"])

    hf_dataset = HFDataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths})

    scores = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    logging.info(f"RAGAS scores : {scores}")

    # Upload dans Langfuse
    try:
        
        client = Langfuse()
        for i, pair in enumerate(qa_pairs):
            trace = client.trace(
                name="ragas-eval",
                input={"question": pair["question"]},
                output={"answer": answers[i]},
                tags=["evaluation", dataset_name],
            )
            for metric_name, metric_values in scores.items():
                v = (
                    metric_values[i]
                    if hasattr(metric_values, "__getitem__")
                    else float(metric_values))

                client.score(trace_id=trace.id,
                              name=metric_name,
                                value=float(v))
        
        client.flush()
        logging.info(f"Scores uploadés dans Langfuse (dataset : {dataset_name})")
    
    except Exception as exc:
        logging.warning(f"Upload Langfuse échoué (non bloquant) : {exc}")
   
    return dict(scores)
