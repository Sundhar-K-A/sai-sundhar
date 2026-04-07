# ─────────────────────────────────────────────────────────────
# TASK 20 — Run an Evaluation with LangSmith
# ─────────────────────────────────────────────────────────────
"""
TASK 20: LangSmith Evaluation (evaluate)
------------------------------------------
Run an automated evaluation of your RAG pipeline using the
dataset created in Task 19.

Steps:
  1. Define a target function that takes a dict {"question": str}
     and returns {"answer": str} using the basic RAG pipeline.
  2. Define a custom evaluator that checks if the expected
     answer appears (case-insensitive) in the generated answer.
  3. Run the evaluation using langsmith.evaluate().
  4. Return the evaluation results summary dict:
     {"dataset": str, "num_examples": int, "pass_rate": float}

HINT:
  from langsmith.evaluation import evaluate, LangChainStringEvaluator

  def target(inputs: dict) -> dict:
      return {"answer": basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])}

  results = evaluate(
      target,
      data="rag-eval-dataset",
      evaluators=[...],
      experiment_prefix="rag-challenge-eval",
  )
"""
from dotenv import load_dotenv
from langsmith.evaluation import evaluate

load_dotenv()


def run_langsmith_evaluation() -> dict:
    """Evaluates the RAG pipeline on the LangSmith dataset."""

    RAG_DOCUMENTS = [
        "RAG stands for Retrieval-Augmented Generation.",
        "pgvector is a PostgreSQL extension for vector search.",
        "LangSmith provides observability tools for LLM applications."
    ]

    def basic_rag_pipeline(documents, question):
        for doc in documents:
            if any(word.lower() in doc.lower() for word in question.split()):
                return doc
        return documents[0]

    def target(inputs: dict) -> dict:
        return {"answer": basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])}

    def evaluator(run, example):
        pred = (run.outputs or {}).get("answer", "").lower()
        expected = (example.outputs or {}).get("answer", "").lower()
        return {"key": "substring_match", "score": int(expected in pred)}

    results = evaluate(
        target,
        data="rag-eval-dataset",
        evaluators=[evaluator],
        experiment_prefix="rag-challenge-eval",
    )

    total = 0
    passed = 0

    for r in results:
        total += 1
        eval_results = r["evaluation_results"]["results"]
        for ev in eval_results:
            if ev.key == "substring_match":
                passed += int(ev.score)

    pass_rate = passed / total if total > 0 else 0.0

    return {
        "dataset": "rag-eval-dataset",
        "num_examples": total,
        "pass_rate": pass_rate,
    }


if __name__ == "__main__":
    result = run_langsmith_evaluation()
    print(result)