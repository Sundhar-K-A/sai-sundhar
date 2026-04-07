# ─────────────────────────────────────────────────────────────
# TASK 19 — Create a LangSmith Dataset
# ─────────────────────────────────────────────────────────────
"""
TASK 19: Create a LangSmith Dataset and Add Examples
------------------------------------------------------
Use the LangSmith SDK to:
  1. Create a dataset named "rag-eval-dataset".
  2. Add 3 question-answer example pairs to it.
  3. Return the dataset id as a string.

Examples to add:
  Q: "What does RAG stand for?"
     A: "Retrieval-Augmented Generation"
  Q: "What PostgreSQL extension enables vector search?"
     A: "pgvector"
  Q: "What LangChain tool provides observability?"
     A: "LangSmith"

HINT:
  from langsmith import Client
  client = Client()

  dataset = client.create_dataset("rag-eval-dataset")
  client.create_examples(
      inputs=[{"question": q} for q in questions],
      outputs=[{"answer": a} for a in answers],
      dataset_id=dataset.id
  )
"""

from langsmith import Client
from dotenv import load_dotenv

load_dotenv()


def create_langsmith_dataset() -> str:
    """Creates a LangSmith dataset with 3 Q-A examples. Returns dataset id."""
    client = Client()

    DATASET_NAME = "rag-eval-dataset"
    QA_PAIRS = [
        ("What does RAG stand for?", "Retrieval-Augmented Generation"),
        ("What PostgreSQL extension enables vector search?", "pgvector"),
        ("What LangChain tool provides observability?", "LangSmith"),
    ]

    dataset = client.create_dataset(DATASET_NAME)

    client.create_examples(
        inputs=[{"question": q} for q, a in QA_PAIRS],
        outputs=[{"answer": a} for q, a in QA_PAIRS],
        dataset_id=dataset.id,
    )

    return str(dataset.id)


dataset_id = create_langsmith_dataset()
print("Dataset ID:", dataset_id)