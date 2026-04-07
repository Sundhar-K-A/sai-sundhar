# ─────────────────────────────────────────────────────────────
# TASK 16 — Conversational RAG with Chat History
# ─────────────────────────────────────────────────────────────
"""
TASK 16: Conversational RAG
------------------------------
Build a RAG pipeline that is aware of conversation history.

Requirements:
  - Use create_history_aware_retriever to rephrase follow-up
    questions into standalone queries.
  - Use create_retrieval_chain + create_stuff_documents_chain
    to answer with context.
  - Run a 2-turn conversation:
      Turn 1: "What is LangChain?"
      Turn 2: "What version introduced LCEL?"  ← follow-up
  - Return both answers as a list: [answer1, answer2]

HINT:
  from langchain.chains import create_history_aware_retriever
  from langchain.chains import create_retrieval_chain
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain_core.messages import HumanMessage, AIMessage

  contextualize_prompt — asks the LLM to rephrase the question
                         given history.
  qa_prompt           — answers based on context + history.
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import PGVector

load_dotenv()

CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")

RAG_DOCUMENTS = [
    "LangChain is a framework for building applications powered by large language models (LLMs).",
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]


def _build_vectorstore(documents: list):
    """Build a PGVector vectorstore from a list of plain strings."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = [Document(page_content=text) for text in documents]
    return PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="conv_rag_collection",
        connection_string=CONNECTION_STRING,
    )


def conversational_rag(documents: list) -> list:
    """
    Runs a 2-turn history-aware RAG conversation.

    Turn 1 — "What is LangChain?"
    Turn 2 — "What version introduced LCEL?"  (follow-up that needs history)

    Returns:
        [answer_turn1, answer_turn2]
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    store = _build_vectorstore(documents)
    retriever = store.as_retriever(search_kwargs={"k": 3})

    contextualize_system_prompt = (
        "Given a chat history and the latest user question which might "
        "reference context in the chat history, formulate a standalone "
        "question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and "
        "otherwise return it as is."
    )
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history: list = []

    q1 = "What is LangChain?"
    response1 = rag_chain.invoke({"input": q1, "chat_history": chat_history})
    answer1 = response1["answer"]
    chat_history.extend([HumanMessage(content=q1), AIMessage(content=answer1)])

    q2 = "What version introduced LCEL?"
    response2 = rag_chain.invoke({"input": q2, "chat_history": chat_history})
    answer2 = response2["answer"]

    return [answer1, answer2]

if __name__ == "__main__":
    answers = conversational_rag(RAG_DOCUMENTS)
    print("Turn 1:", answers[0])
    print("Turn 2:", answers[1])
