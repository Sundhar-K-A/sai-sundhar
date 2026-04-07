import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain_postgres import PGVector

load_dotenv()

CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")

RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]


def rag_agent(question: str) -> str:
    """Uses an agent with a retriever tool to answer the question."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = [Document(page_content=text) for text in RAG_DOCUMENTS]

    store = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="agent_rag",
        connection=CONNECTION_STRING,
    )

    retriever = store.as_retriever(search_kwargs={"k": 3})

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="knowledge_base",
        description="Search the knowledge base for technical information.",
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_agent(
        model=model,
        tools=[retriever_tool],
        system_prompt=(
            "You are a helpful technical assistant. "
            "Use the knowledge_base tool when the answer depends on the provided documents. "
            "Answer concisely."
        ),
    )

    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    )

    return response["messages"][-1].content


if __name__ == "__main__":
    q = "What distance metrics does pgvector support?"
    answer = rag_agent(q)
    print("Question:", q)
    print("Answer:", answer)