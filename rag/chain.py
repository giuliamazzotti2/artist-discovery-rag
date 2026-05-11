"""RAG chain for artist discovery using LCEL (LangChain Expression Language)."""

import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
import chromadb

load_dotenv()

CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))

SYSTEM_PROMPT = (
    "You are a music discovery assistant. Use only the retrieved artist documents. "
    "Always list exactly 5 artists, no more and no less. "
    "Format each entry as: - **Name**: 1-2 sentence explanation of why they match. "
    "No intro, no conclusion, no filler."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Query: {question}\n\nContext:\n{context}"),
])


class SentenceTransformerEmbeddings:
    """Embed text using the local all-MiniLM-L6-v2 model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@lru_cache(maxsize=1)
def _build_chain():
    """Build and cache the LCEL retrieval chain."""
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not anthropic_api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or your shell environment."
        )

    model_name = os.getenv("CLAUDE_MODEL_NAME", "claude-haiku-4-5-20251001")
    llm = ChatAnthropic(model=model_name, temperature=0.2, anthropic_api_key=anthropic_api_key)

    if not os.path.isdir(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB index not found at '{CHROMA_DIR}'. "
            "Run 'python embeddings/build_index.py' to build it first."
        )

    embedding = SentenceTransformerEmbeddings()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    vectorstore = Chroma(client=client, collection_name="artists", embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def query(user_input: str) -> str:
    """Query the RAG chain and return a generated answer."""
    if not user_input or not user_input.strip():
        return "Please enter a descriptive music query to discover artists."
    try:
        chain = _build_chain()
        return chain.invoke(user_input)
    except Exception as error:
        return (
            "Unable to generate a response. Please check that the Chroma index exists "
            "and that ANTHROPIC_API_KEY is configured.\n"
            f"Details: {error}"
        )
