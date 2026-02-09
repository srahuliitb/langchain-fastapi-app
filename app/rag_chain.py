"""
RAG chain for Axis Max Life insurance Q&A.
Uses Chroma + HuggingFaceEmbeddings and supports optional conversation history.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from app.ollama_llm import get_local_llm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_db")
if not os.path.isabs(VECTOR_PATH):
    VECTOR_PATH = str(PROJECT_ROOT / VECTOR_PATH)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are a knowledgeable AI assistant focused on Indian insurance (Axis Max Life).
Answer ONLY based on the document passages below. If the answer is not supported by the indexed content, say "I don't know".

Relevant excerpts from the prospectuses:
{context}

Recent conversation (for context only):
{chat_history}

User question:
{question}

Answer:""",
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


def _get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=VECTOR_PATH,
        embedding_function=_get_embeddings(),
    )


def build_rag_chain():
    """Build the RAG chain (retriever + LLM). Used for backward compatibility and simple /query without history."""
    llm = get_local_llm()
    db = _get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 5})

    def _run(query: str, chat_history: str = "") -> dict:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        prompt = _PROMPT.format(
            context=context or "(No relevant passages found.)",
            question=query,
            chat_history=chat_history or "(None)",
        )
        result = llm.invoke(prompt)
        return {
            "result": result if isinstance(result, str) else getattr(result, "content", str(result)),
            "source_documents": docs,
        }

    return _run


def get_rag_runner():
    """Return the RAG runner function (question, chat_history) -> { result, source_documents }."""
    return build_rag_chain()
