"""
Axis Max Life Insurance Q&A API.
- Runs locally on FastAPI
- Uses Redis for conversation memory (session-scoped)
- Supports local LLM via Ollama
"""
import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.rag_chain import build_rag_chain
from app.ingest_policies import ingest

app = FastAPI(
    title="Axis Max Life Insurance Q&A",
    description="RAG API over Axis Max Life PDF prospectuses with Redis conversation memory and Ollama LLM.",
)

# Lazy-init chain on first request to avoid loading models at import time
_chain = None

def _get_chain():
    global _chain
    if _chain is None:
        _chain = build_rag_chain()
    return _chain


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _get_chat_history_text(session_id: str | None) -> str:
    """Format last 10 turns from Redis for the RAG prompt."""
    if not session_id:
        return ""
    try:
        from langchain_community.chat_message_histories import RedisChatMessageHistory
        history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
        messages = history.messages[-20:]  # last 20 messages (10 turns)
        parts = []
        for m in messages:
            role = "User" if m.type == "human" else "Assistant"
            content = getattr(m, "content", str(m))
            parts.append(f"{role}: {content}")
        return "\n".join(parts) if parts else ""
    except Exception:
        return ""


class Query(BaseModel):
    question: str
    session_id: str | None = None  # optional; when set, uses Redis conversation memory


class IngestResponse(BaseModel):
    status: str
    message: str


@app.get("/")
def root():
    return {
        "app": "Axis Max Life Insurance Q&A",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
        "ingest": "POST /ingest",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def ask(q: Query):
    chain = _get_chain()
    chat_history = _get_chat_history_text(q.session_id)

    # Optional response cache keyed by question (and session if provided)
    cache_key = f"qa:{q.session_id or 'anon'}:{q.question}"
    try:
        import redis
        cache = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        if cached := cache.get(cache_key):
            out = json.loads(cached)
            # Still append to conversation memory when we have a session
            if q.session_id:
                _append_to_history(q.session_id, q.question, out["answer"])
            return out
    except Exception:
        cache = None

    response = chain(q.question, chat_history=chat_history)
    answer = response["result"]
    sources = [doc.metadata.get("source", "") for doc in response["source_documents"]]
    result = {"answer": answer, "sources": sources}

    if cache:
        try:
            cache.set(cache_key, json.dumps(result), ex=3600)
        except Exception:
            pass

    if q.session_id:
        _append_to_history(q.session_id, q.question, answer)

    return result


def _append_to_history(session_id: str, question: str, answer: str):
    try:
        from langchain_community.chat_message_histories import RedisChatMessageHistory
        history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
        history.add_user_message(question)
        history.add_ai_message(answer)
    except Exception:
        pass


@app.post("/ingest", response_model=IngestResponse)
def run_ingest():
    """Ingest PDF prospectuses from data/ into the vector store."""
    try:
        ingest()
        return IngestResponse(status="ok", message="Ingestion completed.")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
