"""
Axis Max Life Insurance Q&A API.
- Runs locally on FastAPI
- Uses Redis for conversation memory (session-scoped)
- Supports local LLM via Ollama
"""
import os
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.rag_chain import build_rag_chain
from app.ingest_policies import ingest

try:
    from langchain_community.llms.ollama import OllamaEndpointNotFoundError
except ImportError:
    OllamaEndpointNotFoundError = Exception  # noqa: S001

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
BASE_DIR = Path(__file__).resolve().parent


def load_root_html() -> str:
    """Load the root HTML UI from a separate file."""
    html_path = BASE_DIR / "templates" / "index.html"
    try:
        return html_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Fallback minimal message if template is missing
        return "<html><body><h2>Axis Max Life Insurance Q&A</h2><p>Root UI template not found.</p></body></html>"


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


@app.get("/", response_class=HTMLResponse)
def root():
    """Simple, minimal chat-style UI over the /query API."""
    return load_root_html()


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

    try:
        response = chain(q.question, chat_history=chat_history)
    except OllamaEndpointNotFoundError as e:
        model = os.getenv("OLLAMA_MODEL", "llama2")
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama model not available. Ensure Ollama is running (e.g. run `ollama serve` or start the Ollama app) "
                f"and the model is pulled: `ollama pull {model}`. Original error: {e!s}"
            ),
        ) from e
    except Exception as e:
        err = str(e).lower()
        if "connection" in err or "refused" in err or "ollama" in err:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Ollama is not reachable. Start Ollama (e.g. run `ollama serve` or open the Ollama app), "
                    f"then pull your model: `ollama pull {os.getenv('OLLAMA_MODEL', 'llama2')}`. Error: {e!s}"
                ),
            ) from e
        raise

    answer = response["result"]
    # Make sources user-friendly: just file names, not absolute paths
    sources: list[str] = []
    for doc in response["source_documents"]:
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source", "")
        if src:
            src = os.path.basename(str(src))
        sources.append(src)
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
