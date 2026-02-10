"""
Axis Max Life Insurance Q&A API.
- Runs locally on FastAPI
- Uses Redis for conversation memory (session-scoped)
- Supports local LLM via Ollama
"""
import os
import json

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
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Axis Max Life Insurance Q&A</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      display: flex;
      min-height: 100vh;
      align-items: center;
      justify-content: center;
    }
    .shell {
      width: 100%;
      max-width: 960px;
      padding: 24px;
    }
    .card {
      background: #020617;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.75);
      padding: 20px 22px 18px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .title {
      font-size: 16px;
      font-weight: 600;
      color: #e5e7eb;
    }
    .subtitle {
      font-size: 12px;
      color: #9ca3af;
    }
    .pill {
      font-size: 11px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.5);
      color: #9ca3af;
      background: rgba(15, 23, 42, 0.85);
    }
    .input-area {
      margin-top: 8px;
      margin-bottom: 12px;
    }
    textarea {
      width: 100%;
      min-height: 80px;
      resize: vertical;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.5);
      background: #020617;
      color: #e5e7eb;
      padding: 10px 11px;
      font-size: 13px;
      line-height: 1.5;
      outline: none;
    }
    textarea:focus {
      border-color: #38bdf8;
      box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.35);
    }
    .actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 8px;
      gap: 10px;
    }
    .hint {
      font-size: 11px;
      color: #6b7280;
    }
    button {
      border-radius: 999px;
      border: none;
      padding: 7px 16px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      background: #38bdf8;
      color: #0f172a;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      box-shadow: 0 12px 25px rgba(56, 189, 248, 0.25);
    }
    button:disabled {
      opacity: 0.6;
      cursor: default;
      box-shadow: none;
    }
    .answer-card {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid rgba(31, 41, 55, 0.9);
    }
    .answer-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #6b7280;
      margin-bottom: 6px;
    }
    .answer {
      font-size: 13px;
      line-height: 1.6;
      color: #e5e7eb;
      white-space: pre-wrap;
    }
    .sources {
      margin-top: 8px;
      font-size: 11px;
      color: #9ca3af;
    }
    .sources ul {
      margin: 4px 0 0;
      padding-left: 18px;
    }
    .status {
      margin-top: 6px;
      font-size: 11px;
      color: #9ca3af;
    }
    a {
      color: #38bdf8;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="card">
      <div class="header">
        <div>
          <div class="title">Axis Max Life Insurance Q&A</div>
          <div class="subtitle">Ask questions about the indexed Axis Max Life prospectuses.</div>
        </div>
        <div class="pill">Local · FastAPI · Redis · Ollama</div>
      </div>

      <div class="input-area">
        <textarea id="question" placeholder="Ask anything about the STAR ULIP plan, death benefits, charges, etc."></textarea>
        <div class="actions">
          <div class="hint">Press “Ask” to send. Session is kept in this browser only.</div>
          <button id="askBtn" type="button">
            <span>Ask</span>
          </button>
        </div>
      </div>

      <div class="answer-card">
        <div class="answer-label">Answer</div>
        <div id="answer" class="answer">No question asked yet.</div>
        <div id="sources" class="sources"></div>
        <div id="status" class="status"></div>
      </div>

      <div class="status" style="margin-top: 10px;">
        API docs: <a href="/docs" target="_blank" rel="noreferrer">/docs</a> · Health: <code>/health</code>
      </div>
    </div>
  </div>

  <script>
    (function () {
      const questionEl = document.getElementById("question");
      const askBtn = document.getElementById("askBtn");
      const answerEl = document.getElementById("answer");
      const sourcesEl = document.getElementById("sources");
      const statusEl = document.getElementById("status");

      const SESSION_KEY = "axis-maxlife-session-id";
      let sessionId = window.localStorage.getItem(SESSION_KEY);
      if (!sessionId) {
        sessionId = "sess-" + Math.random().toString(36).slice(2, 10);
        window.localStorage.setItem(SESSION_KEY, sessionId);
      }

      async function sendQuery() {
        const q = questionEl.value.trim();
        if (!q) return;
        askBtn.disabled = true;
        answerEl.textContent = "Thinking…";
        sourcesEl.innerHTML = "";
        statusEl.textContent = "";

        try {
          const res = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: q, session_id: sessionId }),
          });

          const text = await res.text();
          let data;
          try {
            data = JSON.parse(text);
          } catch {
            throw new Error(text || ("Unexpected response with status " + res.status));
          }

          if (!res.ok) {
            answerEl.textContent = data.detail || ("Error " + res.status);
            return;
          }

          answerEl.textContent = data.answer || "";

          if (data.sources && data.sources.length) {
            const cleanSources = data.sources.filter(Boolean);
            if (cleanSources.length) {
              const items = cleanSources
                .map((s) => "<li>" + s + "</li>")
                .join("");
              sourcesEl.innerHTML = "Sources:<ul>" + items + "</ul>";
            } else {
              sourcesEl.innerHTML = "";
            }
          } else {
            sourcesEl.innerHTML = "";
          }
        } catch (err) {
          console.error(err);
          answerEl.textContent = "There was an error talking to the API.";
          statusEl.textContent = String(err && err.message ? err.message : err);
        } finally {
          askBtn.disabled = false;
        }
      }

      askBtn.addEventListener("click", sendQuery);
      questionEl.addEventListener("keydown", function (ev) {
        if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
          ev.preventDefault();
          sendQuery();
        }
      });
    })();
  </script>
</body>
</html>
    """


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
