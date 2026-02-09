# Axis Max Life Insurance Q&A (LangChain + FastAPI)

RAG API over Axis Max Life PDF prospectuses. Runs locally on macOS with FastAPI, Redis conversation memory, and optional local LLM via Ollama.

## Prerequisites

- Python 3.10+
- **Redis** running locally: `brew services start redis` (or `redis-server`)
- **Ollama** (optional, for local LLM): [ollama.ai](https://ollama.ai), then `ollama pull llama2` (or your `OLLAMA_MODEL`)

## Environment

In `.zshrc` (or `.env` in the project root):

```bash
export LANGCHAIN_EMBEDDING_PROVIDER=sentence_transformers
export VECTOR_STORE_PATH=./vector_db
export REDIS_URL=redis://localhost:6379
export OLLAMA_MODEL=llama2
```

## Install

```bash
cd langchain-fastapi-app
python -m venv rag-app-venv
source rag-app-venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Ingest PDFs

Place prospectus PDFs in `data/` and run ingestion once:

```bash
python -m app.ingest_policies
```

Or via API after starting the server:

```bash
curl -X POST http://localhost:8000/ingest
```

## Run the API

From the project root:

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

## Endpoints

| Method | Path   | Description |
|--------|--------|-------------|
| GET    | `/`    | App info and links |
| GET    | `/health` | Health check |
| POST   | `/query`  | Ask a question (optional `session_id` for Redis conversation memory) |
| POST   | `/ingest` | Re-run PDF ingestion into the vector store |

### Query examples

Without conversation memory:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the STAR ULIP plan?"}'
```

With conversation memory (Redis, same `session_id` for a thread):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the charges?", "session_id": "user-123"}'
```

## Data

Expected PDFs in `data/`:

- `STAR-ULIP-Prospectus.pdf`
- `max-life-guaranteed-income-plan-leaflet.pdf`
- `max-life-smart-value-income-benefit-enhancer.pdf`
