"""
Ingest Axis Max Life insurance PDF prospectuses into the vector store.
Uses PyPDFLoader (pypdf) and SentenceTransformer embeddings via HuggingFaceEmbeddings.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # or: from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Paths relative to project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_db")
# Resolve to absolute so it works regardless of cwd
if not os.path.isabs(VECTOR_PATH):
    VECTOR_PATH = str(PROJECT_ROOT / VECTOR_PATH)

# Match actual filenames in data/
PDF_FILES = [
    "STAR-ULIP-Prospectus.pdf",
    "max-life-guaranteed-income-plan-leaflet.pdf",
    "max-life-smart-value-income-benefit-enhancer.pdf",
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


def ingest():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path(VECTOR_PATH).mkdir(parents=True, exist_ok=True)

    docs = []
    for pdf in PDF_FILES:
        path = DATA_DIR / pdf
        if not path.exists():
            print(f"[!] Skipping {pdf} (not found at {path})")
            continue
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())

    if not docs:
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR}. Expected files: {PDF_FILES}"
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=VECTOR_PATH,
    )
    db.persist()
    print(f"[✔] Indexed {len(chunks)} chunks into {VECTOR_PATH}")


if __name__ == "__main__":
    ingest()
