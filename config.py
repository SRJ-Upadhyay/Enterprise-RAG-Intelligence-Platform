#Enterprise RAG Intelligence Platform - Config
import os
import warnings
import logging
from dotenv import load_dotenv
load_dotenv()

# 🔇 Clean console output (no warnings / progress bars)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# 🔑 Keys and paths
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/enterprise_rag")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
DOCS_DIR = "data/documents"
UPLOAD_DIR = "data/uploads"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🧠 Models
LLM_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "enterprise_docs"

# ✂️ Chunking and retrieval
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K_CANDIDATES = 10   # per retriever (vector + BM25)
RERANK_TOP_N = 6        # final chunks sent to the LLM

# 🔐 Access control : role -> access levels the role can read
ALLOWED_LEVELS = ["public", "internal", "confidential"]
ROLE_LEVELS = {
    "employee" : ["public", "internal"],
    "manager" : ["public", "internal", "confidential"],
    "admin" : ["public", "internal", "confidential"],
}
