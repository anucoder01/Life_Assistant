# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Groq API (FREE) ───────────────────────────────────────
# Get free key at: https://console.groq.com
# Windows: $env:GROQ_API_KEY="gsk_..."
# Mac/Linux: export GROQ_API_KEY="gsk_..."
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"   # Free, fast, very capable
LLM_MAX_TOKENS = 1500

# ── Embedding Model ───────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Vector Database (ChromaDB) ────────────────────────────
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
CHROMA_COLLECTION = "life_assistant"

# ── Long-Term Memory (SQLite) ─────────────────────────────
MEMORY_DB_PATH = BASE_DIR / "data" / "memory.db"

# ── Face Recognition ──────────────────────────────────────
FACES_DIR = BASE_DIR / "data" / "faces"
FACE_TOLERANCE = 0.5

# ── Data Ingestion ────────────────────────────────────────
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────
TOP_K_RESULTS = 5

# ── API Server ────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Ensure directories exist ──────────────────────────────
for directory in [CHROMA_DIR, FACES_DIR, UPLOADS_DIR, BASE_DIR / "data"]:
    directory.mkdir(parents=True, exist_ok=True)