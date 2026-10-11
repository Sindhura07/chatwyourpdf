"""Central configuration. Override anything via environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

FRONTEND_DIR = ROOT / "frontend"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Safety limits (protect your free Gemini quota on a public site)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "1500"))      # per visitor workspace
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))   # visitors kept in memory
