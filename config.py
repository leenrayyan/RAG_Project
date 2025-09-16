# config.py
from pathlib import Path

# Directories
PERSIST_DIR = str(Path("./chroma_db").resolve())
UPLOAD_DIR  = str(Path("./Uploads").resolve())

# Models (Ollama)
#   ollama pull llama3.1
#   ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.1"

# Chroma
COLLECTION_NAME = "docs"
