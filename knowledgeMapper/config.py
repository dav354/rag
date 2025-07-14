import os
from pathlib import Path

# Language selection for filtering documents (used in mongo_loader or processing)
LANGUAGE = os.getenv("LANGUAGE", "de").lower()  # 'all', 'de', or 'en'

# Directory for all vector/graph storage
BASE_STORAGE_DIR = Path("../RAG_STORAGE")

# MongoDB connection config
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_DB_NAME = "askthws_scraper"
MONGO_USER = os.getenv("MONGO_USER", "scraper")
MONGO_PASS = os.getenv("MONGO_PASS", "password")
MONGO_PAGES_COLLECTION = "pages"
MONGO_FILES_COLLECTION = "files"
MONGO_EXTRACTED_CONTENT_COLLECTION = "extracted_content"

# Embedding model settings
EMBEDDING_MODEL_NAME = "aari1995/German_Semantic_V3"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_CONCURRENCY = 1  # Controls number of concurrent embedding jobs

# LLM configuration (e.g., for Ollama server)
OLLAMA_MODEL_NAME = "gemma3:4B"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_NUM_CTX = 16384
OLLAMA_NUM_PREDICT = 4096

# Controls LightRAG's entity extraction feature (0 disables it)
ENTITY_EXTRACT_MAX_GLEANING = 1

RERANKER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-xsmall-v1"