import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Config ---
COLLECTION_NAME = "thws_data_chunks"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL_NAME = "BAAI/bge-m3"
OLLAMA_MODEL = "gemma:7b"  # beste ergebnisse
# OLLAMA_MODEL = "zephyr"
# OLLAMA_MODEL = "mixtral"
TOP_K = 5

# --- Load Embedding Model with CUDA ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"
print(f"🔥 Using device: {device}")
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

# --- Init Qdrant Client ---
client = QdrantClient(url=QDRANT_URL)
while True:
    try:
        query = input("\n❓ Deine Frage (ENTER zum Beenden): ").strip()
        if not query:
            break

        # --- Embed Query ---
        query_vec = embedder.encode(query, device=device)

        # --- Search Qdrant ---
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec.tolist(),
            limit=TOP_K,
            with_payload=True,
        )

        # --- Deduplicate Chunks by Source ---
        unique_chunks = {}
        for res in search_results:
            src = res.payload["source"]
            if src not in unique_chunks:
                unique_chunks[src] = res

        context = "\n\n".join(res.payload["text"] for res in unique_chunks.values())

        # --- Build Prompt ---
        prompt = f"""
Du bist ein hilfreicher Assistent der Hochschule THWS.
Beantworte die folgende Frage basierend auf dem gegebenen Kontext.
Antworte ausschließlich auf Deutsch und fasse dich klar und präzise.
Wenn du es nicht weißt, sag "Ich weiß es leider nicht."

Kontext:
{context}

Frage:
{query}

Antwort:
"""

        # --- Call Ollama ---
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )

        answer = response.json().get("response")
        if answer:
            print("\n▸ ", answer.strip())
        else:
            print("\n⚠️ Keine Antwort vom Modell.")
            print("Raw response:", response.json())

        # --- Show Sources ---
        print("\n🔗 Quellen:")
        for src in sorted(unique_chunks.keys()):
            print("-", src)

    except KeyboardInterrupt:
        print("\n👋 Tschüss!")
        break
