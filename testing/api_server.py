# File: api_server.py
# DEBUGGING VERSION: Hardcoding environment variables to test Neo4j connection.

import time
import torch
import subprocess
import atexit
import os
import signal
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# ==============================================================================
# TEMPORARY DEBUGGING STEP
# We are setting the environment variables directly in the code to bypass any
# potential issues with the .env file.
#
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Custom Module Imports (adapted for new local_models.py) ---
from knowledgeMapper.utils.local_models import (
    HFEmbedFunc,
    OllamaLLM,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL_NAME,
)

# Import the updated retrieval logic
from knowledgeMapper.retrieval import (prepare_and_execute_retrieval, MODE)

# --- LightRAG Library Imports ---
import lightrag
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status

# --- Device Info ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"🔥 Using device: {device}")


# --- Lifespan to load all models and initialize LightRAG ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup events for the FastAPI application."""
    print("🚀 Server starting up...")
    print("🧠 Initializing LightRAG framework...")
    app.state.rag = LightRAG(
        working_dir="../RAG_STORAGE",
        embedding_func=HFEmbedFunc(),
        llm_model_func=OllamaLLM(),
        enable_llm_cache=False,
    )

    await app.state.rag.initialize_storages()
    print("✅ LightRAG storages initialized.")
    await initialize_pipeline_status()
    print("✅ LightRAG pipeline status initialized.")
    print("✅ Server is ready to accept requests.")
    yield
    print("🔌 Server shutting down.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="THWS KG-RAG API (Final Architecture)",
    description="Ein API-Server, der die stabile `aquery`-Methode mit einem intelligenten Prompt für maximale Antwortqualität und Transparenz verwendet.",
    version="18.0.3_debug",  # Version bumped for debug
    lifespan=lifespan
)

# ⬇️ NEU: CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # oder ["*"] zum Testen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str


# --- Ollama Background Server Management ---
print("🚓 Starting Ollama server in the background...")
ollama_process = subprocess.Popen(
    ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    preexec_fn=os.setsid if os.name != 'nt' else None
)


@atexit.register
def shutdown_ollama():
    """Function to gracefully shut down the Ollama server process."""
    print("Shutting down Ollama server...")
    if ollama_process:
        try:
            if os.name == 'nt':
                ollama_process.terminate()
            else:
                os.killpg(os.getpgid(ollama_process.pid), signal.SIGTERM)
            ollama_process.wait(timeout=5)
            print("🚓 Ollama server stopped successfully.")
        except Exception as e:
            print(f"Could not stop Ollama server gracefully: {e}")


# --- API Endpoints ---
@app.post("/ask")
async def ask(data: Question, request: Request):
    """
    Implements a controlled query pipeline by delegating to the retrieval module.
    """
    start_time = time.time()
    print(f"\n--- New Request ---")
    print(f"Received German query: '{data.query}'")
    try:
        rag: LightRAG = request.app.state.rag
        # Instantiate the LLM for this request. It's a lightweight wrapper.
        llm_instance = OllamaLLM()

        # Delegate the entire logic to the retrieval function
        final_answer = await prepare_and_execute_retrieval(
            user_query=data.query,
            rag_instance=rag,
        )

        duration = round(time.time() - start_time, 2)
        print(f"--- Request completed in {duration} seconds. ---")

        return {
            "question": data.query,
            "answer": final_answer,
            "mode": "controlled_aquery_pipeline",
            "duration_seconds": duration,
        }
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error. Details: {e}")


@app.get("/")
def read_root():
    """Root endpoint providing basic information about the API."""
    return {"message": "Welcome to the THWS KG-RAG API (Final Architecture)."}


@app.get("/metadata")
def metadata(request: Request):
    """Provides metadata about the running service."""
    rag: LightRAG = request.app.state.rag
    retriever_info = rag.vector_storage

    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": OLLAMA_MODEL_NAME,
        "device": device,
        "retrieval_mode": MODE,
    }


# --- Run FastAPI Server ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
