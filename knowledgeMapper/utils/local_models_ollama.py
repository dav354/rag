from __future__ import annotations
import asyncio
import requests
import torch
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from knowledgeMapper.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CONCURRENCY,
    OLLAMA_MODEL_NAME,
    OLLAMA_HOST,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    RERANKER_MODEL_NAME,
    # Added for Gemini Integration
    GEMINI_API_KEY,
    GEMINI_API_URL,
    GEMINI_MODEL_NAME,
)

# --- Device selection: prefer CUDA, then Apple MPS, else CPU ---
if torch.cuda.is_available():
    _DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"

# Semaphore to throttle concurrency of embedding requests (avoids OOM)
_EMBED_SEMAPHORE = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

# HuggingFace embeddings wrapper using LangChain's integration
_hf = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},  # Ensure unit-length vectors
    model_kwargs={"device": _DEVICE},  # auto: cuda -> mps -> cpu
)

# Calculate and expose the dimensionality of the embedding space
EMBED_DIM = len(_hf.embed_query("test"))


class AsyncEmbedder:
    """
    Async-compatible, memory-safe wrapper for HuggingFace embedding generation.
    Uses:
    - semaphore to control parallelism,
    - `to_thread()` to move blocking code out of the main event loop,
    - torch.no_grad() and empty_cache() to reduce GPU pressure.
    """

    embedding_dim: int = EMBED_DIM

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        async with _EMBED_SEMAPHORE:
            return await asyncio.to_thread(self._embed_chunked, texts)

    def _embed_chunked(self, texts: list[str]) -> list[list[float]]:
        # Split into batches and embed each chunk
        vecs: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            with torch.no_grad():  # No gradients needed for inference
                vecs.extend(_hf.embed_documents(batch))
            # Free accelerator memory after each batch (helps with OOM)
            try:
                if _DEVICE == "cuda":
                    torch.cuda.empty_cache()
                elif _DEVICE == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
            except Exception:
                pass
        return vecs


_async_embedder_instance = AsyncEmbedder()


async def embedding_wrapper_func(texts: list[str]) -> list[list[float]]:
    """Embedding API used by LightRAG (callable + exposes .embedding_dim)."""
    return await _async_embedder_instance(texts)


embedding_wrapper_func.embedding_dim = _async_embedder_instance.embedding_dim

# Exported function used by the rest of the app
embedding_func = embedding_wrapper_func


class Reranker:
    """
    A wrapper for the mxbai-rerank-xsmall-v1 model using sentence-transformers.
    """
    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: str = _DEVICE):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        """
        Reranks a list of documents based on a query.

        Args:
            query: The user's query.
            documents: A list of documents to be reranked.

        Returns:
            A list of indices representing the reranked order of the documents.
        """
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


class OllamaLLM:
    """
    Async LLM-Client für den lokalen Ollama-Server.
    Nutzt primär POST /api/chat (stream=false); Fallback auf /api/generate.
    """
    def __init__(self, host: str | None = None, model: str | None = None):
        from knowledgeMapper.config import (
            OLLAMA_HOST,
            OLLAMA_MODEL_NAME,
            OLLAMA_NUM_CTX,
            OLLAMA_NUM_PREDICT,
        )
        self.host = (host or OLLAMA_HOST).rstrip("/")
        self.model = model or OLLAMA_MODEL_NAME
        self.num_ctx = OLLAMA_NUM_CTX
        self.num_predict = OLLAMA_NUM_PREDICT

    def _build_messages(self, prompt: str,
                        system_prompt: str | None,
                        history_messages: list[dict] | None) -> list[dict]:
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if history_messages:
            for m in history_messages:
                role = m.get("role") or "user"
                content = m.get("content", "")
                if content:
                    msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _chat_call(self, messages: list[dict]) -> str:
        import requests

        # 1) Neues Chat-API
        url_chat = f"{self.host}/api/chat"
        payload_chat = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        try:
            r = requests.post(url_chat, json=payload_chat, timeout=120)
            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content", "")
            if content:
                return content.strip()
        except Exception:
            pass  # Fallback versuchen

        # 2) Fallback: altes /api/generate
        url_gen = f"{self.host}/api/generate"
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            tag = "SYSTEM" if role == "system" else ("USER" if role == "user" else "ASSISTANT")
            parts.append(f"[{tag}]\n{content}\n")
        full_prompt = "\n".join(parts)
        payload_gen = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        try:
            r = requests.post(url_gen, json=payload_gen, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            return f"Ollama API Error: {e}"

    async def __call__(self, prompt: str,
                       system_prompt: str | None = None,
                       history_messages: list[dict] | None = None,
                       **kwargs) -> str:
        import asyncio
        messages = self._build_messages(prompt, system_prompt, history_messages)
        return await asyncio.to_thread(self._chat_call, messages)


class HFEmbedFunc:
    """Legacy alias to maintain compatibility with older imports."""

    def __new__(cls, *_, **__):
        return embedding_func


__all__ = [
    "embedding_func",
    "OllamaLLM",
    "HFEmbedFunc",
    "Reranker",
    "EMBEDDING_MODEL_NAME",
    "OLLAMA_MODEL_NAME",
    "_DEVICE",
]
