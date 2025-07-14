from __future__ import annotations
import asyncio
import requests
import torch
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
)

# Semaphore to throttle concurrency of embedding requests (avoids OOM)
_EMBED_SEMAPHORE = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

# HuggingFace embeddings wrapper using LangChain's integration
_hf = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},  # Ensure unit-length vectors
    model_kwargs={"device": EMBEDDING_DEVICE},  # e.g., "cuda" or "cpu"
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
            torch.cuda.empty_cache()  # Free VRAM after each batch (helps with OOM)
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
    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: str = EMBEDDING_DEVICE):
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
    Async wrapper around Ollama's local LLM endpoint (`/api/generate`).
    Suitable for fast interaction with locally running models.
    """

    async def __call__(
            self,
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict] | None = None,
            **kwargs,
    ) -> str:
        # Strip unused keys to avoid API incompatibilities
        for k in ("hashing_kv", "max_tokens", "response_format"):
            kwargs.pop(k, None)

        # Concatenate prompt components in proper order
        parts: list[str] = []
        if history_messages:
            parts.append("\n".join(m.get("content", "") for m in history_messages))
        if system_prompt:
            parts.append(system_prompt)
        parts.append(prompt)
        full_prompt = "\n".join(parts)

        # Actual HTTP call is made in a background thread to avoid blocking
        def _call() -> str:
            r = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": OLLAMA_MODEL_NAME,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": OLLAMA_NUM_CTX,
                        "num_predict": OLLAMA_NUM_PREDICT,
                    },
                },
                timeout=10_000,
            )
            r.raise_for_status()
            return r.json()["response"]

        return await asyncio.to_thread(_call)


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
]