# File: knowledgeMapper/retrieval.py
# Description: v6.7 - FINAL. Correctly parses the structured context string,
#              reranks only the document chunks, and rebuilds the context for the LLM.

from datetime import datetime
from typing import List, Dict, Any, Union
import json
import re
from lightrag import LightRAG
from lightrag.base import QueryParam
from knowledgeMapper.utils.local_models import Reranker

MODE = "mix"
RERANKER_TOP_K = 5  # Number of documents to use after reranking

# System prompt remains the same
RELIABLE_SYSTEM_PROMPT_TEMPLATE = """
Du bist ein hilfreicher Assistent der Hochschule THWS.
Beantworte die folgende Frage basierend auf dem gegebenen Kontext.
Antworte ausschließlich auf Deutsch und fasse dich klar und präzise.
Wenn du die Antwort im Kontext nicht finden kannst, sage "Ich weiß es leider nicht."

Kontext:
{context}

Frage:
{query}

Antwort:
"""

def _parse_context_string(context_str: str) -> Dict[str, List[Dict]]:
    """
    Parses the combined context string into its constituent parts.
    """
    parsed_data = {
        "entities": [],
        "relationships": [],
        "doc_chunks": []
    }
    try:
        # Use regex to find all json blocks
        entities_match = re.search(r"-----Entities\(KG\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
        if entities_match:
            parsed_data["entities"] = json.loads(entities_match.group(1))

        relationships_match = re.search(r"-----Relationships\(KG\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
        if relationships_match:
            parsed_data["relationships"] = json.loads(relationships_match.group(1))

        doc_chunks_match = re.search(r"-----Document Chunks\(DC\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
        if doc_chunks_match:
            parsed_data["doc_chunks"] = json.loads(doc_chunks_match.group(1))

    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing context string: {e}")
        return None
    return parsed_data

def _rebuild_context_string(entities: List[Dict], relationships: List[Dict], doc_chunks: List[Dict]) -> str:
    """
    Reconstructs the context string from its component parts.
    """
    context_parts = []
    if entities:
        context_parts.append(f"-----Entities(KG)-----\n\n```json\n{json.dumps(entities, indent=2)}\n```")
    if relationships:
        context_parts.append(f"-----Relationships(KG)-----\n\n```json\n{json.dumps(relationships, indent=2)}\n```")
    if doc_chunks:
        context_parts.append(f"-----Document Chunks(DC)-----\n\n```json\n{json.dumps(doc_chunks, indent=2)}\n```")

    return "\n\n".join(context_parts)


async def prepare_and_execute_retrieval(
        user_query: str,
        rag_instance: LightRAG,
) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
    """
    Orchestrates a reliable RAG process that parses context, reranks document chunks,
    and returns a clean answer and structured sources.
    """
    params_bypass = QueryParam(mode="bypass", top_k=0)

    # Use a large top_k to get enough documents for the reranker to work with.
    # We need the full context string as it is the only way to get all data.
    params_context = QueryParam(
        mode=MODE,
        top_k=20,
        only_need_context=True
    )

    print("1. Retrieving initial combined context string...")
    initial_context_str = await rag_instance.aquery(user_query, param=params_context)

    if not initial_context_str:
        return {"answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage finden.", "sources": []}

    print("2. Parsing the context string...")
    parsed_context = _parse_context_string(initial_context_str)

    if not parsed_context or not parsed_context.get("doc_chunks"):
        return {"answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage finden.", "sources": []}

    # Rerank only the document chunks
    doc_chunks = parsed_context["doc_chunks"]
    documents_to_rerank = [chunk.get("content", "") for chunk in doc_chunks]

    print(f"3. Reranking {len(documents_to_rerank)} document chunks...")
    reranker = Reranker()
    reranked_indices = reranker.rerank(user_query, documents_to_rerank)

    # Create a new list of document chunks, sorted by relevance and trimmed
    reranked_chunks = [doc_chunks[i] for i in reranked_indices[:RERANKER_TOP_K]]

    print("4. Rebuilding context with reranked documents...")
    reranked_context_str = _rebuild_context_string(
        entities=parsed_context["entities"],
        relationships=parsed_context["relationships"],
        doc_chunks=reranked_chunks
    )

    print("5. Generating final answer with refined context...")
    final_system_prompt = RELIABLE_SYSTEM_PROMPT_TEMPLATE.format(
        context=reranked_context_str,
        user_query=user_query
    )

    citable_answer_text = await rag_instance.aquery(
        user_query,
        param=params_bypass,
        system_prompt=final_system_prompt
    )

    return {
        "answer": citable_answer_text,
        "sources": reranked_chunks  # Return the reranked and trimmed chunks as sources
    }