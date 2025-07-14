# File: knowledgeMapper/retrieval.py
# Description: v6.8 - FINAL. Intelligently parses context from both 'naive' and 'mix' modes,
#              reranks only the document chunks, and rebuilds the context for the LLM.

from datetime import datetime
from typing import List, Dict, Any, Union
import json
import re
from lightrag import LightRAG
from lightrag.base import QueryParam
from knowledgeMapper.utils.local_models import Reranker

# You can now set this to "naive" or "mix" and the code will work
MODE = "mix"
RERANKER_TOP_K = 5 # Number of documents to use after reranking

RELIABLE_SYSTEM_PROMPT_TEMPLATE = """
**SYSTEMBEFEHL FÜR PRÄZISE WISSENSBASIERTE ANTWORTEN:**
1.  **SPRACHE:** Antworte **AUSSCHLIESSLICH** auf **DEUTSCH**.

2.  **INFORMATIONSEINSCHRÄNKUNG & KONTEXTTREUE:** Deine Antwort MUSS **VOLLSTÄNDIG** und **AUSSCHLIESSLICH** auf den Informationen im bereitgestellten `KONTEXT` basieren.
    * **KEINE Halluzinationen:** Generiere **KEINE** neuen Informationen, spekuliere **NICHT** und füge **NICHTS HINZU**, was nicht im `KONTEXT` explizit genannt ist. Dies beinhaltet auch die Verknüpfung von Informationen, die im Kontext nicht direkt miteinander verbunden sind (z.B. geografische Nähe ohne explizite Verbindungsbeschreibung).
    * **Umgang mit unbekannten Begriffen/fehlenden direkten Verknüpfungen:** Wenn der `KONTEXT` eine Abkürzung, Domänen-Slang, einen Fachbegriff oder eine Entität (z.B. einen spezifischen Gebäudecode wie "SHL") enthält, dessen Bedeutung oder deren direkte Beziehung zu anderen relevanten Konzepten (z.B. Buslinien) nicht direkt im `KONTEXT` erklärt oder verknüpft wird, **dann verwende den Begriff genau so, wie er im Kontext steht, und versuche NICHT, ihn zu erklären, zu interpretieren oder nicht explizit genannte Verbindungen herzustellen.** Halluziniere KEINE Bedeutungen, Annahmen oder implizite geografische oder funktionale Verknüpfungen.
    * **Zitierungspflicht:** Füge am Ende JEDES Satzes oder Absatzes, der Informationen aus dem `KONTEXT` verwendet, die **ID der Quelle in Klammern** hinzu, z.B. (DC-ID: 1) oder (KG-ID: 5). Wenn Informationen aus mehreren Quellen in einem Satz oder Absatz kombiniert werden, nenne alle relevanten IDs (DC-ID: 1, KG-ID: 5).

3.  **PRÄZISION & KONSISTENZ:**
    * Synthetisiere relevante Fakten aus den `Relationships(KG)`- und `Document Chunks(DC)`-Abschnitten zu einer **flüssigen, kohärenten und gut lesbaren Antwort**.
    * Wenn der `KONTEXT` widersprüchliche Informationen zu einem Thema enthält, gib **BEIDE Versionen an** und nenne die jeweiligen Quell-IDs.
    * Formuliere eine Aussagekräftige Antwort auf die gestellte Frage des Users!
    
4.  **FALLBACK-PROZEDERE:** Falls die `NUTZERFRAGE` **NICHT** oder **NICHT ausreichend** im `KONTEXT` beantwortet werden kann und **auch keine indirekten, zitierfähigen Informationen** (weder aus DC noch aus KG) vorhanden sind, antworte **AUSSCHLIESSLICH** und wortwörtlich mit:
    "Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden."
    Verändere diese Formulierung **NICHT**.

5.  **UNTERDRÜCKUNG VON PLAPPERN/ERGÄNZUNGEN:** Gehe direkt zur Antwort über. Vermeide einleitende Phrasen wie "Basierend auf dem Kontext..." oder abschließende Bemerkungen. Die Antwort soll **NUR** die Beantwortung der Nutzerfrage sein.

---
**ZUSATZDATEN:**
- Heutiges Datum: {current_date}
- Standort: {location}
---
**KONTEXT:**
{context}
**NUTZERFRAGE:**
{user_query}
"""

def _parse_context_string(context_str: str) -> Dict[str, List[Dict]]:
    """
    Parses the context string, intelligently handling structured 'mix' mode,
    plain JSON 'naive' mode, and the wrapped JSON format.
    """
    is_structured_mode = '-----Entities(KG)-----' in context_str

    parsed_data = {
        "entities": [],
        "relationships": [],
        "doc_chunks": []
    }

    if is_structured_mode:
        # This part remains the same, for handling the full "mix" mode output.
        print("   -> Parsing structured ('mix' mode) context.")
        try:
            # ... (original regex logic for entities, relationships, and doc_chunks)
            entities_match = re.search(r"-----Entities\(KG\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
            if entities_match:
                parsed_data["entities"] = json.loads(entities_match.group(1))
            #... etc. for relationships and doc_chunks
            relationships_match = re.search(r"-----Relationships\(KG\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
            if relationships_match:
                parsed_data["relationships"] = json.loads(relationships_match.group(1))

            doc_chunks_match = re.search(r"-----Document Chunks\(DC\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
            if doc_chunks_match:
                parsed_data["doc_chunks"] = json.loads(doc_chunks_match.group(1))

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing structured context string: {e}")
            return None
    else:
        # --- NEW, SMARTER NAIVE MODE LOGIC ---
        # Handles both plain JSON and the wrapped format you provided.
        print("   -> Parsing 'naive' mode context.")
        try:
            # First, try to find a JSON block wrapped in ```json ... ```
            json_match = re.search(r"```json\s*\n(.*?)\n```", context_str, re.DOTALL)
            if json_match:
                # If found, parse the content inside the block
                json_to_parse = json_match.group(1)
            else:
                # Otherwise, assume the whole string is the JSON content
                json_to_parse = context_str

            parsed_data["doc_chunks"] = json.loads(json_to_parse)

        except json.JSONDecodeError as e:
            print(f"Error parsing plain/wrapped JSON context string: {e}")
            return None

    return parsed_data

def _rebuild_context_string(entities: List[Dict], relationships: List[Dict], doc_chunks: List[Dict]) -> str:
    """
    Reconstructs the context string from its component parts, ensuring correct UTF-8 encoding.
    """
    # Define JSON settings to be used in all dumps calls
    # ensure_ascii=False allows special characters like ö, ä, ü directly in the output string.
    json_kwargs = {"indent": 2, "ensure_ascii": False}

    context_parts = []
    if entities:
        # The **json_kwargs unpacks the dictionary into arguments for the function
        context_parts.append(f"-----Entities(KG)-----\n\n```json\n{json.dumps(entities, **json_kwargs)}\n```")
    if relationships:
        context_parts.append(f"-----Relationships(KG)-----\n\n```json\n{json.dumps(relationships, **json_kwargs)}\n```")
    if doc_chunks:
        context_parts.append(f"-----Document Chunks(DC)-----\n\n```json\n{json.dumps(doc_chunks, **json_kwargs)}\n```")

    return "\n\n".join(context_parts)


async def prepare_and_execute_retrieval(
        user_query: str,
        rag_instance: LightRAG,
) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
    """
    Orchestrates a reliable RAG process with parallel reranking for documents and KG items.
    """
    params_bypass = QueryParam(mode="bypass", top_k=0)
    params_context = QueryParam(mode=MODE, top_k=20, only_need_context=True)

    print(f"1. Retrieving initial combined context string in '{MODE}' mode...")
    initial_context_str = await rag_instance.aquery(user_query, param=params_context)
    if not initial_context_str:
        return {"answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage finden.", "sources": []}

    print("2. Parsing the context string...")
    parsed_context = _parse_context_string(initial_context_str)
    if not parsed_context:
        return {"answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage finden.", "sources": []}

    reranker = Reranker()

    # --- STAGE 1: Rerank Document Chunks ---
    doc_chunks = parsed_context.get("doc_chunks", [])
    reranked_chunks = []
    if doc_chunks:
        print(f"3a. Reranking {len(doc_chunks)} document chunks...")
        doc_texts = [chunk.get("content", "") for chunk in doc_chunks]
        doc_indices = reranker.rerank(user_query, doc_texts)
        # Keep the top K documents (e.g., K=3)
        reranked_chunks = [doc_chunks[i] for i in doc_indices[:RERANKER_TOP_K]]

        # --- STAGE 2: Rerank Knowledge Graph Items (Entities & Relationships) ---
    kg_items = []
    kg_texts = []
    for entity in parsed_context.get("entities", []):
        item = entity.copy()
        item['__source_type'] = 'entity'
        kg_items.append(item)
        kg_texts.append(f"Entität: Der Ort '{entity.get('name', '')}' ist vom Typ '{entity.get('type', '')}'.")

    for rel in parsed_context.get("relationships", []):
        item = rel.copy()
        item['__source_type'] = 'relationship'
        kg_items.append(item)
        kg_texts.append(f"Beziehung: '{rel.get('source', '')}' hat die Beziehung '{rel.get('type', '').replace('_', ' ')}' zu '{rel.get('target', '')}'.")

    reranked_entities = []
    reranked_relationships = []
    if kg_items:
        print(f"3b. Reranking {len(kg_items)} KG items...")
        kg_indices = reranker.rerank(user_query, kg_texts)
        # Keep the top K KG items (e.g., K=3)
        top_kg_items = [kg_items[i] for i in kg_indices[:RERANKER_TOP_K]]

        # Separate them back out
        reranked_entities = [item for item in top_kg_items if item['__source_type'] == 'entity']
        reranked_relationships = [item for item in top_kg_items if item['__source_type'] == 'relationship']
        # Clean up temporary key
        for item in reranked_entities + reranked_relationships:
            del item['__source_type']

    # --- STAGE 3: Combine and Rebuild Context ---
    # Check if we found anything at all after reranking
    if not reranked_chunks and not reranked_entities and not reranked_relationships:
        return {"answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden.", "sources": []}

    print("4. Rebuilding context with the best documents and KG items...")
    reranked_context_str = _rebuild_context_string(
        entities=reranked_entities,
        relationships=reranked_relationships,
        doc_chunks=reranked_chunks
    )

    print("5. Generating final answer with refined context...")
    final_system_prompt = RELIABLE_SYSTEM_PROMPT_TEMPLATE.format(
        current_date=datetime.now().strftime("%d. %B %Y"),
        location="Würzburg",
        context=reranked_context_str,
        user_query=user_query
    )

    citable_answer_text = await rag_instance.aquery(
        user_query,
        param=params_bypass,
        system_prompt=final_system_prompt
    )

    # Combine all top sources for the final output
    final_sources = reranked_chunks + reranked_entities + reranked_relationships

    return {
        "answer": citable_answer_text,
        "sources": final_sources,
    }