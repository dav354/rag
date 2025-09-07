"""
File: knowledgeMapper/retrieval.py (drop-in)
Version: v7.1 — Low-Effort/High-Impact Retrieval Quality Upgrade for LightRAG (VDB + KG)

What’s new vs v7.0:
- Higher initial recall (INITIAL_TOP_K)
- Lightweight MMR-like diversification for docs & KG items (no new deps)
- Stronger KG re-ranking templates (neutral, attribute-rich)
- Consistent citation rules (links only, no internal IDs)
- Unified fallback strings
- Parameterized tunables at top for quick iteration

Note: Structure, prints, and overall flow match your existing style.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Union, Optional
import json
import re

from lightrag import LightRAG
from lightrag.base import QueryParam
from knowledgeMapper.utils.local_models import Reranker

# ==============================================
#                 T U N A B L E S
# ==============================================
MODE = "mix"  # LightRAG retrieval mode: "naive" or "mix"

# 1) Raise recall early; trim later via re-ranking + diversity
INITIAL_TOP_K = 60  # initial LightRAG top_k before re-ranking
RERANKER_TOP_K = 10  # final number of document chunks
RERANKER_TOP_K_KG = 12  # final number of KG items (entities + relationships)

# 2) Lightweight diversity (MMR-like without external libs)
MMR_LAMBDA = 0.7  # 1.0=only relevance, 0.0=only diversity
MMR_MAX_CANDIDATES = 50  # cap number of candidates for MMR diversification

RELIABLE_SYSTEM_PROMPT_TEMPLATE = """
**SYSTEMBEFEHL FÜR PRÄZISE WISSENSBASIERTE ANTWORTEN:**
1.  **SPRACHE:** Antworte **AUSSCHLIESSLICH** auf **DEUTSCH**.

2.  **INFORMATIONSEINSCHRÄNKUNG & KONTEXTTREUE:** Deine Antwort MUSS **VOLLSTÄNDIG** und **AUSSCHLIESSLICH** auf den Informationen im bereitgestellten `KONTEXT` basieren.
    * **KEINE Halluzinationen:** Generiere **KEINE** neuen Informationen, spekuliere **NICHT** und füge **NICHTS HINZU**, was nicht im `KONTEXT` explizit genannt ist. Dies beinhaltet auch die Verknüpfung von Informationen, die im Kontext nicht direkt miteinander verbunden sind (z.B. geografische Nähe ohne explizite Verbindungsbeschreibung).
    * **Umgang mit unbekannten Begriffen/fehlenden direkten Verknüpfungen:** Wenn der `KONTEXT` eine Abkürzung, Domänen-Slang, einen Fachbegriff oder eine Entität (z.B. einen spezifischen Gebäudecode wie "SHL") enthält, dessen Bedeutung oder deren direkte Beziehung zu anderen relevanten Konzepten (z.B. Buslinien) nicht direkt im `KONTEXT` erklärt oder verknüpft wird, **dann verwende den Begriff genau so, wie er im Kontext steht, und versuche NICHT, ihn zu erklären, zu interpretieren oder nicht explizit genannte Verbindungen herzustellen.** Halluziniere KEINE Bedeutungen, Annahmen oder implizite geografische oder funktionale Verknüpfungen.
    **Zitierungspflicht:** Füge am Ende jedes Satzes/Absatzes, der Informationen aus dem `KONTEXT` verwendet, den **Link der Quelle** in Klammern hinzu, z. B. (https://fiw.thws.de/studium/ihr-weg-durchs-studium/projektarbeit). Entferne dabei den abschließenden Schrägstrich /, falls der Link damit endet. **Nenne keine internen IDs.**
3.  **PRÄZISION & KONSISTENZ:**
    * Synthetisiere relevante Fakten aus den `Relationships(KG)`- und `Document Chunks(DC)`-Abschnitten zu einer **flüssigen, kohärenten und gut lesbaren Antwort**.
    * Wenn der `KONTEXT` widersprüchliche Informationen zu einem Thema enthält, gib **BEIDE Versionen an** und zitiere jeweils die **Links**.
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


# ==============================================
#           Context parsing / rebuilding
# ==============================================

def _parse_context_string(context_str: str) -> Optional[Dict[str, List[Dict]]]:
    """
    Parses the context string, intelligently handling structured 'mix' mode,
    plain JSON 'naive' mode, and the wrapped JSON format.
    Returns None on parse error.
    """
    is_structured_mode = '-----Entities(KG)-----' in context_str

    parsed_data: Dict[str, List[Dict]] = {
        "entities": [],
        "relationships": [],
        "doc_chunks": []
    }

    if is_structured_mode:
        print("   -> Parsing structured ('mix' mode) context.")
        try:
            entities_match = re.search(r"-----Entities\(KG\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
            if entities_match:
                parsed_data["entities"] = json.loads(entities_match.group(1))

            relationships_match = re.search(r"-----Relationships\(KG\)-----\s*```json\n(.*?)\n```", context_str,
                                            re.DOTALL)
            if relationships_match:
                parsed_data["relationships"] = json.loads(relationships_match.group(1))

            doc_chunks_match = re.search(r"-----Document Chunks\(DC\)-----\s*```json\n(.*?)\n```", context_str,
                                         re.DOTALL)
            if doc_chunks_match:
                parsed_data["doc_chunks"] = json.loads(doc_chunks_match.group(1))

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing structured context string: {e}")
            return None
    else:
        print("   -> Parsing 'naive' mode context.")
        try:
            json_match = re.search(r"```json\s*\n(.*?)\n```", context_str, re.DOTALL)
            if json_match:
                json_to_parse = json_match.group(1)
            else:
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
    json_kwargs = {"indent": 2, "ensure_ascii": False}

    context_parts = []
    if entities:
        context_parts.append(f"-----Entities(KG)-----\n\n```json\n{json.dumps(entities, **json_kwargs)}\n```")
    if relationships:
        context_parts.append(f"-----Relationships(KG)-----\n\n```json\n{json.dumps(relationships, **json_kwargs)}\n```")
    if doc_chunks:
        context_parts.append(f"-----Document Chunks(DC)-----\n\n```json\n{json.dumps(doc_chunks, **json_kwargs)}\n```")

    return "\n\n".join(context_parts)


# ==============================================
#        Lightweight MMR (diversity) helpers
# ==============================================

def _normalize_text(t: str) -> List[str]:
    if not t:
        return []
    t = re.sub(r"\s+", " ", t.lower()).strip()
    tokens = re.findall(r"[a-zäöüß0-9]+", t)
    ngrams = [" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2)] if len(tokens) >= 3 else [" ".join(tokens)]
    return ngrams


def _overlap_score(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union


def _mmr_select(candidate_indices: List[int], texts: List[str], base_order: List[int], top_k: int,
                lambda_mult: float) -> List[int]:
    """
    Very simple MMR variant:
    - 'base_order' approximates relevance (earlier = more relevant)
    - Diversity via n-gram overlap penalty
    """
    if top_k <= 0 or not candidate_indices:
        return []

    norm_texts = [_normalize_text(texts[i]) for i in range(len(texts))]
    selected: List[int] = []

    # relevance weight (inverse of position)
    rel_weight = {idx: 1.0 / (1 + base_order.index(idx)) if idx in base_order else 0.0 for idx in candidate_indices}

    while len(selected) < min(top_k, len(candidate_indices)):
        best_idx, best_score = None, -1.0
        for idx in candidate_indices:
            if idx in selected:
                continue
            max_sim = max((_overlap_score(norm_texts[idx], norm_texts[j]) for j in selected), default=0.0)
            score = lambda_mult * rel_weight.get(idx, 0.0) - (1 - lambda_mult) * max_sim
            if score > best_score:
                best_score, best_idx = score, idx
        if best_idx is None:
            break
        selected.append(best_idx)

    return selected


# ==============================================
#        Query optimization and main pipeline
# ==============================================


async def prepare_and_execute_retrieval(
        user_query: str,
        rag_instance: LightRAG,
) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
    """
    Orchestrates a reliable RAG process with parallel reranking for documents and KG items.
    Adds a preliminary LLM call that optimizes the query used for LightRAG retrieval.
    The final answer still uses the original user_query.
    """
    params_bypass = QueryParam(mode="bypass", top_k=0)
    # Higher recall; we trim later via re-ranking + MMR diversification
    params_context = QueryParam(mode=MODE, top_k=INITIAL_TOP_K, only_need_context_only=True) if hasattr(QueryParam,
                                                                                                        "only_need_context_only") else QueryParam(
        mode=MODE, top_k=INITIAL_TOP_K, only_need_context=True)

    # --- Step 1: retrieve combined context string ---
    print(f"1. Retrieving initial combined context string in '{MODE}' mode (top_k={INITIAL_TOP_K})...")
    initial_context_str = await rag_instance.aquery(user_query, param=params_context)
    if not initial_context_str:
        return {
            "answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden.",
            "sources": []}

    # --- Step 2: parse context ---
    print("2. Parsing the context string...")
    parsed_context = _parse_context_string(initial_context_str)
    if not parsed_context:
        return {
            "answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden.",
            "sources": []}

    reranker = Reranker()

    # --- STAGE 1: Re-rank Document Chunks (with light diversity) ---
    doc_chunks = parsed_context.get("doc_chunks", [])
    reranked_chunks: List[Dict[str, Any]] = []
    if doc_chunks:
        print(
            f"3a. Reranking {len(doc_chunks)} document chunks (MMR candidates={min(MMR_MAX_CANDIDATES, len(doc_chunks))}, final_k={RERANKER_TOP_K})...")
        doc_texts = [chunk.get("content", "") for chunk in doc_chunks]
        base_order = reranker.rerank(user_query, doc_texts)
        candidates = base_order[:min(MMR_MAX_CANDIDATES, len(base_order))]
        mmr_selected = _mmr_select(candidates, doc_texts, base_order, RERANKER_TOP_K, MMR_LAMBDA)
        reranked_chunks = [doc_chunks[i] for i in mmr_selected]

    # --- STAGE 2: Re-rank KG items (Entities & Relationships) with better templates + diversity ---
    kg_items: List[Dict[str, Any]] = []
    kg_texts: List[str] = []

    for entity in parsed_context.get("entities", []):
        item = entity.copy()
        item['__source_type'] = 'entity'
        kg_items.append(item)
        name = entity.get('name', '')
        etype = entity.get('type', '')
        attrs = []
        for k, v in entity.items():
            if k in ('name', 'type', '__source_type'):
                continue
            if isinstance(v, (str, int, float)) and len(str(v)) <= 120:
                attrs.append(f"{k}={v}")
        attr_str = "; ".join(attrs[:5])
        kg_texts.append(f"ENTITY | name={name} | type={etype}" + (f" | {attr_str}" if attr_str else ""))

    for rel in parsed_context.get("relationships", []):
        item = rel.copy()
        item['__source_type'] = 'relationship'
        kg_items.append(item)
        rtype = (rel.get('type', '') or '').replace('_', ' ')
        source = rel.get('source', '')
        target = rel.get('target', '')
        props = []
        for k, v in rel.items():
            if k in ('type', 'source', 'target', '__source_type'):
                continue
            if isinstance(v, (str, int, float)) and len(str(v)) <= 120:
                props.append(f"{k}={v}")
        prop_str = "; ".join(props[:5])
        kg_texts.append(f"RELATION | {source} -[{rtype}]-> {target}" + (f" | {prop_str}" if prop_str else ""))

    reranked_entities: List[Dict[str, Any]] = []
    reranked_relationships: List[Dict[str, Any]] = []

    if kg_items:
        print(
            f"3b. Reranking {len(kg_items)} KG items (MMR candidates={min(MMR_MAX_CANDIDATES, len(kg_items))}, final_k={RERANKER_TOP_K_KG})...")
        kg_base_order = reranker.rerank(user_query, kg_texts)
        kg_candidates = kg_base_order[:min(MMR_MAX_CANDIDATES, len(kg_base_order))]
        kg_mmr_selected = _mmr_select(kg_candidates, kg_texts, kg_base_order, RERANKER_TOP_K_KG, MMR_LAMBDA)
        top_kg_items = [kg_items[i] for i in kg_mmr_selected]

        reranked_entities = [item for item in top_kg_items if item['__source_type'] == 'entity']
        reranked_relationships = [item for item in top_kg_items if item['__source_type'] == 'relationship']
        for item in reranked_entities + reranked_relationships:
            if '__source_type' in item:
                del item['__source_type']

    # --- STAGE 3: Combine & rebuild context ---
    if not reranked_chunks and not reranked_entities and not reranked_relationships:
        return {
            "answer": 'Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden.',
            "sources": []}

    print("4. Rebuilding context with the best documents and KG items...")
    reranked_context_str = _rebuild_context_string(
        entities=reranked_entities,
        relationships=reranked_relationships,
        doc_chunks=reranked_chunks
    )

    # --- STAGE 4: Generate final answer ---
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

    final_sources = reranked_chunks + reranked_entities + reranked_relationships

    return {
        "answer": citable_answer_text,
        "sources": final_sources,
    }
