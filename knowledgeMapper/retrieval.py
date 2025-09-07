"""
File: knowledgeMapper/retrieval.py (drop-in)
Version: v7.3 — Retrieval Upgrade (LightRAG + Lexical Pre-Search)

Adds:
- Step 0: Lightweight keyword/FTS pre-search over all chunks (no extra deps)
- Inject 1–2 lexical chunks before LightRAG retrieval flow
- Auto-build SQLite FTS5 index from storage artifacts
- Robust JSON parsing incl. dict-mapped chunks {"chunk-...": {...}}
- NEW: W-question stopwords + relaxed FTS query with OR groups for contact terms

Storage:
- Uses BASE_STORAGE_DIR = ../RAG_STORAGE
- Default FTS DB path: ../RAG_STORAGE/fts/keyword_fts.sqlite
- Override via env KEYWORD_FTS_DB, fallback: /mnt/data/keyword_fts.sqlite
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Iterable
from pathlib import Path
import json
import re
import os
import sqlite3
import hashlib

from lightrag import LightRAG
from lightrag.base import QueryParam
from knowledgeMapper.utils.local_models import Reranker

# =========================================================
#                    STORAGE ROOT (IMPORTANT)
# =========================================================
BASE_STORAGE_DIR = Path("../RAG_STORAGE").resolve()
FTS_DIR = (BASE_STORAGE_DIR / "fts").resolve()
FTS_DB_FILENAME = "keyword_fts.sqlite"

# =========================================================
#                       T U N A B L E S
# =========================================================
MODE = "mix"  # LightRAG retrieval mode: "naive" oder "mix"

# 1) Erst-Recall via LightRAG; später trimmen
INITIAL_TOP_K = 15       # initial LightRAG top_k vor Reranking
RERANKER_TOP_K = 10      # finale Anzahl Dokument-Chunks
RERANKER_TOP_K_KG = 10   # finale Anzahl KG-Items

# 2) MMR-ähnliche Diversifikation
MMR_LAMBDA = 0.7
MMR_MAX_CANDIDATES = 12

# 3) Lexikalische Pre-Suche (neu)
LEXICAL_PRESEARCH_ENABLED = True
LEXICAL_INJECT_K = 6                 # wie gewünscht
LEXICAL_MIN_SCORE = 1
LEXICAL_USE_SQLITE_FTS = True
LEXICAL_AUTOBUILD_FTS = True
MAX_INDEX_CHUNKS = 500_000
VERBOSE_LOG = True

# 4) Stopwörter & Synonyme
GER_STOP = {
    "die","der","das","und","oder","für","von","mit","im","in","am","an","auf",
    "ist","sind","ein","eine","den","des","zu","zur","zum","bei","aus","auch",
    # W-Fragen → rausfiltern (NEU)
    "was","wie","wer","wo","wann","wieso","warum","wodurch","womit","wohin","woher",
    "wieviel","wieviele","welche","welcher","welches","welchen","welchem"
}

# Kontakt-/Feld-Synonyme (für OR-Gruppen in der FTS-Query)
EMAIL_HINTS = ["email","e-mail","mail","kontakt","kontaktadresse"]
PHONE_HINTS = ["telefon","tel","phone","rufnummer","telefonnummer"]
ROOM_HINTS  = ["raum","zimmer"]
URL_HINTS   = ["url","website","webseite","homepage","http","https","www"]

# 5) Systemprompt (unverändert)
RELIABLE_SYSTEM_PROMPT_TEMPLATE = """
**SYSTEMBEFEHL FÜR PRÄZISE WISSENSBASIERTE ANTWORTEN:**
1.  **SPRACHE:** Antworte **AUSSCHLIESSLICH** auf **DEUTSCH**.

2.  **INFORMATIONSEINSCHRÄNKUNG & KONTEXTTREUE:** Deine Antwort MUSS **VOLLSTÄNDIG** und **AUSSCHLIESSLICH** auf den Informationen im bereitgestellten `KONTEXT` basieren.
    * **KEINE Halluzinationen:** Generiere **KEINE** neuen Informationen, spekuliere **NICHT** und füge **NICHTS HINZU**, was nicht im `KONTEXT` explizit genannt ist. Dies beinhaltet auch die Verknüpfung von Informationen, die im Kontext nicht direkt miteinander verbunden sind (z. B. geografische Nähe ohne explizite Verbindungsbeschreibung).
    * **Umgang mit unbekannten Begriffen/fehlenden direkten Verknüpfungen:** Wenn der `KONTEXT` eine Abkürzung, Domänen-Slang, einen Fachbegriff oder eine Entität (z. B. einen spezifischen Gebäudecode wie "SHL") enthält, dessen Bedeutung oder deren direkte Beziehung zu anderen relevanten Konzepten (z. B. Buslinien) nicht direkt im `KONTEXT` erklärt oder verknüpft wird, **dann verwende den Begriff genau so, wie er im Kontext steht, und versuche NICHT, ihn zu erklären, zu interpretieren oder nicht explizit genannte Verbindungen herzustellen.** Halluziniere KEINE Bedeutungen, Annahmen oder implizite geografische oder funktionale Verknüpfungen.
    * **Zitierungspflicht:** Füge am Ende jedes Absatzes, der Informationen aus dem `KONTEXT` verwendet, den **Link der Quelle** in Klammern hinzu, z. B. (https://fiw.thws.de/studium/ihr-weg-durchs-studium/projektarbeit). Entferne dabei den abschließenden Schrägstrich /, falls der Link damit endet. **Nenne keine internen IDs.**

3.  **PRÄZISION & KONSISTENZ:**
    * Synthetisiere relevante Fakten aus den `Relationships(KG)`- und `Document Chunks(DC)`-Abschnitten zu einer **flüssigen, kohärenten und gut lesbaren Antwort**.
    * Wenn der `KONTEXT` widersprüchliche Informationen zu einem Thema enthält, gib **BEIDE Versionen an** und zitiere jeweils die **Links**.
    * Formuliere eine **aussagekräftige, direkte Antwort** auf die gestellte Frage des Users.

4.  **FALLBACK-PROZEDERE:** Falls die `NUTZERFRAGE` **NICHT** oder **NICHT ausreichend** im `KONTEXT` beantwortet werden kann und **auch keine indirekten, zitierfähigen Informationen** (weder aus DC noch aus KG) vorhanden sind, dann gilt Folgendes:
    * Gib folgende wörtliche Standardantwort aus:  
      **"Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden."**
    * Falls im Abschnitt `Document Chunks` mindestens ein Eintrag vorhanden ist, gib zusätzlich den Link aus dem ersten Chunk aus – als potenziell hilfreiche Quelle zur weiteren Recherche. Verwende dafür die Formulierung:  
      **"Eventuell finden Sie weiterführende Informationen unter: first_chunk_link**  
      Ersetze `first_chunk_link` durch den `file_path` des ersten Eintrags im `Document Chunks`-Array. Stelle aber **klar**, dass dies **nicht** Teil des verifizierten Kontexts ist.

5.  **UNTERDRÜCKUNG VON PLAPPERN/ERGÄNZUNGEN:** Gehe **direkt zur Antwort** über. Vermeide einleitende Phrasen wie "Basierend auf dem Kontext..." oder abschließende Bemerkungen. Die Antwort soll **AUSSCHLIESSLICH** die direkte Beantwortung der Nutzerfrage sein — **ohne Meta-Kommentare** oder Zusammenfassungen.

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

# =========================================================
#               Context parsing / rebuilding
# =========================================================

def _parse_context_string(context_str: str) -> Optional[Dict[str, List[Dict]]]:
    """
    Parses the context string for 'mix' (structured blocks) or 'naive' JSON list.
    """
    is_structured_mode = '-----Entities(KG)-----' in context_str

    parsed_data: Dict[str, List[Dict]] = {
        "entities": [],
        "relationships": [],
        "doc_chunks": []
    }

    if is_structured_mode:
        if VERBOSE_LOG: print("   -> Parsing structured ('mix') context.")
        try:
            entities_match = re.search(r"-----Entities\(KG\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
            if entities_match:
                parsed_data["entities"] = json.loads(entities_match.group(1) or "[]")

            relationships_match = re.search(r"-----Relationships\(KG\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
            if relationships_match:
                parsed_data["relationships"] = json.loads(relationships_match.group(1) or "[]")

            doc_chunks_match = re.search(r"-----Document Chunks\(DC\)-----\s*```json\n(.*?)\n```", context_str, re.DOTALL)
            if doc_chunks_match:
                parsed_data["doc_chunks"] = json.loads(doc_chunks_match.group(1) or "[]")

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing structured context string: {e}")
            return None
    else:
        if VERBOSE_LOG: print("   -> Parsing 'naive' mode context.")
        try:
            json_match = re.search(r"```json\s*\n(.*?)\n```", context_str, re.DOTALL)
            json_to_parse = json_match.group(1) if json_match else context_str
            parsed_data["doc_chunks"] = json.loads(json_to_parse or "[]")
        except json.JSONDecodeError as e:
            print(f"Error parsing plain/wrapped JSON context string: {e}")
            return None

    return parsed_data


def _rebuild_context_string(entities: List[Dict], relationships: List[Dict], doc_chunks: List[Dict]) -> str:
    """
    Reconstructs the context string for the final answer prompt.
    """
    json_kwargs = {"indent": 2, "ensure_ascii": False}
    parts = []
    if entities:
        parts.append(f"-----Entities(KG)-----\n\n```json\n{json.dumps(entities, **json_kwargs)}\n```")
    if relationships:
        parts.append(f"-----Relationships(KG)-----\n\n```json\n{json.dumps(relationships, **json_kwargs)}\n```")
    if doc_chunks:
        parts.append(f"-----Document Chunks(DC)-----\n\n```json\n{json.dumps(doc_chunks, **json_kwargs)}\n```")
    return "\n\n".join(parts)

# =========================================================
#           Lightweight MMR (diversity) helpers
# =========================================================

def _normalize_text(t: str) -> List[str]:
    if not t:
        return []
    t = re.sub(r"\s+", " ", t.lower()).strip()
    tokens = re.findall(r"[a-zäöüß0-9]+", t)
    ngrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)] if len(tokens) >= 3 else [" ".join(tokens)]
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
    Simple MMR variant: relevance from base_order; penalize high n-gram overlap.
    """
    if top_k <= 0 or not candidate_indices:
        return []

    norm_texts = [_normalize_text(texts[i]) for i in range(len(texts))]
    selected: List[int] = []

    rel_weight = {idx: 1.0 / (1 + base_order.index(idx)) if idx in base_order else 0.0
                  for idx in candidate_indices}

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

# =========================================================
#        Keyword / FTS — helpers (no extra deps)
# =========================================================

def _extract_keywords(q: str) -> List[str]:
    """
    Tokenize + drop stopwords (incl. W-questions) and add light synonym hints.
    """
    qn = re.sub(r"[^\w@\.\-\+ ]+", " ", q.lower())
    toks = [t for t in qn.split() if len(t) > 2 and t not in GER_STOP]

    # leichte semantische Expansion
    if any(k in qn for k in ("mail","e-mail","email","kontakt")):
        toks += EMAIL_HINTS
    if any(k in qn for k in ("telefon","tel.","tel","phone","rufnummer","telefonnummer")):
        toks += PHONE_HINTS
    if any(k in qn for k in ("raum","zimmer")):
        toks += ROOM_HINTS
    if any(k in qn for k in ("url","website","webseite","homepage","http","https","www")):
        toks += URL_HINTS

    kws = sorted(set(toks))
    if VERBOSE_LOG:
        print(f"[KW] extracted keywords: {kws}")
    return kws

def _build_fts_match_query(kws: List[str]) -> str:
    """
    Build a relaxed FTS5 MATCH query:
      (email OR mail OR kontakt) AND michael AND rott
    Only create OR groups if the hint family is present in kws.
    """
    ks = set(kws)

    terms: List[str] = []

    # helper to quote tokens that contain special chars (rare here, but safe)
    def qtok(t: str) -> str:
        if re.search(r'[^a-z0-9äöüß]', t):
            # wrap in double quotes for FTS (keeps operators intact)
            return f'"{t}"'
        return t

    # OR groups for hint families (only if any hint appears)
    if any(h in ks for h in EMAIL_HINTS):
        terms.append("(" + " OR ".join(qtok(h) for h in EMAIL_HINTS) + ")")
        ks -= set(EMAIL_HINTS)
    if any(h in ks for h in PHONE_HINTS):
        terms.append("(" + " OR ".join(qtok(h) for h in PHONE_HINTS) + ")")
        ks -= set(PHONE_HINTS)
    if any(h in ks for h in ROOM_HINTS):
        terms.append("(" + " OR ".join(qtok(h) for h in ROOM_HINTS) + ")")
        ks -= set(ROOM_HINTS)
    if any(h in ks for h in URL_HINTS):
        terms.append("(" + " OR ".join(qtok(h) for h in URL_HINTS) + ")")
        ks -= set(URL_HINTS)

    # remaining plain tokens as AND-terms
    # (drop overly generic leftovers like 'kontaktadresse' if we've already added the email group)
    for t in sorted(ks):
        if not t: continue
        terms.append(qtok(t))

    # Default to simple join if nothing special
    query = " AND ".join(terms) if terms else ""
    if VERBOSE_LOG:
        print(f"[FTS] MATCH query: {query}")
    return query

def _keyword_score(text: str, kws: List[str]) -> int:
    if not text or not kws: return 0
    t = " " + text.lower() + " "
    score = 0
    for kw in kws:
        if "@" in kw or "." in kw:
            score += len(re.findall(re.escape(kw), t))
        else:
            score += len(re.findall(rf"\b{re.escape(kw)}\b", t))
    return score

def _dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set(); out = []
    for c in chunks:
        key = (c.get("file_path",""), hashlib.md5((c.get("content","")[:512]).encode("utf-8","ignore")).hexdigest())
        if key in seen: continue
        seen.add(key); out.append(c)
    return out

# ---------- FTS path resolution ----------

def _resolve_fts_db(rag_instance: Optional[LightRAG] = None) -> Path:
    candidates = []
    env = os.getenv("KEYWORD_FTS_DB")
    if env:
        candidates.append(Path(env))
    candidates.append(FTS_DIR / FTS_DB_FILENAME)
    try:
        if rag_instance and getattr(rag_instance, "working_dir", None):
            candidates.append(Path(rag_instance.working_dir) / "fts" / FTS_DB_FILENAME)
    except Exception:
        pass
    candidates.append(Path("/mnt/data/keyword_fts.sqlite"))

    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return FTS_DIR / FTS_DB_FILENAME

def _fts_exists(rag_instance: Optional[LightRAG] = None) -> bool:
    return _resolve_fts_db(rag_instance).exists()

def _fts_ensure_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _fts_connect(rag_instance: Optional[LightRAG] = None) -> Optional[sqlite3.Connection]:
    try:
        db_path = _resolve_fts_db(rag_instance)
        _fts_ensure_dir(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        if VERBOSE_LOG:
            print(f"[FTS] Using DB @ {db_path}")
        return conn
    except Exception as e:
        print(f"FTS connect error: {e}")
        return None

def _lexical_via_fts(user_query: str, limit: int, rag_instance: Optional[LightRAG] = None) -> List[Dict]:
    conn = _fts_connect(rag_instance)
    if not conn:
        return []
    try:
        kws = _extract_keywords(user_query)
        if not kws:
            conn.close()
            return []
        match_q = _build_fts_match_query(kws)
        if not match_q:
            conn.close()
            return []
        rows = conn.execute(
            "SELECT chunk_id, file_path, content FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            (match_q, limit * 5)
        ).fetchall()
        conn.close()
        if VERBOSE_LOG:
            print(f"[FTS] raw rows: {len(rows)}")
        scored = []
        for r in rows:
            s = _keyword_score(r["content"], kws)
            if s >= LEXICAL_MIN_SCORE:
                scored.append((s, {"chunk_id": r["chunk_id"], "file_path": r["file_path"], "content": r["content"]}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:limit]]
    except Exception as e:
        print(f"[FTS] query error: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return []

# ---------- Auto-build FTS from storage artifacts ----------

def _yield_from_json_any(data: Any, source_name: str = "") -> Iterable[Dict[str, str]]:
    """
    Yields dicts with keys: chunk_id, file_path, content
    Accepts:
      - list[dict]
      - dict[str, dict] -> z.B. {"chunk-...": {...}} (dein Format)
    """
    if isinstance(data, list):
        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                continue
            content = obj.get("content")
            if not content:
                continue
            file_path = obj.get("file_path") or obj.get("source") or obj.get("url") or ""
            chunk_id  = obj.get("chunk_id") or obj.get("id") or f"json@{source_name}:{i}"
            yield {"chunk_id": str(chunk_id), "file_path": str(file_path), "content": str(content)}
    elif isinstance(data, dict):
        for k, obj in data.items():
            if not isinstance(obj, dict):
                continue
            content = obj.get("content")
            if not content:
                continue
            file_path = obj.get("file_path") or obj.get("source") or obj.get("url") or ""
            chunk_id  = obj.get("chunk_id") or obj.get("id") or k  # key ist z. B. "chunk-..."
            yield {"chunk_id": str(chunk_id), "file_path": str(file_path), "content": str(content)}

def _iter_chunks_from_storage(max_items: int = MAX_INDEX_CHUNKS) -> Iterable[Dict[str, str]]:
    """
    Sucht nach Chunks in:
      - *.jsonl   (eine JSON pro Zeile)
      - *.json    (Liste ODER Dict mit chunk_id->obj)
      - *.sqlite  (Tabellen mit 'content', optional 'file_path'/'source'/'url' und 'chunk_id'/'id')
    und yieldet {chunk_id, file_path, content}.
    """
    yielded = 0

    # 1) JSONL
    for p in BASE_STORAGE_DIR.rglob("*.jsonl"):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if yielded >= max_items: return
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    for ch in _yield_from_json_any([obj], p.name):
                        yield ch
                        yielded += 1
                        if yielded >= max_items: return
        except Exception:
            continue

    # 2) JSON (Liste ODER Dict)
    for p in BASE_STORAGE_DIR.rglob("*.json"):
        if p.name.endswith(FTS_DB_FILENAME) or p.name.endswith(".schema.json"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            for ch in _yield_from_json_any(data, p.name):
                yield ch
                yielded += 1
                if yielded >= max_items: return
        except Exception:
            continue

    # 3) SQLite
    for p in BASE_STORAGE_DIR.rglob("*.sqlite"):
        if p.name == FTS_DB_FILENAME:
            continue
        try:
            conn = sqlite3.connect(str(p)); conn.row_factory = sqlite3.Row
            tbls = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for t in tbls:
                try:
                    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
                    if "content" not in cols:
                        continue
                    has_fp = "file_path" in cols or "source" in cols or "url" in cols
                    has_id = "chunk_id" in cols or "id" in cols
                    sel_cols = ["content"]
                    if has_fp:
                        sel_cols.append("file_path" if "file_path" in cols else ("source" if "source" in cols else "url"))
                    if has_id:
                        sel_cols.append("chunk_id" if "chunk_id" in cols else "id")
                    rows = conn.execute(
                        f"SELECT {', '.join(sel_cols)} FROM {t} LIMIT {max(10_000, LEXICAL_INJECT_K*50)}"
                    ).fetchall()
                    for i, r in enumerate(rows):
                        if yielded >= max_items: break
                        content = r["content"]
                        if not content: continue
                        file_path = r[sel_cols[1]] if len(sel_cols) >= 2 else ""
                        chunk_id  = r[sel_cols[2]] if len(sel_cols) >= 3 else f"db@{t}:{i}"
                        yield {"chunk_id": str(chunk_id), "file_path": str(file_path), "content": str(content)}
                        yielded += 1
                except Exception:
                    continue
            conn.close()
        except Exception:
            continue

def _build_fts_index(rag_instance: Optional[LightRAG] = None) -> bool:
    """
    Baut (oder rebaut) eine sehr einfache FTS5-Tabelle:
      CREATE VIRTUAL TABLE chunks_fts(chunk_id, file_path, content)
    Indexiert bis MAX_INDEX_CHUNKS Chunks aus Storage-Artefakten.
    """
    try:
        db_path = _resolve_fts_db(rag_instance)
        _fts_ensure_dir(db_path)
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id, file_path, content)")
        cur.execute("DELETE FROM chunks_fts")  # rebuild
        count = 0
        for ch in _iter_chunks_from_storage(max_items=MAX_INDEX_CHUNKS):
            cur.execute("INSERT INTO chunks_fts (chunk_id, file_path, content) VALUES (?,?,?)",
                        (ch.get("chunk_id",""), ch.get("file_path",""), ch.get("content","")))
            count += 1
            if count >= MAX_INDEX_CHUNKS:
                break
        conn.commit(); conn.close()
        if VERBOSE_LOG: print(f"[FTS] Indexed {count} chunks into {db_path}")
        return count > 0
    except Exception as e:
        print(f"[FTS] Build error: {e}")
        return False

# =========================================================
#      Query orchestration (with lexical pre-search)
# =========================================================

async def prepare_and_execute_retrieval(
        user_query: str,
        rag_instance: LightRAG,
) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
    """
    Orchestrates a reliable RAG process with lexical pre-search (Step 0),
    LightRAG retrieval, and parallel reranking (docs + KG).
    """

    # ---------- STEP 0: Lexikalische Pre-Suche VOR Retrieval ----------
    lexical_extras: List[Dict[str, Any]] = []
    if LEXICAL_PRESEARCH_ENABLED:
        if LEXICAL_USE_SQLITE_FTS and _fts_exists(rag_instance):
            if VERBOSE_LOG: print(f"[Step 0] FTS present → keyword pre-search")
            lexical_extras = _lexical_via_fts(user_query, LEXICAL_INJECT_K, rag_instance)
        elif LEXICAL_USE_SQLITE_FTS and LEXICAL_AUTOBUILD_FTS:
            if VERBOSE_LOG: print(f"[Step 0] No FTS found. Autobuilding index from {BASE_STORAGE_DIR} ...")
            built = _build_fts_index(rag_instance)
            if built:
                lexical_extras = _lexical_via_fts(user_query, LEXICAL_INJECT_K, rag_instance)
            else:
                if VERBOSE_LOG: print("[Step 0] FTS autobuild failed or no chunks found; skipping lexical pre-search.")

    # ---------- STEP 1: LightRAG — Context holen ----------
    params_bypass = QueryParam(mode="bypass", top_k=0)
    params_context = QueryParam(
        mode=MODE, top_k=INITIAL_TOP_K,
        only_need_context_only=True
    ) if hasattr(QueryParam, "only_need_context_only") else QueryParam(
        mode=MODE, top_k=INITIAL_TOP_K, only_need_context=True
    )

    if VERBOSE_LOG:
        print(f"[Step 1] Retrieving initial context in '{MODE}' (top_k={INITIAL_TOP_K}) ...")

    initial_context_str = await rag_instance.aquery(user_query, param=params_context)

    # ---------- STEP 2: Kontext parsen + Lexikalische Extras injizieren ----------
    parsed_context: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if initial_context_str:
        if VERBOSE_LOG: print("[Step 2] Parsing context string ...")
        parsed_context = _parse_context_string(initial_context_str)

    if not parsed_context:
        if lexical_extras:
            if VERBOSE_LOG: print("[Step 2] No context from LightRAG, but lexical extras available → continue with extras.")
            parsed_context = {"entities": [], "relationships": [], "doc_chunks": lexical_extras}
        else:
            return {
                "answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden.",
                "sources": []
            }
    else:
        if lexical_extras:
            merged = _dedupe_chunks((parsed_context.get("doc_chunks", []) or []) + lexical_extras)
            parsed_context["doc_chunks"] = merged
            if VERBOSE_LOG: print(f"[Step 2] Injected {len(lexical_extras)} lexical chunk(s). Now have {len(merged)} doc chunks.")

    reranker = Reranker()

    # ---------- STAGE 1: Re-rank Document Chunks (MMR) ----------
    doc_chunks = parsed_context.get("doc_chunks", [])
    reranked_chunks: List[Dict[str, Any]] = []
    if doc_chunks:
        if VERBOSE_LOG:
            print(f"[Stage 1] Reranking {len(doc_chunks)} document chunks (MMR candidates={min(MMR_MAX_CANDIDATES, len(doc_chunks))}, final_k={RERANKER_TOP_K})")
        doc_texts = [chunk.get("content", "") for chunk in doc_chunks]
        base_order = reranker.rerank(user_query, doc_texts)
        candidates = base_order[:min(MMR_MAX_CANDIDATES, len(base_order))]
        mmr_selected = _mmr_select(candidates, doc_texts, base_order, RERANKER_TOP_K, MMR_LAMBDA)
        reranked_chunks = [doc_chunks[i] for i in mmr_selected]

    # ---------- STAGE 2: Re-rank KG items ----------
    kg_items: List[Dict[str, Any]] = []
    kg_texts: List[str] = []

    for entity in parsed_context.get("entities", []):
        item = entity.copy(); item['__source_type'] = 'entity'; kg_items.append(item)
        name = entity.get('name', ''); etype = entity.get('type', '')
        attrs = []
        for k, v in entity.items():
            if k in ('name', 'type', '__source_type'): continue
            if isinstance(v, (str, int, float)) and len(str(v)) <= 120:
                attrs.append(f"{k}={v}")
        attr_str = "; ".join(attrs[:5])
        kg_texts.append(f"ENTITY | name={name} | type={etype}" + (f" | {attr_str}" if attr_str else ""))

    for rel in parsed_context.get("relationships", []):
        item = rel.copy(); item['__source_type'] = 'relationship'; kg_items.append(item)
        rtype = (rel.get('type', '') or '').replace('_', ' ')
        source = rel.get('source', ''); target = rel.get('target', '')
        props = []
        for k, v in rel.items():
            if k in ('type', 'source', 'target', '__source_type'): continue
            if isinstance(v, (str, int, float)) and len(str(v)) <= 120:
                props.append(f"{k}={v}")
        prop_str = "; ".join(props[:5])
        kg_texts.append(f"RELATION | {source} -[{rtype}]-> {target}" + (f" | {prop_str}" if attr_str else ""))

    reranked_entities: List[Dict[str, Any]] = []
    reranked_relationships: List[Dict[str, Any]] = []

    if kg_items:
        if VERBOSE_LOG:
            print(f"[Stage 2] Reranking {len(kg_items)} KG items (MMR candidates={min(MMR_MAX_CANDIDATES, len(kg_items))}, final_k={RERANKER_TOP_K_KG})")
        kg_base_order = reranker.rerank(user_query, kg_texts)
        kg_candidates = kg_base_order[:min(MMR_MAX_CANDIDATES, len(kg_base_order))]
        kg_mmr_selected = _mmr_select(kg_candidates, kg_texts, kg_base_order, RERANKER_TOP_K_KG, MMR_LAMBDA)
        top_kg_items = [kg_items[i] for i in kg_mmr_selected]

        reranked_entities = [it for it in top_kg_items if it['__source_type'] == 'entity']
        reranked_relationships = [it for it in top_kg_items if it['__source_type'] == 'relationship']
        for it in reranked_entities + reranked_relationships:
            if '__source_type' in it: del it['__source_type']

    # ---------- STAGE 3: Combine & rebuild context ----------
    if not reranked_chunks and not reranked_entities and not reranked_relationships:
        return {
            "answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden.",
            "sources": []
        }

    if VERBOSE_LOG: print("[Stage 3] Rebuilding final context ...")
    reranked_context_str = _rebuild_context_string(
        entities=reranked_entities,
        relationships=reranked_relationships,
        doc_chunks=reranked_chunks
    )

    # ---------- STAGE 4: Generate final answer ----------
    if VERBOSE_LOG: print("[Stage 4] Generating final answer ...")
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
