# File: knowledgeMapper/retrieval.py (drop-in)
# Version: v7.3 — Retrieval Upgrade (LightRAG + Lexical Pre-Search)
#
# Adds:
# - Step 0: Lightweight keyword/FTS pre-search over all chunks (no extra deps)
# - Inject 1–2 lexical chunks before LightRAG retrieval flow
# - Auto-build SQLite FTS5 index from storage artifacts
# - Robust JSON parsing incl. dict-mapped chunks {"chunk-...": {...}}
# - NEW: W-question stopwords + relaxed FTS query with OR groups for contact terms
# - NEW: Bool-Parser (und/oder/nicht, +/-, &/|, Klammern) und Query-Relaxation
# - NEW: Mehrere Tokenizer-Fallbacks für FTS5 (kompatibel) + Prefix
# - NEW: Fix für "Non-relative patterns are unsupported" in rglob
#
# Storage:
# - Uses BASE_STORAGE_DIR = ../RAG_STORAGE
# - Default FTS DB path: ../RAG_STORAGE/fts/keyword_fts.sqlite
# - Override via env KEYWORD_FTS_DB, fallback: /mnt/data/keyword_fts.sqlite

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Iterable, Tuple
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
INITIAL_TOP_K = 25
RERANKER_TOP_K = 6
RERANKER_TOP_K_KG = 25

# 2) MMR-ähnliche Diversifikation
MMR_LAMBDA = 1
MMR_MAX_CANDIDATES = 25

# 3) Lexikalische Pre-Suche (neu)
LEXICAL_PRESEARCH_ENABLED = True
LEXICAL_INJECT_K = 4
LEXICAL_MIN_SCORE = 1
LEXICAL_USE_SQLITE_FTS = True
LEXICAL_AUTOBUILD_FTS = True
MAX_INDEX_CHUNKS = 500_000
VERBOSE_LOG = True

# 4) Stopwörter & Synonyme
GER_STOP = {
    "die","der","das","und","oder","für","von","mit","im","in","am","an","auf",
    "ist","sind","ein","eine","den","des","zu","zur","zum","bei","aus","auch",
    # W-Fragen raus
    "was","wie","wer","wo","wann","wieso","warum","wodurch","womit","wohin","woher",
    "wieviel","wieviele","welche","welcher","welches","welchen","welchem"
}
EMAIL_HINTS = ["email","e-mail","mail","kontakt","kontaktadresse"]
PHONE_HINTS = ["telefon","tel","phone","rufnummer","telefonnummer"]
ROOM_HINTS  = ["raum","zimmer"]
URL_HINTS   = ["url","website","webseite","homepage","http","https","www"]

RELIABLE_SYSTEM_PROMPT_TEMPLATE = """
**SYSTEMBEFEHL FÜR PRÄZISE WISSENSBASIERTE ANTWORTEN:**
1.  **SPRACHE:** Antworte **AUSSCHLIESSLICH** auf **DEUTSCH**.
2.  **INFORMATIONSEINSCHRÄNKUNG & KONTEXTTREUE:** Deine Antwort MUSS **VOLLSTÄNDIG** und **AUSSCHLIESSLICH** auf den Informationen im bereitgestellten `KONTEXT` basieren.
    * **KEINE Halluzinationen.**
    * **Unbekannte Begriffe nicht deuten.**
    * **Zitierungspflicht:** Link am Absatzende, ohne abschließenden Slash.
3.  **PRÄZISION & KONSISTENZ:** Synthese aus KG/Chunks; Widersprüche benennen (jeweils mit Link).
4.  **FALLBACK:** Wenn nichts Passendes: genau den Satz ausgeben + optional first_chunk_link.
5.  **KEIN Meta-Gerede.**
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
    is_structured_mode = '-----Entities(KG)-----' in context_str
    parsed_data: Dict[str, List[Dict]] = {"entities": [], "relationships": [], "doc_chunks": []}

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
    if not t: return []
    t = re.sub(r"\s+", " ", t.lower()).strip()
    tokens = re.findall(r"[a-zäöüß0-9]+", t)
    ngrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)] if len(tokens) >= 3 else [" ".join(tokens)]
    return ngrams

def _overlap_score(a: List[str], b: List[str]) -> float:
    if not a or not b: return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb); union = len(sa | sb) or 1
    return inter / union

def _mmr_select(candidate_indices: List[int], texts: List[str], base_order: List[int], top_k: int, lambda_mult: float) -> List[int]:
    if top_k <= 0 or not candidate_indices: return []
    norm_texts = [_normalize_text(texts[i]) for i in range(len(texts))]
    selected: List[int] = []
    rel_weight = {idx: 1.0 / (1 + base_order.index(idx)) if idx in base_order else 0.0 for idx in candidate_indices}
    while len(selected) < min(top_k, len(candidate_indices)):
        best_idx, best_score = None, -1.0
        for idx in candidate_indices:
            if idx in selected: continue
            max_sim = max((_overlap_score(norm_texts[idx], norm_texts[j]) for j in selected), default=0.0)
            score = lambda_mult * rel_weight.get(idx, 0.0) - (1 - lambda_mult) * max_sim
            if score > best_score:
                best_score, best_idx = score, idx
        if best_idx is None: break
        selected.append(best_idx)
    return selected

# =========================================================
#        Keyword / FTS — helpers (no extra deps)
# =========================================================

def _extract_keywords(q: str) -> List[str]:
    qn = re.sub(r"[^\w@\.\-\+ ]+", " ", q.lower())
    toks = [t for t in qn.split() if len(t) > 2 and t not in GER_STOP]
    if any(k in qn for k in ("mail","e-mail","email","kontakt")): toks += EMAIL_HINTS
    if any(k in qn for k in ("telefon","tel.","tel","phone","rufnummer","telefonnummer")): toks += PHONE_HINTS
    if any(k in qn for k in ("raum","zimmer")): toks += ROOM_HINTS
    if any(k in qn for k in ("url","website","webseite","homepage","http","https","www")): toks += URL_HINTS
    kws = sorted(set(toks))
    if VERBOSE_LOG: print(f"[KW] extracted keywords: {kws}")
    return kws

def _build_fts_match_query(kws: List[str]) -> str:
    ks = set(kws)
    terms: List[str] = []
    def qtok(t: str) -> str:
        return f'"{t}"' if re.search(r'[^a-z0-9äöüß]', t) else t
    if any(h in ks for h in EMAIL_HINTS):
        terms.append("(" + " OR ".join(qtok(h) for h in EMAIL_HINTS) + ")"); ks -= set(EMAIL_HINTS)
    if any(h in ks for h in PHONE_HINTS):
        terms.append("(" + " OR ".join(qtok(h) for h in PHONE_HINTS) + ")"); ks -= set(PHONE_HINTS)
    if any(h in ks for h in ROOM_HINTS):
        terms.append("(" + " OR ".join(qtok(h) for h in ROOM_HINTS) + ")"); ks -= set(ROOM_HINTS)
    if any(h in ks for h in URL_HINTS):
        terms.append("(" + " OR ".join(qtok(h) for h in URL_HINTS) + ")"); ks -= set(URL_HINTS)
    for t in sorted(ks):
        if not t: continue
        terms.append(qtok(t))
    query = " AND ".join(terms) if terms else ""
    if VERBOSE_LOG: print(f"[FTS] MATCH query: {query}")
    return query

def _keyword_score(text: str, kws: List[str]) -> int:
    if not text or not kws: return 0
    t = " " + text.lower() + " "
    score = 0
    for kw in kws:
        if "@" in kw or "." in kw: score += len(re.findall(re.escape(kw), t))
        else: score += len(re.findall(rf"\b{re.escape(kw)}\b", t))
    return score

def _dedupe_chunks(chunks: List[Dict]) -> List[Dict]:
    seen = set(); out = []
    for c in chunks:
        key = (c.get("file_path",""), hashlib.md5((c.get("content","")[:512]).encode("utf-8","ignore")).hexdigest())
        if key in seen: continue
        seen.add(key); out.append(c)
    return out

# ---------- BOOL/RELAX: Operatoren & Utilities ----------
_OP_MAP = {
    "und": "AND", "oder": "OR", "nicht": "NOT",
    "and": "AND", "or": "OR", "not": "NOT",
    "&": "AND", "|": "OR"
}
_PARENS = {"(", ")"}
MAX_MATCH_TERMS = 12

def _qtok(t: str) -> str:
    if not t: return ""
    if t.startswith("(") and t.endswith(")"): return t
    return f'"{t}"' if re.search(r'[^a-z0-9äöüß]', t) else t

def _family_group_for(token: str) -> Optional[str]:
    t = (token or "").lower()
    fam = None
    if t in EMAIL_HINTS: fam = EMAIL_HINTS
    elif t in PHONE_HINTS: fam = PHONE_HINTS
    elif t in ROOM_HINTS: fam = ROOM_HINTS
    elif t in URL_HINTS: fam = URL_HINTS
    if fam: return "(" + " OR ".join(_qtok(h) for h in fam) + ")"
    return None

def _extract_keywords_strict(q: str) -> List[str]:
    qn = re.sub(r"[^\w@\.\-\+:/ ]+", " ", (q or "").lower())
    toks = [t for t in qn.split() if t and t not in GER_STOP]
    out = [t for t in toks if len(t) > 1]
    uniq, seen = [], set()
    for t in sorted(out, key=lambda x: (-len(x), x)):
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq[: max(MAX_MATCH_TERMS, 6)]

def _build_fts_match_query_bool(user_query: str, kws: List[str]) -> str:
    q = (user_query or "").strip()
    raw_toks = re.findall(r'\(|\)|\+?[^\s()]+|\-[^\s()]+', q, flags=re.UNICODE)
    has_ops = any(t.lower() in _OP_MAP or t in _PARENS or t.startswith(("+","-")) for t in raw_toks)
    if not has_ops:
        ks = set(kws); parts: List[str] = []
        if any(h in ks for h in EMAIL_HINTS):
            parts.append("(" + " OR ".join(_qtok(h) for h in EMAIL_HINTS) + ")"); ks -= set(EMAIL_HINTS)
        if any(h in ks for h in PHONE_HINTS):
            parts.append("(" + " OR ".join(_qtok(h) for h in PHONE_HINTS) + ")"); ks -= set(PHONE_HINTS)
        if any(h in ks for h in ROOM_HINTS):
            parts.append("(" + " OR ".join(_qtok(h) for h in ROOM_HINTS) + ")"); ks -= set(ROOM_HINTS)
        if any(h in ks for h in URL_HINTS):
            parts.append("(" + " OR ".join(_qtok(h) for h in URL_HINTS) + ")"); ks -= set(URL_HINTS)
        for t in sorted(ks):
            if not t: continue
            g = _family_group_for(t); parts.append(g if g else _qtok(t))
        query = " AND ".join(parts)
        if VERBOSE_LOG: print(f"[FTS] MATCH query (fallback AND): {query}")
        return query

    out: List[str] = []
    for tok in raw_toks:
        tl = tok.lower()
        if tl in _OP_MAP: out.append(_OP_MAP[tl]); continue
        if tok in _PARENS: out.append(tok); continue
        neg = False
        if tok.startswith("-"): neg = True; term = tok[1:]
        elif tok.startswith("+"): term = tok[1:]
        else: term = tok
        term = term.strip()
        if not term: continue
        if not neg and term.lower() in GER_STOP: continue
        group = _family_group_for(term)
        term_expr = group if group else _qtok(term)
        out.append(("NOT " if neg else "") + term_expr)

    final: List[str] = []
    prev_was_term = False
    for piece in out:
        if piece in {"AND","OR"}:
            final.append(piece); prev_was_term = False; continue
        if piece == "(":
            if prev_was_term: final.append("AND")
            final.append(piece); prev_was_term = False; continue
        if piece == ")":
            final.append(piece); prev_was_term = True; continue
        if prev_was_term: final.append("AND")
        final.append(piece); prev_was_term = True
    query = " ".join(final).strip()
    if VERBOSE_LOG: print(f"[FTS] MATCH query (bool): {query}")
    return query

def _prefix_token(t: str) -> str:
    if any(c in t for c in "@/:"): return _qtok(t)
    return f"{t}*" if len(t) >= 4 else _qtok(t)

def _mk_group_or(tokens: List[str]) -> str:
    return "(" + " OR ".join(_qtok(t) for t in tokens) + ")"

def _mk_group_or_prefix(tokens: List[str]) -> str:
    return "(" + " OR ".join(_prefix_token(t) for t in tokens) + ")"

def _mk_near_pairs(tokens: List[str], window: int = 6) -> List[str]:
    toks = [t for t in tokens if not t.startswith("(")]
    pairs = []
    for i in range(len(toks)-1):
        a, b = _qtok(toks[i]), _qtok(toks[i+1])
        pairs.append(f"{a} NEAR {b}")  # <-- ohne /N (FTS5)
    return pairs[:6]

# ---------- FTS path resolution ----------

def _resolve_fts_db(rag_instance: Optional[LightRAG] = None) -> Path:
    candidates = []
    env = os.getenv("KEYWORD_FTS_DB")
    if env: candidates.append(Path(env))
    candidates.append(FTS_DIR / FTS_DB_FILENAME)
    try:
        if rag_instance and getattr(rag_instance, "working_dir", None):
            candidates.append(Path(rag_instance.working_dir) / "fts" / FTS_DB_FILENAME)
    except Exception:
        pass
    candidates.append(Path("/mnt/data/keyword_fts.sqlite"))
    for p in candidates:
        try:
            if p.exists(): return p
        except Exception:
            continue
    return FTS_DIR / FTS_DB_FILENAME

def _fts_exists(rag_instance: Optional[LightRAG] = None) -> bool:
    return _resolve_fts_db(rag_instance).exists()

def _fts_ensure_dir(path: Path) -> None:
    try: path.parent.mkdir(parents=True, exist_ok=True)
    except Exception: pass

def _fts_connect(rag_instance: Optional[LightRAG] = None) -> Optional[sqlite3.Connection]:
    try:
        db_path = _resolve_fts_db(rag_instance)
        _fts_ensure_dir(db_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        if VERBOSE_LOG: print(f"[FTS] Using DB @ {db_path}")
        return conn
    except Exception as e:
        print(f"FTS connect error: {e}")
        return None

# === Lexikalische Suche mit Bool + Relaxation ===
def _lexical_via_fts(user_query: str, limit: int, rag_instance: Optional[LightRAG] = None) -> List[Dict]:
    conn = _fts_connect(rag_instance)
    if not conn: return []
    try:
        raw_kws = _extract_keywords_strict(user_query)
        if not raw_kws:
            conn.close(); return []

        def expand_families(tokens: List[str]) -> List[str]:
            out: List[str] = []; fam_seen = set()
            for t in tokens:
                g = _family_group_for(t)
                if g and g not in fam_seen:
                    out.append(g); fam_seen.add(g)
                elif not g:
                    out.append(t)
            return out

        toks = expand_families(raw_kws)

        queries: List[str] = []
        raw_toks = re.findall(r'\(|\)|\+?[^\s()]+|\-[^\s()]+', (user_query or "").strip(), flags=re.UNICODE)
        has_bool = any(t.lower() in _OP_MAP or t in _PARENS or t.startswith(("+","-")) for t in raw_toks)
        if has_bool:
            q_bool = _build_fts_match_query_bool(user_query, raw_kws)
            if q_bool: queries.append(q_bool)

        if toks:
            queries.append(" AND ".join(_qtok(t) for t in toks))

        near_pairs = _mk_near_pairs(toks, window=6)
        if near_pairs:
            queries.append(" OR ".join(near_pairs))

        base_tokens = [t for t in toks if not t.startswith("(")]
        if base_tokens:
            queries.append(_mk_group_or(base_tokens))
            queries.append(_mk_group_or_prefix(base_tokens))

        rows = []; tried = 0
        for mq in queries:
            tried += 1
            if VERBOSE_LOG: print(f"[FTS] TRY {tried}: {mq}")
            rows = conn.execute(
                "SELECT chunk_id, file_path, content FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
                (mq, limit * 5)
            ).fetchall()
            if rows:
                if VERBOSE_LOG: print(f"[FTS] → HIT on try {tried}: {len(rows)} row(s)")
                break

        if not rows:
            conn.close(); return []
        kws_for_score = [re.sub(r'\*$', '', t) for t in raw_kws]
        scored = []
        for r in rows:
            s = _keyword_score(r["content"], kws_for_score)
            if s >= LEXICAL_MIN_SCORE:
                scored.append((s, {"chunk_id": r["chunk_id"], "file_path": r["file_path"], "content": r["content"]}))
        scored.sort(key=lambda x: x[0], reverse=True)
        conn.close()
        return [c for _, c in scored[:limit]]
    except Exception as e:
        print(f"[FTS] query error: {e}")
        try: conn.close()
        except Exception: pass
        return []

# ---------- Auto-build FTS from storage artifacts ----------

def _candidate_storage_roots(rag_instance: Optional[LightRAG] = None) -> List[Path]:
    roots: List[Path] = [BASE_STORAGE_DIR]
    try:
        if rag_instance and getattr(rag_instance, "working_dir", None):
            roots.append(Path(rag_instance.working_dir).resolve())
    except Exception:
        pass
    roots.append((Path.cwd() / "RAG_STORAGE").resolve())
    uniq: List[Path] = []; seen = set()
    for r in roots:
        p = r.resolve()
        if p in seen: continue
        seen.add(p); uniq.append(p)
    return uniq

def _yield_from_json_any(data: Any, source_name: str = "") -> Iterable[Dict[str, str]]:
    if isinstance(data, list):
        for i, obj in enumerate(data):
            if not isinstance(obj, dict): continue
            content = obj.get("content")
            if not content: continue
            file_path = obj.get("file_path") or obj.get("source") or obj.get("url") or ""
            chunk_id  = obj.get("chunk_id") or obj.get("id") or f"json@{source_name}:{i}"
            yield {"chunk_id": str(chunk_id), "file_path": str(file_path), "content": str(content)}
    elif isinstance(data, dict):
        for k, obj in data.items():
            if not isinstance(obj, dict): continue
            content = obj.get("content")
            if not content: continue
            file_path = obj.get("file_path") or obj.get("source") or obj.get("url") or ""
            chunk_id  = obj.get("chunk_id") or obj.get("id") or k
            yield {"chunk_id": str(chunk_id), "file_path": str(file_path), "content": str(content)}

def _iter_chunks_from_storage(max_items: int = MAX_INDEX_CHUNKS,
                              rag_instance: Optional[LightRAG] = None
                              ) -> Iterable[Dict[str, str]]:
    yielded = 0
    roots = _candidate_storage_roots(rag_instance)
    if VERBOSE_LOG:
        print(f"[FTS] Scanning storage roots: {', '.join(map(str, roots))}")
    for root in roots:
        if not root.exists():
            if VERBOSE_LOG: print(f"[FTS]   - SKIP (not exists): {root}")
            continue
        if VERBOSE_LOG: print(f"[FTS]   - ENTER: {root}")

        # 1) JSONL / NDJSON
        for ext in ("*.jsonl", "*.ndjson"):
            for p in root.rglob(ext):
                try:
                    with p.open("r", encoding="utf-8") as f:
                        for line in f:
                            if yielded >= max_items: return
                            line = line.strip()
                            if not line: continue
                            try: obj = json.loads(line)
                            except Exception: continue
                            for ch in _yield_from_json_any([obj], p.name):
                                yield ch; yielded += 1
                                if yielded >= max_items: return
                except Exception as e:
                    if VERBOSE_LOG: print(f"[FTS]     jsonl read error @ {p}: {e}")

        # 2) JSON
        for p in root.rglob("*.json"):
            if p.name.endswith(FTS_DB_FILENAME) or p.name.endswith(".schema.json"):
                continue
            try: data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                if VERBOSE_LOG: print(f"[FTS]     json read error @ {p}: {e}")
                continue
            try:
                for ch in _yield_from_json_any(data, p.name):
                    yield ch; yielded += 1
                    if yielded >= max_items: return
            except Exception as e:
                if VERBOSE_LOG: print(f"[FTS]     json yield error @ {p}: {e}")
                continue

        # 3) SQLite
        for p in root.rglob("*.sqlite"):
            if p.name == FTS_DB_FILENAME:
                continue
            try:
                conn = sqlite3.connect(str(p)); conn.row_factory = sqlite3.Row
                tbls = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                for t in tbls:
                    try:
                        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
                        if "content" not in cols: continue
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
            except Exception as e:
                if VERBOSE_LOG: print(f"[FTS]     sqlite scan error @ {p}: {e}")
                continue

def _build_fts_index(rag_instance: Optional[LightRAG] = None) -> bool:
    """
    Baut (oder rebaut) eine FTS5-Tabelle. Probiert mehrere Tokenizer-Varianten,
    fällt sonst auf plain fts5 zurück. Detaillierte Logs.
    """
    try:
        db_path = _resolve_fts_db(rag_instance)
        _fts_ensure_dir(db_path)
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        def try_create(sql: str) -> Tuple[bool, Optional[str]]:
            try:
                cur.execute(sql)
                return True, None
            except Exception as e:
                return False, str(e)

        variants = [
            # weit kompatibel
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
              chunk_id,
              file_path,
              content,
              tokenize='unicode61 tokenchars "-@._:/"',
              prefix='2 3 4'
            )
            """,
            # einfacher unicode61
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
              chunk_id,
              file_path,
              content,
              tokenize='unicode61',
              prefix='2 3 4'
            )
            """,
            # plain
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(chunk_id, file_path, content)
            """
        ]

        created = False; last_err = None
        for v in variants:
            ok, err = try_create(v)
            if ok:
                created = True
                break
            last_err = err
            if VERBOSE_LOG: print(f"[FTS] CREATE failed variant: {err}")
        if not created:
            print(f"[FTS] ERROR: CREATE TABLE failed. Last error: {last_err}")
            conn.close()
            return False

        cur.execute("DELETE FROM chunks_fts")  # rebuild

        count = 0; sample_printed = 0
        for ch in _iter_chunks_from_storage(max_items=MAX_INDEX_CHUNKS, rag_instance=rag_instance):
            cur.execute("INSERT INTO chunks_fts (chunk_id, file_path, content) VALUES (?,?,?)",
                        (ch.get("chunk_id",""), ch.get("file_path",""), ch.get("content","")))
            if VERBOSE_LOG and sample_printed < 3:
                print(f"[FTS]   + example row: chunk_id={ch.get('chunk_id','')[:40]} file={ch.get('file_path','')[:60]}")
                sample_printed += 1
            count += 1
            if count >= MAX_INDEX_CHUNKS:
                break
        conn.commit(); conn.close()
        if VERBOSE_LOG:
            roots = _candidate_storage_roots(rag_instance)
            print(f"[FTS] Indexed {count} chunks into {db_path}  (roots: {', '.join(map(str, roots))})")
        return count > 0
    except Exception as e:
        print(f"[FTS] Build error: {e}")
        return False

def _fts_diagnostics(rag_instance: Optional[LightRAG] = None) -> None:
    roots = _candidate_storage_roots(rag_instance)
    print("[FTS] Diagnostics — roots:")
    for r in roots:
        print(f"  - {r}  {'(exists)' if r.exists() else '(missing)'}")
    total = 0
    for _ in _iter_chunks_from_storage(max_items=50, rag_instance=rag_instance):
        total += 1
    print(f"[FTS] Diagnostics — first 50 potential chunks seen: {total}")

# =========================================================
#      Query orchestration (with lexical pre-search)
# =========================================================

async def prepare_and_execute_retrieval(
        user_query: str,
        rag_instance: LightRAG,
) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
    # ---------- STEP 0 ----------
    lexical_extras: List[Dict[str, Any]] = []
    if LEXICAL_PRESEARCH_ENABLED:
        if LEXICAL_USE_SQLITE_FTS and _fts_exists(rag_instance):
            if VERBOSE_LOG: print(f"[Step 0] FTS present → keyword pre-search")
            lexical_extras = _lexical_via_fts(user_query, LEXICAL_INJECT_K, rag_instance)
        elif LEXICAL_USE_SQLITE_FTS and LEXICAL_AUTOBUILD_FTS:
            if VERBOSE_LOG:
                print(f"[Step 0] No FTS found. Autobuilding index from storage roots ...")
                _fts_diagnostics(rag_instance)
            built = _build_fts_index(rag_instance)
            if built:
                lexical_extras = _lexical_via_fts(user_query, LEXICAL_INJECT_K, rag_instance)
            else:
                if VERBOSE_LOG:
                    print("[Step 0] FTS autobuild failed or no chunks found; skipping lexical pre-search.")

    # ---------- STEP 1 ----------
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

    # ---------- STEP 2 ----------
    parsed_context: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if initial_context_str:
        if VERBOSE_LOG: print("[Step 2] Parsing context string ...")
        parsed_context = _parse_context_string(initial_context_str)

    if not parsed_context:
        if lexical_extras:
            if VERBOSE_LOG: print("[Step 2] No context from LightRAG, but lexical extras available → continue with extras.")
            parsed_context = {"entities": [], "relationships": [], "doc_chunks": lexical_extras}
        else:
            return {"answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden.", "sources": []}
    else:
        if lexical_extras:
            merged = _dedupe_chunks((parsed_context.get("doc_chunks", []) or []) + lexical_extras)
            parsed_context["doc_chunks"] = merged
            if VERBOSE_LOG: print(f"[Step 2] Injected {len(lexical_extras)} lexical chunk(s). Now have {len(merged)} doc chunks.")

    reranker = Reranker()

    # ---------- STAGE 1 ----------
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

    # ---------- STAGE 2 ----------
    kg_items: List[Dict[str, Any]] = []
    kg_texts: List[str] = []

    for entity in parsed_context.get("entities", []):
        item = entity.copy(); item['__source_type'] = 'entity'; kg_items.append(item)
        name = entity.get('name', ''); etype = entity.get('type', '')
        attrs = []
        for k, v in entity.items():
            if k in ('name', 'type', '__source_type'): continue
            if isinstance(v, (str, int, float)) and len(str(v)) <= 120: attrs.append(f"{k}={v}")
        attr_str = "; ".join(attrs[:5])
        kg_texts.append(f"ENTITY | name={name} | type={etype}" + (f" | {attr_str}" if attr_str else ""))

    for rel in parsed_context.get("relationships", []):
        item = rel.copy(); item['__source_type'] = 'relationship'; kg_items.append(item)
        rtype = (rel.get('type', '') or '').replace('_', ' ')
        source = rel.get('source', ''); target = rel.get('target', '')
        props = []
        for k, v in rel.items():
            if k in ('type', 'source', 'target', '__source_type'): continue
            if isinstance(v, (str, int, float)) and len(str(v)) <= 120: props.append(f"{k}={v}")
        prop_str = "; ".join(props[:5])
        kg_texts.append(f"RELATION | {source} -[{rtype}]-> {target}" + (f" | {prop_str}" if prop_str else ""))

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

    # ---------- STAGE 3 ----------
    if not reranked_chunks and not reranked_entities and not reranked_relationships:
        return {"answer": "Ich konnte keine passenden Informationen zu Ihrer Anfrage in meiner Wissensdatenbank finden.", "sources": []}

    if VERBOSE_LOG: print("[Stage 3] Rebuilding final context ...")
    reranked_context_str = _rebuild_context_string(entities=reranked_entities, relationships=reranked_relationships, doc_chunks=reranked_chunks)

    # ---------- STAGE 4 ----------
    if VERBOSE_LOG: print("[Stage 4] Generating final answer ...")
    final_system_prompt = RELIABLE_SYSTEM_PROMPT_TEMPLATE.format(
        current_date=datetime.now().strftime("%d. %B %Y"),
        location="Würzburg",
        context=reranked_context_str,
        user_query=user_query
    )
    citable_answer_text = await rag_instance.aquery(user_query, param=QueryParam(mode="bypass", top_k=0), system_prompt=final_system_prompt)
    final_sources = reranked_chunks + reranked_entities + reranked_relationships
    return {"answer": citable_answer_text, "sources": final_sources}
