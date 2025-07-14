# 🧠 LightRAG Knowledge Builder

This project builds **vector databases** or a **unified knowledge graph** from **scraped content** in MongoDB using the [LightRAG](https://github.com/hkunlp/lightrag) framework. It supports **text extraction**, **preprocessing**, **language filtering**, and **LLM-based entity enrichment**.

---

## ✅ Installation

1. **Install Poetry**

2. **Install dependencies**:

```bash
poetry install --no-root
```

---

## 🔧 Configuration

The system is configured entirely via **environment variables**. You can either export them or use a `.env` file.

### Core Options

| Variable                      | Description                                                                            | Default         |
| ----------------------------- | -------------------------------------------------------------------------------------- | --------------- |
| `MODE`                        | `vectors` to build per-subdomain vector DBs<br>`kg` to build a unified knowledge graph | `vectors`       |
| `LANGUAGE`                    | `all`, `de`, or `en` – filters documents by language metadata                          | `all`           |


---

## 🛠 Usage

### Build per-subdomain vector DBs

```bash
MODE=vectors poetry run python build_dbs.py
```

### Build a unified Knowledge Graph

```bash
MODE=kg poetry run python build_dbs.py
```

### Build for a specific subdomain

You need to replace all `dots` with an `underscore` for the subdomains.

```bash
MODE=vectors poetry run python build_dbs.py --subdomain fiw_thws_de --subdomain www_thws_de
```

### Check the progress for a specific subdomain

```bash
jq '([.[] | select(.status != "pending")] | length) as $processed | (length) as $total | "\($processed) / \($total) Dokumente verarbeitet"' \
"$(find ../RAG_STORAGE -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -n1 | cut -d' ' -f2)/kv_store_doc_status.json"
```

---

## 🧹 What the pipeline does

### 1. **Load documents from MongoDB**

* Connects to two MongoDB collections: `pages` (HTML) and `files` (PDF).
* Filters by `LANGUAGE` setting (if not `all`).
* Extracts content:

  * `HTML` → converted to clean **Markdown**
  * `PDF` → converted to plain **text** via `PyMuPDF`

### 2. **Clean & sanitize**

* Filters out empty or broken documents
* Removes null characters and unwanted whitespace
* Organizes docs by **subdomain** (from URL)

### 3. **Vectorization or Knowledge Graph building**

Depending on `MODE`:

| MODE      | What happens                                      |
| --------- | ------------------------------------------------- |
| `vectors` | Each subdomain is indexed into its own vector DB  |
| `kg`      | All documents go into one unified knowledge graph |

---

## 🧠 Optional: Entity Extraction (KG mode)

If `ENTITY_EXTRACT_MAX_GLEANING > 0`, the pipeline will:

* Send up to N chunks to the LLM (via Ollama)
* Extract structured entities and relationships
* Integrate them into the graph

Use this if you're planning to query via knowledge triples later.

---

## 📁 Output

All output is stored in the `RAG_STORAGE` directory:

```
RAG_STORAGE/
├── thws.de/             ← vector DB for one subdomain
├── some.other.site/
└── _UNIFIED_KG/         ← single knowledge graph (if MODE=kg)
```

---
