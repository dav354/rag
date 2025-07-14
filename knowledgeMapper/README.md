# 🧠 Advanced RAG Knowledge Builder

This project implements a high-performance, asynchronous pipeline to build and query a **unified knowledge graph** or **per-subdomain vector databases** from scraped web content. Built on the [LightRAG](https://github.com/HKUDS/LightRAG) framework, it features a sophisticated, multi-stage architecture designed for efficiency, accuracy, and scalability.

The system intelligently processes diverse data formats (HTML, PDF, iCal), leverages a hybrid text extraction strategy with OCR fallback, and employs advanced RAG techniques like query expansion to deliver precise, context-aware answers.

---

## ✨ Key Features

- **Asynchronous Pipeline:** Built entirely with `asyncio` for high-throughput, non-blocking I/O during data loading, embedding, and LLM interactions.
- **Hybrid Data Loading:** Pre-processes and caches PDF text in MongoDB to accelerate subsequent builds, while processing lighter formats (HTML, iCal) in parallel on-the-fly.
- **Advanced Text Extraction:**
    - Converts HTML to clean Markdown.
    - Extracts text from PDFs using `PyMuPDF`.
    - **Automatic OCR Fallback:** If a PDF contains minimal text, it automatically performs OCR with `pytesseract` to capture content from scanned documents.
    - Parses `.ics` (iCal) files into human-readable event descriptions.
- **GPU-Accelerated & Memory-Safe:**
    - Leverages CUDA for high-performance HuggingFace sentence embeddings.
    - Manages GPU memory with a semaphore to prevent OOM errors during concurrent embedding tasks.
- **Advanced RAG Strategy:**
    - **Query Expansion:** Uses an LLM to rewrite the user's initial query into more comprehensive versions, improving retrieval accuracy.
    - **Detailed Guardrail Prompting:** Implements a highly structured system prompt with a "Chain-of-Thought" process and strict guardrails to ensure answers are factual and based *only* on the provided context.
- **Developer-Friendly:**
    - **Rich CLI:** Provides detailed progress bars and configuration summaries using the `rich` library.
    - **Cost Estimator:** Includes a script (`cost_estimator.py`) to estimate token counts and potential costs for using cloud-based embedding or generation models.

---

## 🏗️ System Architecture

The pipeline follows a clear, multi-stage process designed for efficiency and modularity:

1.  **PDF Pre-processing:**
    - The `preprocess_pdfs.py` script scans the `files` collection in MongoDB for PDFs.
    - It performs hybrid text extraction (standard + OCR fallback) and saves the clean text to a separate `extracted_content` collection.
    - This heavy lifting is done once, making subsequent builds much faster.

2.  **Main Build Process (`build_dbs.py`):**
    - **Load Data:**
        - Fetches pre-processed PDF text directly from the `extracted_content` cache.
        - Loads raw HTML and iCal content from the `pages` and `files` collections.
    - **Process Live Data:**
        - Converts HTML to Markdown and parses iCal files in parallel using a `ProcessPoolExecutor`.
    - **Chunk & Index:**
        - Splits documents into structured chunks (respecting Markdown headers).
        - Initializes the `LightRAG` instance with the configured embedding and LLM models.
        - Enqueues all document chunks and processes them to build the knowledge graph or vector stores.

3.  **Retrieval (`retrieval.py`)**:
    - **Query Expansion:** The initial user query is sent to the LLM to generate alternative, more detailed questions.
    - **Context Retrieval:** The original and expanded queries are used to retrieve the most relevant document chunks and knowledge graph relationships.
    - **Answer Generation:** The retrieved context is passed to the LLM with a strict, guardrailed system prompt to generate a final, source-based answer.

---

## 🔧 Configuration

The system is configured entirely via **environment variables**. Create a `.env` file in the `knowledgeMapper` directory or export them in your shell.

| Variable                      | Description                                                                                             | Default                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------ |
| `LANGUAGE`                    | Filters documents by language (`de`, `en`, or `all`).                                                   | `de`                     |
| `BASE_STORAGE_DIR`            | The directory where `LightRAG` will store the knowledge graph and vector indexes.                       | `../RAG_STORAGE`         |
| `ENTITY_EXTRACT_MAX_GLEANING` | Controls how many chunks are sent to the LLM for entity extraction in KG mode (0 disables it).            | `1`                      |
| `MONGO_HOST`                  | MongoDB host.                                                                                           | `localhost`              |
| `MONGO_PORT`                  | MongoDB port.                                                                                           | `27017`                  |
| `MONGO_DB_NAME`               | MongoDB database name.                                                                                  | `askthws_scraper`        |
| `MONGO_USER` / `MONGO_PASS`   | MongoDB credentials.                                                                                    | `scraper` / `password`   |
| `EMBEDDING_MODEL_NAME`        | The HuggingFace model for sentence embeddings.                                                          | `aari1995/German_Semantic_V3` |
| `EMBEDDING_DEVICE`            | The device for embedding (`cuda`, `cpu`).                                                               | `cuda`                   |
| `OLLAMA_MODEL_NAME`           | The model to use via the Ollama server for generation tasks.                                            | `gemma3:4b`              |
| `OLLAMA_HOST`                 | The URL for the Ollama server.                                                                          | `http://localhost:11434` |

---

## 🛠 Usage

### 1. Pre-process PDFs (Recommended First Step)

This script extracts text from all PDFs in your database and caches it for fast access. Run it once before your first build, and again if new PDFs are added.

```bash
poetry run python preprocess_pdfs.py
```

### 2. Build the Knowledge Base

This is the main script to build the knowledge graph or vector stores from the processed data.

**Build a unified Knowledge Graph for all documents:**

```bash
poetry run python build_dbs.py
```

**Build for one or more specific subdomains:**

(Subdomain names are sanitized by replacing `.` with `_`)

```bash
poetry run python build_dbs.py --subdomain fiw_thws_de --subdomain www_thws_de
```

---

## 📁 Output

All generated knowledge bases are stored in the `RAG_STORAGE` directory (or as configured by `BASE_STORAGE_DIR`).
