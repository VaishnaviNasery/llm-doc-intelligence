# LLM Document Intelligence Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you upload PDF documents and ask natural-language questions over them. Built with Groq (free LLM API), FAISS vector search, sentence-transformers for local embeddings, and a clean Flask web UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

| Feature | Details |
|---|---|
| **PDF Ingestion** | Sentence-boundary chunking with overlap (no arbitrary char splits) |
| **Local Embeddings** | `all-MiniLM-L6-v2` via sentence-transformers — no paid embedding API |
| **Vector Search** | FAISS `IndexFlatL2` persisted to disk — survives restarts |
| **MMR Reranking** | Maximal Marginal Relevance balances relevance and diversity |
| **LLM Generation** | Groq `llama-3.1-8b-instant` — fast, free tier available |
| **Grounded Answers** | Strict prompt forces [Source N] citations; refuses to hallucinate |
| **Eval Metrics** | Precision@k, faithfulness, latency tracked per query in SQLite |
| **Web UI** | Dark-mode Flask app with drag-and-drop upload and live metrics |

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  ingestor   │───▶│   embedder      │───▶│  FAISS Index     │
│  (pypdf +   │    │  (MiniLM-L6-v2) │    │  (disk-persisted)│
│   chunking) │    └─────────────────┘    └──────────────────┘
└─────────────┘              │
        │                    │ embed_and_index
        ▼                    ▼
  ┌──────────┐       ┌─────────────────┐
  │ SQLite   │◀──────│   database      │
  │ (chunks  │       │  (metadata)     │
  │  metadata│       └─────────────────┘
  └──────────┘
        │
        │   Query
        ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  retriever  │───▶│   generator      │───▶│  Groq API       │
│  (FAISS +   │    │  (grounded       │    │  (Llama 3.1)    │
│   MMR)      │    │   prompt)        │    └─────────────────┘
└─────────────┘    └──────────────────┘
        │                   │
        ▼                   ▼
  ┌──────────┐      ┌──────────────┐
  │ evaluator│      │  Flask UI    │
  │ (metrics)│      │  (web/server)│
  └──────────┘      └──────────────┘
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/VaishnaviNasery/llm-doc-intelligence.git
cd llm-doc-intelligence

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 3. Run the web app

```bash
python -m web.server
# → http://localhost:5000
```

### 4. Use the UI

1. Drag & drop a PDF into the upload zone (left sidebar)
2. Wait for ingestion to complete (chunk count shown)
3. Type a question and press **Ask** or hit Enter
4. View the grounded answer with source citations and eval metrics

---

## Project Structure

```
llm-doc-intelligence/
├── app/
│   ├── __init__.py
│   ├── database.py      # SQLite metadata store
│   ├── ingestor.py      # PDF parsing + sentence chunking
│   ├── embedder.py      # FAISS vector index (MiniLM-L6-v2)
│   ├── retriever.py     # Top-k retrieval + MMR reranking
│   ├── generator.py     # Groq LLM answer generation
│   └── evaluator.py     # Precision@k, faithfulness, latency
├── web/
│   ├── __init__.py
│   ├── server.py        # Flask routes
│   └── templates/
│       └── index.html   # Dark-mode UI
├── data/                # Auto-created at runtime
│   ├── chunks.db        # SQLite (chunk metadata)
│   ├── faiss.index      # FAISS vector index
│   ├── faiss_map.json   # FAISS pos → SQLite row id
│   └── eval_log.db      # Evaluation metrics log
├── .env.example
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/upload` | Ingest a PDF (multipart form: `file`) |
| `POST` | `/query` | Ask a question (`{"question": "..."}`) |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/<doc_id>` | Remove a document |
| `GET` | `/metrics` | Evaluation history and averages |

### Example: query via curl

```bash
curl -s -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings of this paper?"}' | python -m json.tool
```

---

## Evaluation Metrics

Every query is automatically evaluated and logged to `data/eval_log.db`:

| Metric | Description |
|---|---|
| **Latency (ms)** | End-to-end response time |
| **Precision@k** | Fraction of retrieved chunks sharing ≥2 keywords with the query |
| **Faithfulness** | Fraction of answer sentences grounded in at least one chunk (≥3 keyword overlap) |
| **Token usage** | Prompt + completion tokens from Groq |

View aggregate stats:
```bash
curl http://localhost:5000/metrics
```

---

## Tech Stack

- **LLM**: Groq API — `llama-3.1-8b-instant` (free tier)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Vector DB**: FAISS `IndexFlatL2` (in-process, disk-persisted)
- **Metadata DB**: SQLite (zero setup)
- **PDF parsing**: `pypdf`
- **Web server**: Flask 3
- **Frontend**: Vanilla HTML/CSS/JS (no build step)

---

## License

MIT
