"""
server.py — Flask web application for the LLM Document Intelligence Pipeline.

Routes:
  GET  /              → main UI
  POST /upload        → ingest a PDF
  POST /query         → ask a question
  GET  /documents     → list all ingested documents
  DELETE /documents/<doc_id> → remove a document
  GET  /metrics       → retrieval + generation evaluation stats
"""

import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template, abort

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database  import init_db, insert_chunks, list_documents, delete_doc
from app.ingestor  import ingest_pdf
from app.embedder  import embed_and_index, get_index_size, reset_index
from app.retriever import retrieve, format_context
from app.generator import generate_answer
from app.evaluator import init_eval_db, log_eval, get_eval_history, get_avg_metrics, Timer

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

# ── Startup ───────────────────────────────────────────────────────────────────

@app.before_request
def startup():
    app.before_request_funcs[None].remove(startup)   # run only once
    init_db()
    init_eval_db()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Ingest a PDF: parse → chunk → embed → index → store metadata."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save to a temp file so ingest_pdf can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        doc_id, filename, chunks = ingest_pdf(tmp_path)
        if not chunks:
            return jsonify({"error": "No text could be extracted from this PDF"}), 422

        # Persist metadata → get back the SQLite row ids
        row_ids = insert_chunks(doc_id, filename, chunks)

        # Embed and add to FAISS
        texts = [c["text"] for c in chunks]
        embed_and_index(texts, row_ids)

        return jsonify({
            "doc_id":      doc_id,
            "filename":    filename,
            "chunk_count": len(chunks),
            "index_size":  get_index_size(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/query", methods=["POST"])
def query():
    """Answer a question over the ingested documents."""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    top_k   = int(data.get("top_k", 5))
    fetch_k = int(data.get("fetch_k", 20))

    if get_index_size() == 0:
        return jsonify({"error": "No documents have been ingested yet"}), 422

    with Timer() as t:
        chunks  = retrieve(question, top_k=top_k, fetch_k=fetch_k)
        context = format_context(chunks)
        result  = generate_answer(question, context)

    metrics = log_eval(
        query      = question,
        chunks     = chunks,
        answer     = result["answer"],
        latency_ms = t.elapsed_ms,
        usage      = result["usage"],
    )

    return jsonify({
        "answer":   result["answer"],
        "model":    result["model"],
        "sources":  [
            {
                "filename":        c["filename"],
                "page":            c["page"],
                "chunk_index":     c["chunk_index"],
                "relevance_score": c.get("relevance_score"),
                "text_preview":    c["text"][:200] + ("…" if len(c["text"]) > 200 else ""),
            }
            for c in chunks
        ],
        "metrics":  metrics,
    })


@app.route("/documents", methods=["GET"])
def documents():
    """List all ingested documents."""
    docs = list_documents()
    return jsonify({"documents": docs, "index_size": get_index_size()})


@app.route("/documents/<doc_id>", methods=["DELETE"])
def remove_document(doc_id: str):
    """Remove a document's metadata from SQLite. (FAISS vectors remain but orphan gracefully.)"""
    deleted = delete_doc(doc_id)
    if not deleted:
        abort(404, description=f"Document {doc_id} not found")
    return jsonify({"deleted": doc_id})


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return evaluation history and aggregate averages."""
    limit   = int(request.args.get("limit", 50))
    history = get_eval_history(limit=limit)
    avg     = get_avg_metrics()
    return jsonify({"averages": avg, "history": history})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
