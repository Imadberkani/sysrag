"""
flask_inference_api.py contains the inference API used by Weaviate
for embedding generation and reranking.

Functions
---------
readiness_check:
    Return a simple readiness response for health checks.

readiness_check_2:
    Return a JSON readiness response for metadata checks.

rerank_documents:
    Rerank a list of documents against a query using the BGE reranker model.

vectorize:
    Generate embeddings for one or more input texts.

run_app:
    Start the Flask inference API server.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
sys.path.extend([
    str(Path.cwd().parent),
    str(Path.cwd().parent / "src"),
])

import json
import logging
import threading
from typing import Any, Dict, List, Union, cast

from flask import Flask, Response, jsonify, request
from FlagEmbedding import FlagReranker

from src.utils.rag_weaviate_core import generate_embedding


# =====================================================
# GLOBAL INITIALIZATION
# =====================================================

reranker = FlagReranker(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    cache_dir=".models/",
    use_fp16=False,
)

app = Flask(__name__)


# =====================================================
# HEALTH CHECKS
# =====================================================

@app.route("/.well-known/ready", methods=["GET"])
def readiness_check() -> tuple[str, int]:
    """
    Return a plain-text readiness response.

    Returns
    -------
    tuple[str, int]
        A simple readiness message and HTTP status code.
    """
    return "Ready", 200


@app.route("/meta", methods=["GET"])
def readiness_check_2() -> Response:
    """
    Return a JSON readiness response.

    Returns
    -------
    flask.Response
        A JSON response indicating that the API is ready.
    """
    return jsonify({"status": "Ready"}), 200


# =====================================================
# RERANKING
# =====================================================

@app.route("/rerank", methods=["POST"])
def rerank_documents() -> Response:
    """
    Rerank a list of documents against a query.

    Expected input format
    ---------------------
    {
        "query": "<user query>",
        "documents": ["doc 1", "doc 2", ...]
    }

    Returns
    -------
    flask.Response
        A JSON response containing reranked document scores.
    """
    try:
        payload: Dict[str, Any] | None = request.get_json(silent=True)

        if payload is None:
            raw_text = request.data.decode("utf-8")
            payload = cast(Dict[str, Any], json.loads(raw_text))

        if not isinstance(payload, dict) or "query" not in payload or "documents" not in payload:
            return jsonify({
                "error": "Invalid input format. Expected a dictionary with 'query' and 'documents'."
            }), 400

        query = payload["query"]
        documents = payload["documents"]

        if not isinstance(query, str):
            return jsonify({"error": "'query' must be a string."}), 400

        if not isinstance(documents, list):
            return jsonify({"error": "'documents' must be a list of strings."}), 400

        if not documents:
            return jsonify({"scores": []}), 200

        pairs: List[tuple[str, str]] = [(query, str(doc)) for doc in documents]
        scores = reranker.compute_score(pairs)
        scores_list = scores.tolist() if hasattr(scores, "tolist") else scores

        reranked_results: List[Dict[str, Any]] = []
        for i, doc_text in enumerate(documents):
            reranked_results.append({
                "document": str(doc_text),
                "score": float(scores_list[i]),
            })

        return jsonify({"scores": reranked_results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
# VECTORIZATION
# =====================================================

@app.route("/vectors", methods=["POST"])
def vectorize() -> Response:
    """
    Generate embeddings for one or more input texts.

    Expected input format
    ---------------------
    Either:
    {
        "text": "[\"text 1\", \"text 2\"]"
    }

    or raw JSON body containing a JSON-encoded string/list.

    Returns
    -------
    flask.Response
        A JSON response containing the generated vectors.
    """
    try:
        payload: Dict[str, Any] | None = request.get_json(silent=True)

        if payload is not None and "text" in payload:
            data = payload["text"]
        else:
            data = request.data.decode("utf-8")

        text = json.loads(data)

        if isinstance(text, str):
            texts: List[str] = [text]
        elif isinstance(text, dict) and "text" in text:
            texts = [text["text"]] if isinstance(text["text"], str) else [str(x) for x in text["text"]]
        elif isinstance(text, list):
            texts = [str(x) for x in text]
        else:
            return jsonify({"error": "Invalid input format for vectorization."}), 400
        embeddings = generate_embedding(texts)

        return jsonify({"vector": embeddings[0] if len(texts) == 1 else embeddings}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
# SERVER
# =====================================================

app.logger.disabled = True
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def run_app() -> None:
    """
    Start the Flask inference API server.

    Returns
    -------
    None
    """
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_app, daemon=True)
    flask_thread.start()
    flask_thread.join()