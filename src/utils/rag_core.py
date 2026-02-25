"""
rag_core.py contains the core retrieval utilities for the RAG pipeline.

Functions
---------
query_news:
    Retrieve dataset elements based on a list of indices.

build_embeddings_joblib:
    Generate embeddings for a dataset and save them to a joblib file.

retrieve:
    Retrieve the top-k most relevant documents based on cosine similarity
    between a query embedding and stored document embeddings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Optional

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# DATASET ACCESS
# =====================================================

def query_news(indices: List[int], dataset: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Retrieve elements from a dataset based on specified indices.

    Parameters
    ----------
    indices:
        A list of integer indices.
    dataset:
        Dataset supporting indexing (e.g., list of dicts).

    Returns
    -------
    list[dict[str, Any]]
        The selected dataset items.
    """
    return [dataset[i] for i in indices]


# =====================================================
# EMBEDDING GENERATION
# =====================================================

def build_embeddings_joblib(
    dataset: Sequence[Dict[str, Any]],
    model: SentenceTransformer,
    output_path: str = "embeddings.joblib",
    *,
    fields: Optional[List[str]] = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    max_chars: int = 493,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Generate embeddings for a dataset and persist them in a joblib file.

    Parameters
    ----------
    dataset:
        List of dataset records (dicts).
    model:
        SentenceTransformer model.
    output_path:
        Path where embeddings.joblib will be saved.
    fields:
        Fields to concatenate for embedding generation.
        Defaults to ["title", "description"].
    batch_size:
        Encoding batch size.
    normalize_embeddings:
        Whether to normalize embeddings (recommended for cosine similarity).
    max_chars:
        Maximum number of characters kept per text.
    dtype:
        Numpy dtype used for storage.

    Returns
    -------
    np.ndarray
        Generated embeddings matrix of shape (N, D).
    """
    if fields is None:
        fields = ["title", "description"]

    texts: List[str] = []

    for item in dataset:
        text_parts = []
        for field in fields:
            value = item.get(field, "")
            if value:
                text_parts.append(str(value))
        text = " ".join(text_parts).strip()[:max_chars]
        texts.append(text)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )

    embeddings = np.asarray(embeddings, dtype=dtype)
    joblib.dump(embeddings, output_path)

    return embeddings


# =====================================================
# RETRIEVAL
# =====================================================

def retrieve(
    query: str,
    *,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[int]:
    """
    Retrieve the indices of the top-k most similar documents for a query.

    Parameters
    ----------
    query:
        User query string.
    model:
        SentenceTransformer model (must match embeddings model).
    embeddings:
        Precomputed document embeddings.
    top_k:
        Number of documents to retrieve.

    Returns
    -------
    list[int]
        Indices of top-k most similar documents.
    """
    query_embedding = model.encode(query, normalize_embeddings=True)

    similarity_scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    ranked_indices = np.argsort(-similarity_scores)

    return ranked_indices[:top_k].tolist()


def get_relevant_data(query: str, model: SentenceTransformer, embeddings: np.ndarray, dataset: list[dict], top_k: int = 5) -> list[dict]:
    """
    Retrieve and return the top relevant data items based on a given query.

    This function performs the following steps:
    1. Retrieves the indices of the top 'k' relevant items from a dataset based on the provided `query`.
    2. Fetches the corresponding data for these indices from the dataset.

    Parameters:
    - query (str): The search query string used to find relevant items.
    - top_k (int, optional): The number of top items to retrieve. Default is 5.

    Returns:
    - list[dict]: A list of dictionaries containing the data associated 
      with the top relevant items.

    """
    # Retrieve the indices of the top_k relevant items given the query
    relevant_indices = retrieve(query = query, model=model, embeddings=embeddings, top_k = top_k)

    # Obtain the data related to the items using the indices from the previous step
    relevant_data = query_news(indices = relevant_indices, dataset=dataset)

    return relevant_data