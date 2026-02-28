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

get_relevant_data:
    Retrieve the top-k most relevant documents based on cosine similarity
    between a query embedding and stored document embeddings.

generate_final_prompt:
    Generate a final prompt based on a user query, optionally incorporating relevant data using retrieval-augmented generation (RAG).

generate_with_single_input:
    Generate a response from a Together-hosted language model using a single prompt.
generate_with_multiple_input:
    Generate a response from a Together-hosted language model using a list of chat messages.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence, Optional

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import bm25s
from utils.formatting import format_relevant_data
from together import Together
import ipywidgets as widgets
from IPython.display import display, Markdown


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
    method: str = "semantic",
    BM25_RETRIEVER=None,
    corpus=None
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
    method:
        Retrieval method. "semantic" (default) or "bm25".
    BM25_RETRIEVER:
        Pre-indexed BM25 retriever (required if method="bm25").
    corpus:
        Corpus used by BM25 (required if method="bm25") to map results back to indices.

    Returns
    -------
    list[int]
        Indices of top-k most similar documents.
    """

    if method == "bm25":
        if BM25_RETRIEVER is None or corpus is None:
            raise ValueError("BM25_RETRIEVER and corpus must be provided when method='bm25'.")

        # Tokenize query and retrieve with BM25
        tokenized_query = bm25s.tokenize(query)
        results, scores = BM25_RETRIEVER.retrieve(tokenized_query, k=top_k)

        results = results[0]
        top_k_indices = [corpus.index(results[k]) for k in range(len(results))]

        return top_k_indices

    # --- semantic (default): strictly unchanged ---
    query_embedding = model.encode(query, normalize_embeddings=True)

    similarity_scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    ranked_indices = np.argsort(-similarity_scores)

    return ranked_indices[:top_k].tolist()

#=====================================================
# RELEVANT DATA
#=====================================================

def reciprocal_rank_fusion(list1, list2, top_k=5, K=60):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.

    Args:
        list1 (list[int]): A list of indices of the top-k documents that match the query.
        list2 (list[int]): Another list of indices of the top-k documents that match the query.
        top_k (int): The number of top documents to consider from each list for fusion. Defaults to 5.
        K (int): A constant used in the RRF formula. Defaults to 60.

    Returns:
        list[int]: A list of indices of the top-k documents sorted by their RRF scores.
    """
    
    # Create a dictionary to store the RRF scores for each document index
    rrf_scores = {}

    # Iterate over each document list
    for lst in [list1, list2]:
        # Calculate the RRF score for each document index
        for rank, item in enumerate(lst, start=1): # Start = 1 set the first element as 1 and not 0. 
                                                   # This is a convention on how ranks work (the first element in ranking is denoted by 1 and not 0 as in lists)
            # If the item is not in the dictionary, initialize its score to 0
            if item not in rrf_scores:
                rrf_scores[item] = 0
            # Update the RRF score for each document index using the formula 1 / (rank + K)
            rrf_scores[item] += 1 / (rank + K)
    print(rrf_scores)
    # Sort the document indices based on their RRF scores in descending order
    sorted_items = sorted(rrf_scores, key=rrf_scores.get, reverse = True)

    # Slice the list to get the top-k document indices
    top_k_indices = [int(x) for x in sorted_items[:top_k]]

    return top_k_indices

def get_relevant_data(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    dataset: list[dict],
    top_k: int = 5,
    method: str = "semantic",
    BM25_RETRIEVER=None,
    corpus=None,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Retrieve and return the top relevant data items based on a given query.

    Supported methods:
    - "semantic": embedding cosine similarity retrieval
    - "bm25": BM25 keyword retrieval
    - "hybrid_rrf": combine "semantic" and "bm25" rankings using Reciprocal Rank Fusion (RRF)

    Parameters
    ----------
    query : str
        Search query.
    model : SentenceTransformer
        Embedding model.
    embeddings : np.ndarray
        Precomputed document embeddings.
    dataset : list[dict]
        Dataset aligned with embeddings/corpus (same ordering).
    top_k : int
        Number of items to return.
    method : str
        Retrieval method: "semantic" (default), "bm25", or "hybrid_rrf".
    BM25_RETRIEVER : Any
        Prebuilt BM25 retriever (required for bm25 / hybrid_rrf).
    corpus : list
        Corpus used by BM25 to map results back to indices (required for bm25 / hybrid_rrf).
    rrf_k : int
        RRF constant K (default 60).

    Returns
    -------
    list[dict]
        Top relevant dataset items.
    """

    # --- Hybrid: semantic + bm25 with RRF ---
    if method == "hybrid_rrf":
        if BM25_RETRIEVER is None or corpus is None:
            raise ValueError("BM25_RETRIEVER and corpus must be provided when method='hybrid_rrf'.")

        # Get two ranked lists (indices)
        semantic_indices = retrieve(
            query=query,
            model=model,
            embeddings=embeddings,
            top_k=top_k,
            method="semantic",
        )

        bm25_indices = retrieve(
            query=query,
            model=model,
            embeddings=embeddings,
            top_k=top_k,
            method="bm25",
            BM25_RETRIEVER=BM25_RETRIEVER,
            corpus=corpus,
        )

        # Fuse them with RRF, then fetch data
        fused_indices = reciprocal_rank_fusion(
            semantic_indices,
            bm25_indices,
            top_k=top_k,
            K=rrf_k
        )

        return query_news(indices=fused_indices, dataset=dataset)

    # --- Single-method retrieval (semantic or bm25) ---
    relevant_indices = retrieve(
        query=query,
        model=model,
        embeddings=embeddings,
        top_k=top_k,
        method=method,
        BM25_RETRIEVER=BM25_RETRIEVER,
        corpus=corpus
    )

    relevant_data = query_news(indices=relevant_indices, dataset=dataset)
    return relevant_data


#=====================================================
# PROMPT GENERATION
#=====================================================

def generate_final_prompt(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    dataset: list[dict],
    top_k: int = 5,
    use_rag: bool = True,
    prompt: str = None,
    method: str = "semantic",
    BM25_RETRIEVER=None,
    corpus=None,
    rrf_k: int = 60
    ) -> str:
    """
    Generates a final prompt based on a user query, optionally incorporating relevant data using retrieval-augmented generation (RAG).

    Args:
        query (str): The user query for which the prompt is to be generated.
        top_k (int, optional): The number of top relevant data pieces to retrieve and incorporate. Default is 5.
        use_rag (bool, optional): A flag indicating whether to use retrieval-augmented generation (RAG)
                                  by including relevant data in the prompt. Default is True.
        prompt (str, optional): A template string for the prompt. It can contain placeholders {query} and {documents}
                                for formatting with the query and formatted relevant data, respectively.

    Returns:
        str: The generated prompt, either consisting solely of the query or expanded with relevant data
             formatted for additional context.
    """
    # If RAG is not being used, format the prompt with just the query or return the query directly
    if not use_rag:
        return query

    # Retrieve the top_k relevant data pieces based on the query
    relevant_data = get_relevant_data(query=query, model=model, embeddings=embeddings, dataset=dataset, top_k=top_k, method=method, BM25_RETRIEVER=BM25_RETRIEVER, corpus=corpus, rrf_k=rrf_k)

    # Format the retrieved relevant data
    retrieve_data_formatted = format_relevant_data(relevant_data=relevant_data)

    # If no custom prompt is provided, use the default prompt template
    if prompt is None:
        prompt = (
            f"Answer the user query below. There will be provided additional information for you to compose your answer. "
            f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
            f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
            f"Query: {query}\n"
            f"2024 News: {retrieve_data_formatted}"
        )
    else:
        # If a custom prompt is provided, format it with the query and formatted relevant data
        prompt = prompt.format(query=query, documents=retrieve_data_formatted)

    return prompt

#=====================================================
# LLM CALL
#=====================================================

def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 500,
    model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    together_api_key: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, str]:
    """
    Generate a response from a Together-hosted language model using a single prompt.

    Parameters
    ----------
    prompt : str
        The input text prompt sent to the language model.
    role : str
        Role of the sender in the chat message (default: "user").
    top_p : float, optional
        Nucleus sampling parameter.
    temperature : float, optional
        Sampling temperature.
    max_tokens : int
        Maximum number of tokens to generate.
    model : str
        Model identifier on Together.
    together_api_key : str, optional
        Together API key. If not provided, the function reads
        the "TOGETHER_API_KEY" environment variable.
    **kwargs : Any
        Additional parameters forwarded to the API.

    Returns
    -------
    dict
        A dictionary containing:
        - "role": model role
        - "content": generated text
    """

    # Load API key
    if together_api_key is None:
        together_api_key = os.environ.get("TOGETHER_API_KEY")

    if together_api_key is None:
        raise ValueError("Together API key not provided.")

    # Build payload
    payload = {
        "model": model,
        "messages": [{"role": role, "content": prompt}],
        "max_tokens": max_tokens,
        **kwargs
    }

    if top_p is not None:
        payload["top_p"] = top_p

    if temperature is not None:
        payload["temperature"] = temperature

    # Call Together API
    client = Together(api_key=together_api_key)
    response = client.chat.completions.create(**payload)

    message = response.choices[0].message

    return {
        "role": message.role,
        "content": message.content
    }

def generate_with_multiple_input(
    messages: List[Dict[str, str]],
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 500,
    model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    together_api_key: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, str]:
    """
    Generate a response from a Together-hosted language model 
    using a list of chat messages.

    Parameters
    ----------
    messages : List[Dict[str, str]]
        List of chat messages in OpenAI format:
        [{"role": "system/user/assistant", "content": "..."}]
    top_p : float, optional
        Nucleus sampling parameter.
    temperature : float, optional
        Sampling temperature.
    max_tokens : int
        Maximum number of tokens to generate.
    model : str
        Model identifier on Together.
    together_api_key : str, optional
        Together API key. If not provided, the function reads
        the "TOGETHER_API_KEY" environment variable.
    **kwargs : Any
        Additional parameters forwarded to the API.

    Returns
    -------
    dict
        {
            "role": "assistant",
            "content": "<generated text>"
        }
    """

    # Load API key
    if together_api_key is None:
        together_api_key = os.environ.get("TOGETHER_API_KEY")

    if together_api_key is None:
        raise ValueError("Together API key not provided.")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs
    }

    if top_p is not None:
        payload["top_p"] = top_p

    if temperature is not None:
        payload["temperature"] = temperature

    # Call Together API
    client = Together(api_key=together_api_key)
    response = client.chat.completions.create(**payload)

    message = response.choices[0].message

    return {
        "role": message.role,
        "content": message.content
    }

def llm_call(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    dataset: list[dict],
    top_k: int = 5,
    use_rag: bool = True,
    together_api_key: str = os.getenv("TOGETHER_API_KEY"),
    prompt: str = None,
    method: str = "semantic",
    BM25_RETRIEVER=None,
    corpus=None,
    rrf_k: int = 60
    ) -> str:
    """
    Calls the LLM to generate a response based on a query, optionally using retrieval-augmented generation.

    Args:
        query (str): The user query that will be processed by the language model.
        use_rag (bool, optional): A flag that indicates whether to use retrieval-augmented generation by 
                                  incorporating relevant documents into the prompt. Default is True.

    Returns:
        str: The content of the response generated by the language model.
    """
    

    # Get the prompt with the query + relevant documents
    prompt = generate_final_prompt(query=query, model=model, embeddings=embeddings, dataset=dataset, top_k=top_k, use_rag=use_rag, prompt=prompt, method=method, BM25_RETRIEVER= BM25_RETRIEVER, corpus=corpus, rrf_k=rrf_k)

    # Call the LLM
    generated_response = generate_with_single_input(prompt=prompt, together_api_key=os.getenv("TOGETHER_API_KEY"))

    # Get the content
    generated_message = generated_response['content']
    
    return generated_message

def display_widget(
    llm_call_func: callable,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    dataset: list[dict],
    *,
    BM25_RETRIEVER=None,
    corpus=None,
    rrf_k: int = 60,
) -> None:
    def on_button_click(_):
        # Clear outputs
        out_sem.clear_output()
        out_bm25.clear_output()
        out_rrf.clear_output()
        out_no_rag.clear_output()
        status_output.clear_output()

        status_output.append_stdout("Generating...\n")

        query = query_input.value.strip()
        top_k = slider.value
        custom_prompt = prompt_input.value.strip() if prompt_input.value.strip() else None

        if not query:
            status_output.clear_output()
            status_output.append_stdout("Please enter a query.\n")
            return

        # --- Semantic (RAG) ---
        try:
            resp_sem = llm_call(
                query=query,
                model=model,
                embeddings=embeddings,
                dataset=dataset,
                top_k=top_k,
                use_rag=True,
                prompt=custom_prompt,
                method="semantic",
                BM25_RETRIEVER=BM25_RETRIEVER,
                corpus=corpus,
                rrf_k=rrf_k,
            )
        except Exception as e:
            resp_sem = f"**Error (Semantic):** {e}"

        # --- BM25 (RAG) ---
        try:
            resp_bm25 = llm_call(
                query=query,
                model=model,
                embeddings=embeddings,
                dataset=dataset,
                top_k=top_k,
                use_rag=True,
                prompt=custom_prompt,
                method="bm25",
                BM25_RETRIEVER=BM25_RETRIEVER,
                corpus=corpus,
                rrf_k=rrf_k,
            )
        except Exception as e:
            resp_bm25 = f"**Error (BM25):** {e}"

        # --- Hybrid RRF (RAG) ---
        try:
            resp_rrf = llm_call(
                query=query,
                model=model,
                embeddings=embeddings,
                dataset=dataset,
                top_k=top_k,
                use_rag=True,
                prompt=custom_prompt,
                method="hybrid_rrf",
                BM25_RETRIEVER=BM25_RETRIEVER,
                corpus=corpus,
                rrf_k=rrf_k,
            )
        except Exception as e:
            resp_rrf = f"**Error (RRF):** {e}"

        # --- Without RAG ---
        try:
            resp_no_rag = llm_call(
                query=query,
                model=model,
                embeddings=embeddings,
                dataset=dataset,
                top_k=top_k,
                use_rag=False,
                prompt=custom_prompt,
                method="semantic",  # irrelevant when use_rag=False, but harmless
                BM25_RETRIEVER=BM25_RETRIEVER,
                corpus=corpus,
                rrf_k=rrf_k,
            )
        except Exception as e:
            resp_no_rag = f"**Error (Without RAG):** {e}"

        with out_sem:
            display(Markdown(resp_sem))
        with out_bm25:
            display(Markdown(resp_bm25))
        with out_rrf:
            display(Markdown(resp_rrf))
        with out_no_rag:
            display(Markdown(resp_no_rag))

        status_output.clear_output()

    # Inputs
    query_input = widgets.Text(
        description="Query:",
        placeholder="Type your query here",
        layout=widgets.Layout(width="100%")
    )

    prompt_input = widgets.Textarea(
        description="Augmented prompt layout:",
        placeholder=("Optional custom prompt. Use {query} and {documents} placeholders.\n"
                     "Leave blank to use the default prompt builder."),
        layout=widgets.Layout(width="100%", height="90px"),
        style={"description_width": "initial"}
    )

    slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description="Top K:",
        style={"description_width": "initial"}
    )

    submit_button = widgets.Button(
        description="Get Responses",
        button_style="",  # keep neutral
        layout=widgets.Layout(width="160px")
    )
    submit_button.on_click(on_button_click)

    status_output = widgets.Output()

    # Outputs (4 panels)
    out_sem = widgets.Output(layout={"border": "1px solid #ccc", "height": "320px", "overflow": "auto"})
    out_bm25 = widgets.Output(layout={"border": "1px solid #ccc", "height": "320px", "overflow": "auto"})
    out_rrf = widgets.Output(layout={"border": "1px solid #ccc", "height": "320px", "overflow": "auto"})
    out_no_rag = widgets.Output(layout={"border": "1px solid #ccc", "height": "320px", "overflow": "auto"})

    # Titles
    title_sem = widgets.HTML("<b>Semantic Search</b>")
    title_bm25 = widgets.HTML("<b>BM25 Search</b>")
    title_rrf = widgets.HTML("<b>Reciprocal Rank Fusion</b>")
    title_no_rag = widgets.HTML("<b>Without RAG</b>")

    # Layout: controls on top
    controls = widgets.VBox([
        query_input,
        prompt_input,
        widgets.HBox([slider, submit_button]),
        status_output
    ])

    # Layout: 2x2 grid
    cell_left_top = widgets.VBox([title_sem, out_sem])
    cell_right_top = widgets.VBox([title_bm25, out_bm25])
    cell_left_bottom = widgets.VBox([title_rrf, out_rrf])
    cell_right_bottom = widgets.VBox([title_no_rag, out_no_rag])

    grid = widgets.VBox([
        widgets.HBox([cell_left_top, cell_right_top], layout=widgets.Layout(justify_content="space-between")),
        widgets.HBox([cell_left_bottom, cell_right_bottom], layout=widgets.Layout(justify_content="space-between")),
    ])

    # Make columns consistent width
    for cell in [cell_left_top, cell_right_top, cell_left_bottom, cell_right_bottom]:
        cell.layout.width = "49%"

    display(widgets.HTML("""
    <style>
        .widget-label { font-size: 14px; }
        textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    </style>
    """))

    display(controls, grid)