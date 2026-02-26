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

#=====================================================
# RELEVANT DATA
#=====================================================

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


#=====================================================
# PROMPT GENERATION
#=====================================================

def generate_final_prompt(query: str, model: SentenceTransformer, embeddings: np.ndarray, dataset: list[dict], top_k: int = 5, use_rag: bool = True, prompt: str = None) -> str:
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
    relevant_data = get_relevant_data(query=query, model=model, embeddings=embeddings, dataset=dataset, top_k=top_k)

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

def llm_call(query: str, model: SentenceTransformer, embeddings: np.ndarray, dataset: list[dict], top_k: int = 5, use_rag: bool = True, together_api_key: str = os.getenv("TOGETHER_API_KEY"), prompt: str = None) -> str:
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
    prompt = generate_final_prompt(query=query, model=model, embeddings=embeddings, dataset=dataset, top_k=top_k, use_rag=use_rag, prompt=prompt)

    # Call the LLM
    generated_response = generate_with_single_input(prompt=prompt, together_api_key=os.getenv("TOGETHER_API_KEY"))

    # Get the content
    generated_message = generated_response['content']
    
    return generated_message

def display_widget(llm_call_func: callable, model: SentenceTransformer, embeddings: np.ndarray, dataset: list[dict], top_k: int = 5, use_rag: bool = True, prompt: str = None) -> str:
    def on_button_click(b):
        # Clear outputs
        output1.clear_output()
        output2.clear_output()
        status_output.clear_output()
        # Display "Generating..." message
        status_output.append_stdout("Generating...\n")
        query = query_input.value
        top_k = slider.value
        prompt = prompt_input.value.strip() if prompt_input.value.strip() else None
        response1 = llm_call(query=query, model=model, embeddings=embeddings, dataset=dataset, top_k=top_k, use_rag=True, prompt=prompt)      
        response2 = llm_call(query=query, model=model, embeddings=embeddings, dataset=dataset, top_k=top_k, use_rag=False, prompt=prompt)
        # Update responses
        with output1:
            display(Markdown(response1))
        with output2:
            display(Markdown(response2))
        # Clear "Generating..." message
        status_output.clear_output()

    query_input = widgets.Text(
        description='Query:',
        placeholder='Type your query here',
        layout=widgets.Layout(width='100%')
    )

    prompt_input = widgets.Textarea(
        description='Augmented prompt layout:',
        placeholder=("Type your prompt layout here, don't forget to add {query} and {documents} "
                     "where you want them to be placed! Leaving this blank will default to the "
                     "prompt in generate_final_prompt. Example:\nThis is a query: {query}\nThese are the documents: {documents}"),
        layout=widgets.Layout(width='100%', height='100px'),
        style={'description_width': 'initial'}
    )

    slider = widgets.IntSlider(
        value=5,  # default value
        min=1,
        max=20,
        step=1,
        description='Top K:',
        style={'description_width': 'initial'}
    )

    output1 = widgets.Output(layout={'border': '1px solid #ccc', 'width': '45%'})
    output2 = widgets.Output(layout={'border': '1px solid #ccc', 'width': '45%'})
    status_output = widgets.Output()

    submit_button = widgets.Button(
        description="Get Responses",
        style={'button_color': '#f0f0f0', 'font_color': 'black'}
    )
    submit_button.on_click(on_button_click)

    label1 = widgets.Label(value="With RAG", layout={'width': '45%', 'text_align': 'center'})
    label2 = widgets.Label(value="Without RAG", layout={'width': '45%', 'text_align': 'center'})

    display(widgets.HTML("""
    <style>
        .custom-output {
            background-color: #f9f9f9;
            color: black;
            border-radius: 5px;
        }
        .widget-textarea, .widget-button {
            background-color: #f0f0f0 !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
        .widget-output {
            background-color: #f9f9f9 !important;
            color: black !important;
        }
        textarea {
            background-color: #fff !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
    </style>
    """))

    display(query_input, prompt_input, slider, submit_button, status_output)
    hbox_labels = widgets.HBox([label1, label2], layout={'justify_content': 'space-between'})
    hbox_outputs = widgets.HBox([output1, output2], layout={'justify_content': 'space-between'})

    def style_outputs(*outputs):
        for output in outputs:
            output.layout.margin = '5px'
            output.layout.height = '300px'
            output.layout.padding = '10px'
            output.layout.overflow = 'auto'
            output.add_class("custom-output")

    style_outputs(output1, output2)
    # Display label and output boxes
    display(hbox_labels)
    display(hbox_outputs)