"""
rag_weaviate_core.py contains the core utilities for the RAG pipeline
built with the Weaviate API.

Functions
---------
print_object_properties:
    Print the properties of a dataset item or Weaviate object in a readable
    format, truncating long text fields for easier inspection.

generate_embedding:
    Generate embedding vectors for one or more input texts using the loaded
    embedding model.

get_chunks_fixed_size_with_overlap:
    Split a text into fixed-size chunks with overlap between
    consecutive chunks.

prepare_weaviate_objects:
    Convert a list of articles into chunked Weaviate objects.

filter_by_metadata:
    Retrieve objects from a collection based on metadata filtering criteria.

semantic_search_retrieve:
    Perform a semantic search on a collection and retrieve matching objects.

bm25_retrieve:
    Perform a BM25 search on a collection and retrieve matching objects.

hybrid_retrieve:
    Perform a hybrid search on a collection and retrieve matching objects.

semantic_search_with_reranking:
    Perform a semantic search with reranking on a collection.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Union
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import Filter, Rerank
import numpy as np
import requests
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML
from src.utils.rag_core import (
    generate_with_single_input,
)   
# =====================================================
# GLOBAL INITIALIZATION
# =====================================================

# Loading the pretrained model from Hugging Face
model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder = ".models")
# =====================================================
# OBJECT INSPECTION
# =====================================================

def print_object_properties(obj: Union[dict, list]) -> None:
    """
    Print the properties of a dataset item or Weaviate object.

    Long text fields such as article content, vectors, and chunks are
    truncated to make the output easier to read when inspecting retrieved
    documents or collection objects.

    Parameters
    ----------
    obj:
        A dictionary representing a single object, or a list of objects.

    Returns
    -------
    None
        Prints the object properties to the console.
    """
    t = ""

    if isinstance(obj, dict):
        keys = list(obj.keys())
        keys.sort()

        for x in keys:
            y = obj[x]

            if x == "article_content":
                t += f"{x}: {y[:100]}...(truncated)\n"
            elif x == "main_vector":
                t += f"{x}: {y[:30]}...(truncated)\n"
            elif x == "chunk":
                t += f"{x}: {y[:100]}...(truncated)\n"
            else:
                t += f"{x}: {y}\n"

        print(t)

    else:
        for item in obj:
            print_object_properties(item)


# =====================================================
# EMBEDDING GENERATION
# =====================================================

def generate_embedding(prompt: Union[str, List[str]]) -> List[List[float]]:
    """
    Generate embedding vectors for one or more input texts.

    Parameters
    ----------
    prompt:
        A single input string or a list of input strings.

    Returns
    -------
    list[list[float]]
        The generated embedding vectors.
    """
    embeddings = model.encode(prompt)
    return np.asarray(embeddings).tolist()


# =====================================================
# TEXT CHUNKING
# =====================================================

def get_chunks_fixed_size_with_overlap(
    text: str,
    chunk_size: int,
    overlap_fraction: float
) -> List[str]:
    """
    Split a text into fixed-size chunks (by word count)
    with overlap between consecutive chunks.

    Parameters
    ----------
    text:
        The input text.
    chunk_size:
        Number of words per chunk.
    overlap_fraction:
        Overlap ratio between 0 and 1.
        Example: 0.2 = 20% overlap.

    Returns
    -------
    list[str]
        List of chunk strings.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if not (0 <= overlap_fraction < 1):
        raise ValueError("overlap_fraction must be in [0, 1)")

    words = text.split()
    overlap_int = int(chunk_size * overlap_fraction)
    step = chunk_size - overlap_int

    if step <= 0:
        raise ValueError("overlap too large: chunk_size - overlap must be > 0")

    chunks = []

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

    return chunks

def prepare_weaviate_objects(
    articles: List[Dict[str, Any]],
    chunk_size: int = 120,
    overlap_fraction: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Convert a list of articles into chunked Weaviate objects.

    Parameters
    ----------
    articles:
        List of article dictionaries.
    chunk_size:
        Number of words per chunk.
    overlap_fraction:
        Overlap ratio between 0 and 1.

    Returns
    -------
    list[dict[str, Any]]
        List of chunked Weaviate objects.
    """
    objects = []

    for article in articles:
        title = str(article.get("title", ""))
        description = str(article.get("description", ""))
        article_content = str(article.get("article_content", ""))
        guid = str(article.get("guid", ""))
        link = str(article.get("link", ""))
        pubDate = article.get("pubDate", None)

        chunks = get_chunks_fixed_size_with_overlap(
            text=article_content,
            chunk_size=chunk_size,
            overlap_fraction=overlap_fraction,
        )

        for idx, chunk in enumerate(chunks):
            obj = {
                "title": title,
                "description": description,
                "article_content": article_content,
                "chunk": chunk,
                "chunk_index": idx,
                "guid": guid,
                "link": link,
                "pubDate": pubDate,
            }
            objects.append(obj)

    return objects

# =====================================================
# RETRIEVAL
# =====================================================

def filter_by_metadata(metadata_property: str, 
                       values: list[str], 
                       collection: "weaviate.collections.collection.sync.Collection" , 
                       limit: int = 5) -> list:
    """
    Retrieves objects from a specified collection based on metadata filtering criteria.

    This function queries a collection within the specified client to fetch objects that match 
    certain metadata criteria. It uses a filter to find objects whose specified 'property' contains 
    any of the given 'values'. The number of objects retrieved is limited by the 'limit' parameter.

    Args:
    metadata_property (str): The name of the metadata property to filter on.
    values (List[str]): A list of values to be matched against the specified property.
    collection_name (weaviate.collections.collection.sync.Collection): The collection to query.
    limit (int, optional): The maximum number of objects to retrieve. Defaults to 5.

    Returns:
    List[Object]: A list of objects from the collection that match the filtering criteria.
    """

    
    # Retrieve using collection.query.fetch_objects
    response = collection.query.fetch_objects(
            filters = Filter.by_property(metadata_property).contains_any(values),
            limit = limit
        )
    response_objects = [x.properties for x in response.objects]
    
    return response_objects


def semantic_search_retrieve(query: str,
                             collection: "weaviate.collections.collection.sync.Collection" , 
                             top_k: int = 5) -> list:
    """
    Performs a semantic search on a collection and retrieves the top relevant chunks.

    This function executes a semantic search query on a specified collection to find text chunks 
    that are most relevant to the input 'query'. The search retrieves a limited number of top 
    matching objects, as specified by 'top_k'. The function returns the 'chunk' property of 
    each of the top matching objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the semantic search is performed.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """

    # Retrieve using collection.query.near_text
    response = collection.query.near_text(
            query=query,
            limit=top_k
        )
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects



def bm25_retrieve(query: str, 
                  collection: "weaviate.collections.collection.sync.Collection" , 
                  top_k: int = 5) -> list:
    """
    Performs a BM25 search on a collection and retrieves the top relevant chunks.

    This function executes a BM25-based search query on a specified collection to identify text 
    chunks that are most relevant to the provided 'query'. It retrieves a limited number of the 
    top matching objects, as specified by 'top_k', and returns the 'chunk' property of these objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the BM25 search is performed.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """
    
    # Retrieve using collection.query.bm25
    response = collection.query.bm25(
            query=query,
            limit=top_k
        )
    
    response_objects = [x.properties for x in response.objects]
    return response_objects 



def hybrid_retrieve(query: str, 
                    collection: "weaviate.collections.collection.sync.Collection" , 
                    alpha: float = 0.5,
                    top_k: int = 5
                   ) -> list:
    """
    Performs a hybrid search on a collection and retrieves the top relevant chunks.

    This function executes a hybrid search that combines semantic vector search and traditional 
    keyword-based search on a specified collection to find text chunks most relevant to the 
    input 'query'. The relevance of results is influenced by 'alpha', which balances the weight 
    between vector and keyword matches. It retrieves a limited number of top matching objects, 
    as specified by 'top_k', and returns the 'chunk' property of these objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the hybrid search is performed.
    alpha (float, optional): A weighting factor that balances the contribution of semantic 
    and keyword matches. Defaults to 0.5.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """

    # Retrieve using collection.query.hybrid
    response = collection.query.hybrid(
            query=query,
            alpha=alpha,
            limit=top_k
        )
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects 




def semantic_search_with_reranking(
    query: str,
    rerank_property: str,
    collection: Any,
    rerank_query: str | None = None,
    top_k: int = 10,
    rerank_url: str = "http://127.0.0.1:5000/rerank",
) -> List[Dict[str, Any]]:
    """
    Perform a semantic search in Weaviate, then rerank the results
    using the local Flask reranking API.

    Parameters
    ----------
    query : str
        The user query used for semantic search in Weaviate.

    rerank_property : str
        The document property to send to the reranker
        (for example: "chunk").

    collection : Any
        The Weaviate collection object.

    rerank_query : str | None, optional
        The query to use for reranking. If None, the main query is used.

    top_k : int, optional
        The maximum number of documents to retrieve from Weaviate.

    rerank_url : str, optional
        The URL of the local Flask reranking endpoint.

    Returns
    -------
    List[Dict[str, Any]]
        A list of document properties sorted by rerank score
        in descending order.
    """
    if rerank_query is None:
        rerank_query = query

    response = collection.query.near_text(
        query=query,
        limit=top_k,
    )

    response_objects: List[Dict[str, Any]] = [x.properties for x in response.objects]

    if not response_objects:
        return []

    documents_to_rerank: List[str] = [
        str(obj.get(rerank_property, ""))
        for obj in response_objects
    ]

    rerank_response = requests.post(
        rerank_url,
        json={
            "query": rerank_query,
            "documents": documents_to_rerank,
        },
        timeout=120,
    )
    rerank_response.raise_for_status()

    rerank_payload: Dict[str, Any] = rerank_response.json()
    rerank_scores: List[Dict[str, Any]] = rerank_payload.get("scores", [])

    ranked_results: List[Dict[str, Any]] = []

    for obj, score_item in zip(response_objects, rerank_scores):
        ranked_obj = dict(obj)
        ranked_obj["_rerank_score"] = float(score_item["score"])
        ranked_results.append(ranked_obj)

    ranked_results.sort(key=lambda x: x["_rerank_score"], reverse=True)

    return ranked_results

#=====================================================
# PROMPT GENERATION
#=====================================================

def generate_final_prompt(query: str, 
                          top_k: int, 
                          retrieve_function: callable,
                          collection,
                          rerank_query: str = None, 
                          rerank_property: str = None, 
                          use_rerank: bool = False, 
                          use_rag: bool = True) -> str:
    """
    Generates a final prompt by optionally retrieving and formatting relevant documents using retrieval-augmented generation (RAG).

    Args:
        query (str): The initial query to be used for document retrieval.
        top_k (int): The number of top documents to retrieve and use for generating the prompt.
        retrieve_function (callable): The function used to retrieve documents based on the query.
        collection: The Weaviate collection used for document retrieval.
        rerank_query (str, optional): The query used specifically for reranking documents if reranking is enabled.
        rerank_property (str, optional): The property used for reranking. Required if 'use_rerank' is True.
        use_rerank (bool, optional): Flag to denote whether to use reranking in document retrieval. Defaults to False.
        use_rag (bool, optional): Flag to determine whether to use retrieval-augmented generation. Defaults to True.

    Returns:
        str: A constructed prompt that includes the original query and formatted retrieved documents if 'use_rag' is True.
             Otherwise, it returns the original query.
    """
    # If no rag, return the query
    if not use_rag:
        return query
    
    if use_rerank:
        if rerank_property is None:
            raise ValueError('rerank_property must be set if use_rerank = True')
        top_k_documents = retrieve_function(query=query, top_k=top_k, collection = collection, rerank_property = rerank_property, rerank_query = rerank_query)
    else:
        top_k_documents = retrieve_function(query=query, top_k=top_k, collection = collection)
    
    # Initialize an empty string to store the formatted data.
    formatted_data = ""
    
    # Iterate over each retrieved document.
    for document in top_k_documents:
        # Format each document into a structured string.
        document_layout = (
            f"Title: {document['title']}, Chunk: {document['chunk']}, "
            f"Published at: {document['pubDate']}\nURL: {document['link']}"
        )
        # Append the formatted string to the main data string with a newline for separation.
        formatted_data += document_layout + "\n"
    
    # If use_rag flag is True, construct the enhanced prompt with the augmented data.
    retrieve_data_formatted = formatted_data  # Store formatted data.
    prompt = (
        f"Answer the user query below. There will be provided additional information for you to compose your answer. "
        f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
        f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
        f"The news data is ordered by relevance."
        f"Query: {query}\n"
        f"2024 News: {retrieve_data_formatted}"
    )
    
    return prompt

#=====================================================
# LLM CALL
#=====================================================
def llm_call(query: str, 
             retrieve_function: callable = None, 
             collection = None,
             top_k: int = 5, 
             use_rag: bool = True, 
             use_rerank: bool = False, 
             rerank_property: str = None, 
             rerank_query: str = None) -> str:
    """
    Simulates a call to a language model by generating a prompt and using it to produce a response.

    Args:
        query (str): The initial query for which a response is sought.
        retrieve_function (callable, optional): The function used to retrieve documents related to the query.
        collection (optional): The Weaviate collection used for document retrieval.
        top_k (int, optional): The number of top documents to retrieve and use for generating the prompt. Defaults to 5.
        use_rag (bool, optional): Indicates whether to use retrieval-augmented generation. Defaults to True.
        use_rerank (bool, optional): Indicates whether to apply reranking to the retrieved documents. Defaults to False.
        rerank_property (str, optional): The property to use for reranking. Required if 'use_rerank' is True.
        rerank_query (str, optional): The query used specifically for reranking documents if reranking is enabled.

    Returns:
        str: The generated response content after processing the prompt with a language model.
    """
    
    # Get the prompt
    PROMPT = generate_final_prompt(query, top_k = top_k, retrieve_function = retrieve_function, use_rag = use_rag, use_rerank = use_rerank, rerank_property = rerank_property, rerank_query = rerank_query, collection = collection)
    
    generated_response = generate_with_single_input(PROMPT, together_api_key=os.getenv("TOGETHER_API_KEY"))

    generated_message = generated_response['content']
    
    return generated_message



def display_widget(
    llm_call_func,
    collection,
    semantic_search_retrieve,
    bm25_retrieve,
    hybrid_retrieve,
    semantic_search_with_reranking,
):
    """
    Display an interactive widget to compare multiple retrieval strategies
    with and without RAG.

    Parameters
    ----------
    llm_call_func : callable
        Function used to generate the final answer.

    collection : Any
        The Weaviate collection used for retrieval.

    semantic_search_retrieve : callable
        Semantic retrieval function.

    bm25_retrieve : callable
        BM25 retrieval function.

    hybrid_retrieve : callable
        Hybrid retrieval function.

    semantic_search_with_reranking : callable
        Semantic retrieval function followed by local reranking.
    """

    # =====================================================
    # CALLBACK
    # =====================================================

    def on_button_click(b):
        query = query_input.value.strip()
        top_k = top_k_slider.value
        rerank_property = rerank_property_dropdown.value
        rerank_query = rerank_query_input.value.strip()

        if rerank_query == "":
            rerank_query = None

        for output in outputs.values():
            output.clear_output()

        status_output.clear_output()

        if not query:
            with status_output:
                display(Markdown("**Please enter a query.**"))
            return

        with status_output:
            display(Markdown("**Generating responses...**"))

        retrievals = [
            {
                "name": "Semantic Search",
                "output": outputs["semantic"],
                "retrieve_function": semantic_search_retrieve,
                "use_rag": True,
                "use_rerank": False,
            },
            {
                "name": "Semantic Search with Reranking",
                "output": outputs["rerank"],
                "retrieve_function": semantic_search_with_reranking,
                "use_rag": True,
                "use_rerank": True,
            },
            {
                "name": "BM25 Search",
                "output": outputs["bm25"],
                "retrieve_function": bm25_retrieve,
                "use_rag": True,
                "use_rerank": False,
            },
            {
                "name": "Hybrid Search",
                "output": outputs["hybrid"],
                "retrieve_function": hybrid_retrieve,
                "use_rag": True,
                "use_rerank": False,
            },
            {
                "name": "Without RAG",
                "output": outputs["no_rag"],
                "retrieve_function": None,
                "use_rag": False,
                "use_rerank": False,
            },
        ]

        for retrieval in retrievals:
            try:
                response = llm_call_func(
                    query=query,
                    retrieve_function=retrieval["retrieve_function"],
                    collection=collection,
                    top_k=top_k,
                    use_rag=retrieval["use_rag"],
                    use_rerank=retrieval["use_rerank"],
                    rerank_property=rerank_property if retrieval["use_rerank"] else None,
                    rerank_query=rerank_query if retrieval["use_rerank"] else None,
                )

                with retrieval["output"]:
                    display(Markdown(response))

            except Exception as e:
                with retrieval["output"]:
                    display(Markdown(
                        f"**Error while generating `{retrieval['name']}`**\n\n```python\n{str(e)}\n```"
                    ))

        status_output.clear_output()
        with status_output:
            display(Markdown("**Done.**"))

    # =====================================================
    # INPUTS
    # =====================================================

    query_input = widgets.Textarea(
        value="Tell me about United States and Brazil's relationship over the course of 2024. Provide links for the resources you use in the answer.",
        placeholder="Type your query here...",
        description="",
        layout=widgets.Layout(width="100%", height="110px"),
    )

    query_box = widgets.VBox([
        widgets.HTML("<b>Query</b>"),
        query_input
    ])

    top_k_slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description="Top K:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )

    rerank_property_dropdown = widgets.Dropdown(
        options=["title", "chunk"],
        value="title",
        description="Rerank Property:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )

    rerank_query_input = widgets.Text(
        value="",
        placeholder="Optional rerank query...",
        description="Rerank Query:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="700px"),
    )

    submit_button = widgets.Button(
        description="Generate Responses",
        icon="play",
        button_style="primary",
        layout=widgets.Layout(width="220px", height="40px"),
    )
    submit_button.on_click(on_button_click)

    status_output = widgets.Output()

    controls_row_1 = widgets.HBox(
        [top_k_slider, rerank_property_dropdown],
        layout=widgets.Layout(justify_content="flex-start", gap="20px"),
    )

    controls_row_2 = widgets.HBox(
        [rerank_query_input],
        layout=widgets.Layout(justify_content="flex-start"),
    )

    # =====================================================
    # OUTPUTS
    # =====================================================

    output_layout = widgets.Layout(
        border="1px solid #D9D9D9",
        padding="12px",
        margin="8px 0px 8px 0px",
        min_height="420px",
        max_height="420px",
        overflow_y="auto",
        width="100%",
    )

    outputs = {
        "semantic": widgets.Output(layout=output_layout),
        "rerank": widgets.Output(layout=output_layout),
        "bm25": widgets.Output(layout=output_layout),
        "hybrid": widgets.Output(layout=output_layout),
        "no_rag": widgets.Output(layout=output_layout),
    }

    tab = widgets.Tab(children=[
        outputs["semantic"],
        outputs["rerank"],
        outputs["bm25"],
        outputs["hybrid"],
        outputs["no_rag"],
    ])

    tab.set_title(0, "Semantic")
    tab.set_title(1, "Semantic + Rerank")
    tab.set_title(2, "BM25")
    tab.set_title(3, "Hybrid")
    tab.set_title(4, "Without RAG")

    # =====================================================
    # STYLING
    # =====================================================

    display(HTML("""
    <style>
        .jp-OutputArea-output {
            font-size: 14px;
            line-height: 1.5;
        }
        .widget-tab > .p-TabBar .p-TabBar-tab {
            font-weight: 600;
        }
    </style>
    """))

    header = widgets.HTML(
        """
        <div style="
            background: linear-gradient(90deg, #f8f9fa 0%, #eef3f8 100%);
            padding: 16px 20px;
            border: 1px solid #d9e2ec;
            border-radius: 10px;
            margin-bottom: 12px;">
            <h2 style="margin: 0; color: #1f2937;">RAG Retrieval Comparison</h2>
            <p style="margin: 6px 0 0 0; color: #4b5563;">
                Compare Semantic Search, BM25, Hybrid Search, Semantic Search with Reranking, and a baseline answer without RAG.
            </p>
        </div>
        """
    )

    controls_card = widgets.VBox(
        [
            query_box,
            controls_row_1,
            controls_row_2,
            widgets.HBox([submit_button], layout=widgets.Layout(margin="10px 0px 0px 0px")),
            status_output,
        ],
        layout=widgets.Layout(
            border="1px solid #D9D9D9",
            padding="16px",
            border_radius="10px",
            margin="0px 0px 16px 0px",
        ),
    )

    display(header)
    display(controls_card)
    display(tab)