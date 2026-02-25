"""
formatting.py contains small utility functions used across the project
to format/pretty-print data and load datasets with consistent date formatting.

Functions
---------
pprint_json:
    Pretty-print a Python object as JSON (useful for debugging dict/list outputs).

format_date:
    Parse a date string and return it formatted as "YYYY-MM-DD".

read_dataframe:
    Read a CSV file into a list of dictionaries and format date columns
    ('published_at' and 'updated_at') to "YYYY-MM-DD".

format_relevant_data:
    Retrieves the top_k most relevant documents based on a given query and constructs an augmented prompt for a RAG system.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from dateutil import parser


def pprint_json(obj: Any, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Pretty-print a Python object as JSON.

    Parameters
    ----------
    obj:
        Any JSON-serializable Python object (dict, list, etc.).
    indent:
        JSON indentation level.
    ensure_ascii:
        If False, keeps unicode characters readable (recommended for French text).

    Returns
    -------
    None
    """
    print(json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii))


def format_date(date_string: str, *, output_format: str = "%Y-%m-%d") -> str:
    """
    Parse a date string and return it formatted.

    Parameters
    ----------
    date_string:
        Date string to parse (e.g. "2024-01-15T10:30:00Z", "2024-01-15", etc.).
    output_format:
        strftime format for the output date. Default is "YYYY-MM-DD".

    Returns
    -------
    str
        Formatted date string.

    Raises
    ------
    ValueError
        If the date_string cannot be parsed.
    """
    date_object = parser.parse(date_string)
    return date_object.strftime(output_format)


def read_dataframe(
    path: str,
    *,
    date_columns: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Read a CSV file into a list of records and format date columns.

    This function is designed for the news dataset where date columns
    should be normalized to "YYYY-MM-DD" for consistent filtering/sorting.

    Parameters
    ----------
    path:
        Path to the CSV file.
    date_columns:
        List of columns to format as dates. If None, defaults to
        ["published_at", "updated_at"].

    Returns
    -------
    list[dict[str, Any]]
        List of rows as dictionaries (records).
    """
    if date_columns is None:
        date_columns = ["published_at", "updated_at"]

    df = pd.read_csv(path)

    for col in date_columns:
        if col in df.columns:
            # Fill NaN to avoid parser errors, then format
            df[col] = df[col].fillna("").astype(str)
            df[col] = df[col].apply(lambda x: format_date(x) if x else x)

    return df.to_dict(orient="records")

def format_relevant_data(relevant_data):
    """
    Retrieves the top_k most relevant documents based on a given query and constructs an augmented prompt for a RAG system.

    Parameters:
    relevant_data (list): A list with relevant data.

    Returns:
    str: An augmented prompt with the top_k relevant documents, formatted for use in a Retrieval-Augmented Generation (RAG) system."
    """

    # Create a list so store the formatted documents
    formatted_documents = []
    
    # Iterates over each relevant document.
    for document in relevant_data:

        # Formats each document into a structured layout string. Remember that each document is in one different line. So you should add a new line character after each document added.
        formatted_document = json.dumps(document, indent=2, ensure_ascii=False)
        
        # Append the formatted document string to the formatted_documents list
        formatted_documents.append(formatted_document)
        
    # Returns the final augmented prompt string.
    return "\n".join(formatted_documents)
