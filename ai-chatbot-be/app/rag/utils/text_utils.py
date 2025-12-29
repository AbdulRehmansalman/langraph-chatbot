"""
Text Processing Utilities
=========================
Shared text processing utilities for the RAG pipeline.

This module centralizes text processing logic to eliminate duplication
across rag_chain.py, vector_store.py, and retriever.py.
"""

from typing import List, Set

# Comprehensive stopwords for query processing and relevance checking
# Used by: QueryPreprocessor, ResponseGenerator, retriever query expansion
STOPWORDS: Set[str] = {
    # Articles
    "the",
    "a",
    "an",
    # Be verbs
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    # Auxiliary verbs
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    # Modal verbs
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    # Question words
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
    # Demonstratives
    "this",
    "that",
    "these",
    "those",
    # Pronouns
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    # Conjunctions and prepositions
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "for",
    "to",
    "from",
    "with",
    "about",
    "into",
    "through",
}

# Minimal stopwords for keyword search (preserves more terms)
# Used by: VectorStoreService._extract_keywords, hybrid_search
BASIC_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "and",
    "or",
    "but",
    "if",
    "so",
    "as",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "why",
    "when",
    "where",
    "i",
    "me",
    "my",
    "you",
    "your",
    "he",
    "she",
    "it",
    "we",
    "they",
    "this",
    "that",
    "these",
    "those",
}


def extract_keywords(
    text: str, stopwords: Set[str] = None, min_length: int = 2, lowercase: bool = True
) -> List[str]:
    """
    Extract meaningful keywords from text.

    Args:
        text: Input text to process
        stopwords: Set of stopwords to filter (defaults to BASIC_STOPWORDS)
        min_length: Minimum keyword length
        lowercase: Whether to convert to lowercase

    Returns:
        List of extracted keywords
    """
    if stopwords is None:
        stopwords = BASIC_STOPWORDS

    # Clean and split
    cleaned = text.lower() if lowercase else text
    cleaned = cleaned.replace("?", "").replace(".", "").replace(",", "")
    cleaned = cleaned.replace("!", "").replace(";", "").replace(":", "")
    cleaned = cleaned.replace('"', "").replace("'", "")
    cleaned = cleaned.replace("(", "").replace(")", "")
    cleaned = cleaned.replace("[", "").replace("]", "")
    cleaned = cleaned.replace("{", "").replace("}", "")

    words = cleaned.split()

    # Filter stopwords and short words
    keywords = [
        w.strip()
        for w in words
        if w.strip() and len(w.strip()) > min_length and w.strip() not in stopwords
    ]

    return keywords


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Operations:
    - Remove extra whitespace
    - Normalize newlines
    - Strip leading/trailing whitespace

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


def calculate_word_overlap(text1: str, text2: str, stopwords: Set[str] = None) -> float:
    """
    Calculate keyword overlap ratio between two texts.

    Args:
        text1: First text
        text2: Second text
        stopwords: Set of stopwords to exclude (defaults to STOPWORDS)

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if stopwords is None:
        stopwords = STOPWORDS

    words1 = set(text1.lower().split()) - stopwords
    words2 = set(text2.lower().split()) - stopwords

    if not words1:
        return 0.5  # Default if no keywords

    overlap = len(words1 & words2)
    return overlap / len(words1)
