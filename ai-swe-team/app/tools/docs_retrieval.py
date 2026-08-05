"""
ChromaDB-backed documentation retrieval tool (RAG).

Pre-populate the ChromaDB collection offline by chunking the official docs
for whatever frameworks your generated projects will use (FastAPI, React,
SQLAlchemy, etc.) and calling _collection.add() once before running the team.

Example population script (run once, offline):
    from app.tools.docs_retrieval import _collection
    docs = [("FastAPI route", "..."), ("SQLAlchemy session", "...")]
    _collection.add(
        documents=[d[1] for d in docs],
        ids=[f"doc_{i}" for i in range(len(docs))],
    )
"""

import logging

from langchain_core.tools import tool

from app.config import settings

logger = logging.getLogger(__name__)

_client = None
_collection = None
_embed_fn = None


def _get_embedding_function():
    """Return a ChromaDB-compatible embedding function using available API keys.

    Uses langchain-google-genai directly instead of ChromaDB's built-in Google
    wrapper, which has a compatibility issue with newer google-api-core versions.
    """
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

    # Prefer Google Gemini (free tier available)
    if settings.google_api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _langchain_embed = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.google_api_key,
        )

        class _GeminiEmbedFn(EmbeddingFunction):
            """Thin adapter: langchain embeddings → ChromaDB interface."""
            def __call__(self, input: Documents) -> Embeddings:
                return _langchain_embed.embed_documents(list(input))

        return _GeminiEmbedFn()

    # Fallback to OpenAI if set
    if settings.openai_api_key:
        from chromadb.utils import embedding_functions
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name="text-embedding-3-small",
        )

    # No embedding key available
    return None


def _get_collection():
    """Lazily initialise the ChromaDB client and collection on first use."""
    global _client, _collection, _embed_fn
    if _collection is None:
        _embed_fn = _get_embedding_function()
        if _embed_fn is None:
            logger.warning(
                "No embedding API key found (GOOGLE_API_KEY or OPENAI_API_KEY). "
                "RAG doc retrieval is disabled — agents will work without it."
            )
            return None

        import chromadb

        _client = chromadb.PersistentClient(path="./chroma_docs")
        _collection = _client.get_or_create_collection(
            "library_docs",
            embedding_function=_embed_fn,
        )
    return _collection


@tool
def search_docs(query: str, n_results: int = 5) -> list[str]:
    """
    Search the indexed library/framework documentation for snippets relevant
    to the given query. Returns up to n_results document chunks.

    Used by Backend Developer and Frontend Developer agents to look up real
    API signatures before writing code, reducing hallucinated method calls.

    Args:
        query: Natural-language description of the API or concept to look up.
        n_results: Maximum number of document chunks to return.

    Returns:
        List of matching document text chunks.
    """
    try:
        collection = _get_collection()
        if collection is None:
            return []
        results = collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        logger.warning("RAG doc search failed (non-fatal, agents continue without it): %s", e)
        return []

