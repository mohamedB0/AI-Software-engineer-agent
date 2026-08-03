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

from langchain_core.tools import tool

from app.config import settings

_client = None
_collection = None


def _get_collection():
    """Lazily initialise the ChromaDB client and collection on first use."""
    global _client, _collection
    if _collection is None:
        import chromadb
        from chromadb.utils import embedding_functions

        _client = chromadb.PersistentClient(path="./chroma_docs")
        _embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name="text-embedding-3-small",
        )
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
    results = _get_collection().query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else []
