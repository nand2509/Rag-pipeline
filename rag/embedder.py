import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_openai_embeddings() -> OpenAIEmbeddings:
    """
    Use OpenAI's text-embedding-3-small model.
    Requires OPENAI_API_KEY in your .env file.
    Best quality, but costs money per token.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_local_embeddings(
    model_name: str = "all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    """
    Use a FREE local HuggingFace model — no API key needed.
    all-MiniLM-L6-v2 is fast, small, and good for most use cases.
    Downloads the model (~90MB) on first run.
    """
    print(f"[Embedder] Loading local model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_query(embeddings_model, text: str) -> list[float]:
    """Embed a single query string — useful for testing."""
    vector = embeddings_model.embed_query(text)
    print(f"[Embedder] Query embedded → vector dim: {len(vector)}")
    return vector