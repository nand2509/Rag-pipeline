import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain.embeddings.base import Embeddings


# ── FAISS (local, in-memory, no server needed) ─────────────────────────────

def build_faiss_index(
    chunks: list[Document],
    embeddings: Embeddings,
    save_path: str = "faiss_index",
) -> FAISS:
    """Build a FAISS vector store from chunks and save to disk."""
    print(f"[VectorStore] Building FAISS index from {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"[VectorStore] FAISS index saved to '{save_path}/'")
    return vectorstore


def load_faiss_index(
    save_path: str,
    embeddings: Embeddings,
) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"No FAISS index found at '{save_path}'")
    vectorstore = FAISS.load_local(
        save_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[VectorStore] FAISS index loaded from '{save_path}/'")
    return vectorstore


# ── Chroma (persistent, SQLite-backed, easier to inspect) ──────────────────

def build_chroma_index(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_dir: str = "chroma_db",
    collection_name: str = "rag_collection",
) -> Chroma:
    """Build a Chroma vector store that persists to disk automatically."""
    print(f"[VectorStore] Building Chroma index from {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    print(f"[VectorStore] Chroma index saved to '{persist_dir}/'")
    return vectorstore


def load_chroma_index(
    persist_dir: str,
    embeddings: Embeddings,
    collection_name: str = "rag_collection",
) -> Chroma:
    """Load an existing Chroma collection."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    print(f"[VectorStore] Chroma index loaded from '{persist_dir}/'")
    return vectorstore


# ── Retrieval helper ────────────────────────────────────────────────────────

def retrieve(
    vectorstore,
    query: str,
    k: int = 3,
) -> list[Document]:
    """Find the top-k most similar chunks for a query."""
    results = vectorstore.similarity_search(query, k=k)
    print(f"[Retriever] Query: '{query}'")
    print(f"[Retriever] Found {len(results)} chunks:")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content[:100]}...")
    return results