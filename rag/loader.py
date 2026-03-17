import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document


def load_text_file(file_path: str) -> list[Document]:
    """Load a single .txt file."""
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    print(f"[Loader] Loaded {len(docs)} document(s) from {file_path}")
    return docs


def load_pdf_file(file_path: str) -> list[Document]:
    """Load a single PDF file (each page = one Document)."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"[Loader] Loaded {len(docs)} page(s) from {file_path}")
    return docs


def load_directory(folder_path: str, glob: str = "C://Users//Preneel//Pictures//Rag_Pipeline//data//sample.txt") -> list[Document]:
    """Load all matching files from a directory."""
    loader = DirectoryLoader(folder_path, glob=glob)
    docs = loader.load()
    print(f"[Loader] Loaded {len(docs)} file(s) from {folder_path}")
    return docs


def load_from_string(text: str, source: str = "manual") -> list[Document]:
    """Create a Document directly from a Python string."""
    doc = Document(page_content=text, metadata={"source": source})
    return [doc]