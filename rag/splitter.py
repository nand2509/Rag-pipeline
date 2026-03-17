from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split documents into overlapping chunks.

    chunk_size    = max characters per chunk
    chunk_overlap = characters shared between consecutive chunks
                    (helps avoid losing context at boundaries)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    print(f"[Splitter] {len(docs)} docs → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    # Print a preview of the first chunk
    if chunks:
        print(f"[Splitter] First chunk preview:\n  '{chunks[0].page_content[:120]}...'")

    return chunks