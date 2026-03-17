import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from rag.loader import load_text_file, load_from_string
from rag.splitter import split_documents
from rag.embedder import get_local_embeddings, get_openai_embeddings

from rag.vectorstore import (
    build_faiss_index,
    load_faiss_index,
    retrieve,
)

from rag.prompt import build_prompt

load_dotenv()


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage:
        rag = RAGPipeline()
        rag.index("data/my_document.txt")
        answer = rag.ask("What is Apache Spark?")
        print(answer)
    """

    def __init__(
        self,
        use_openai: bool = False,       # False = free local embeddings
        index_path: str = "faiss_index",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.vectorstore = None

        # Choose embeddings
        if use_openai:
            print("[Pipeline] Using OpenAI embeddings")
            self.embeddings = get_openai_embeddings()
        else:
            print("[Pipeline] Using local HuggingFace embeddings (free)")
            self.embeddings = get_local_embeddings()

        # Choose LLM
        if os.getenv("OPENAI_API_KEY"):
            print("[Pipeline] Using OpenAI GPT-3.5-turbo")
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            self.llm = None
            print("[Pipeline] No OPENAI_API_KEY found — LLM calls will be skipped")

    # ── INDEXING ─────────────────────────────────────────────────────────────

    def index(self, source, force_rebuild: bool = False):
        """
        Index a file or string.
        If an index already exists at self.index_path, loads it
        instead of rebuilding (unless force_rebuild=True).
        """
        # Load from cache if available
        if not force_rebuild and os.path.exists(self.index_path):
            print(f"[Pipeline] Loading existing index from '{self.index_path}'")
            self.vectorstore = load_faiss_index(self.index_path, self.embeddings)
            return self

        # Load documents
        if isinstance(source, str) and os.path.isfile(source):
            docs = load_text_file(source)
        elif isinstance(source, str):
            docs = load_from_string(source, source="inline_text")
        else:
            docs = source  # already a list of Documents

        # Split
        chunks = split_documents(docs, self.chunk_size, self.chunk_overlap)

        # Embed + store
        self.vectorstore = build_faiss_index(chunks, self.embeddings, self.index_path)

        return self  # allow chaining: rag.index(...).ask(...)

    # ── QUERYING ─────────────────────────────────────────────────────────────

    def ask(self, query: str) -> str:
        """
        Full RAG query:
          1. Embed the query
          2. Retrieve top-k similar chunks
          3. Build prompt (context + question)
          4. Call LLM
          5. Return answer string
        """
        if self.vectorstore is None:
            raise RuntimeError("Call .index() before .ask()")

        # Retrieve
        retrieved = retrieve(self.vectorstore, query, k=self.top_k)

        # Build prompt
        prompt_text = build_prompt(query, retrieved)

        # Call LLM
        if self.llm is None:
            # No LLM configured — return the raw prompt so you can inspect it
            print("\n[Pipeline] No LLM configured. Returning assembled prompt.")
            return prompt_text

        print("\n[Pipeline] Calling LLM...")
        response = self.llm([HumanMessage(content=prompt_text)])
        answer = response.content

        print(f"\n[Pipeline] Answer:\n{answer}")
        return answer

    # ── CONVENIENCE ──────────────────────────────────────────────────────────

    def add_text(self, text: str, source: str = "added"):
        """Add new text to an existing index without rebuilding from scratch."""
        from rag.loader import load_from_string
        from rag.splitter import split_documents

        docs   = load_from_string(text, source=source)
        chunks = split_documents(docs, self.chunk_size, self.chunk_overlap)
        self.vectorstore.add_documents(chunks)
        self.vectorstore.save_local(self.index_path)
        print(f"[Pipeline] Added {len(chunks)} new chunk(s) to index")