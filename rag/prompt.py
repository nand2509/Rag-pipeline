from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document




# The core RAG prompt template
RAG_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to
answer the question. If the answer is not in the context, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""


def build_prompt(query: str, retrieved_docs: list[Document]) -> str:
    """
    Combine retrieved chunks + user query into a single prompt string.
    This is what gets sent to the LLM.
    """
    # Join all retrieved chunks with a separator
    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_TEMPLATE,
    )

    filled = prompt.format(context=context, question=query)

    print(f"\n[Prompt] Built prompt ({len(filled)} chars)")
    print(f"[Prompt] Context from {len(retrieved_docs)} chunk(s)")

    return filled