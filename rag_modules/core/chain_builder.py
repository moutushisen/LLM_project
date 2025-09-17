from typing import Any, Tuple

from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(splits: list, embeddings: Any, llm: Any) -> Tuple[Any, Any]:
    """Create and return (vectorstore, retrieval_chain)."""
    vectorstore = FAISS.from_documents(splits, embeddings)

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the provided context. If there is no relevant information in the context, say "I don't know".

<context>
{context}
</context>

Question: {input}

Please answer in English.
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return vectorstore, retrieval_chain


