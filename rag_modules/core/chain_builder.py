from typing import Any, Tuple, Optional

from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(splits: list, embeddings: Any, llm: Any, memory_context: Optional[str] = None) -> Tuple[Any, Any]:
    """Create and return (vectorstore, retrieval_chain).
    
    Args:
        splits: Document splits to embed
        embeddings: Embedding model
        llm: Language model
        memory_context: Optional memory/context to personalize the AI assistant
    """
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Build prompt with optional memory context
    if memory_context and memory_context.strip():
        prompt_template = f"""You are a personalized document reading assistant. You have the following memory about the user and previous interactions:

<memory>
{memory_context}
</memory>

Use this memory to provide more personalized and contextual responses. Answer the question based on the provided context. If there is no relevant information in the context, say "I don't know".

<context>
{{context}}
</context>

Question: {{input}}

Please answer in English, taking into account both the memory and the document context.
"""
    else:
        prompt_template = """
Answer the question based on the provided context. If there is no relevant information in the context, say "I don't know".

<context>
{context}
</context>

Question: {input}

Please answer in English.
"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return vectorstore, retrieval_chain


