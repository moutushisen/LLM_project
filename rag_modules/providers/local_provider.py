from typing import Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    # Try new langchain-ollama package first
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    LOCAL_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old langchain-community package
        from langchain_community.chat_models import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings
        LOCAL_AVAILABLE = True
    except ImportError:
        LOCAL_AVAILABLE = False


def get_local_providers(model_name: str = "phi3:mini") -> Optional[Tuple[HuggingFaceEmbeddings, ChatOllama]]:
    """Return embeddings and chat LLM for local provider if available.

    Returns None if dependencies are missing.
    """
    if not LOCAL_AVAILABLE:
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        llm = ChatOllama(model=model_name, temperature=0.3)
        return embeddings, llm
    except Exception:
        return None


