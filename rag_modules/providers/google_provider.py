import os
from typing import List, Optional, Tuple

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except Exception:
    GOOGLE_AVAILABLE = False


def query_google_models() -> List[str]:
    """Query available Google chat-capable models via Google Generative AI API.

    Returns an empty list if dependencies or API key are missing.
    """
    if not GOOGLE_AVAILABLE:
        return []

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return []

    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        chat_models: List[str] = []
        for model in models:
            if 'generateContent' in getattr(model, 'supported_generation_methods', []):
                name = getattr(model, 'name', '')
                if name:
                    chat_models.append(name.replace('models/', ''))
        return chat_models
    except Exception:
        # Fallback to a common subset
        return [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]


def get_google_providers(model_name: str = "gemini-1.5-flash") -> Optional[Tuple[GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI]]:
    """Return embeddings and chat LLM for Google provider if available.

    Returns None if dependencies are missing.
    """
    if not GOOGLE_AVAILABLE:
        return None

    try:
        # First try to query available models to ensure the model exists
        available_models = query_google_models()
        
        # If the requested model is not available, fall back to a working model
        if available_models and model_name not in available_models:
            print(f"⚠️  Model '{model_name}' not available, trying available models...")
            # Try common working models in order
            fallback_models = ["gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
            for fallback in fallback_models:
                if fallback in available_models:
                    model_name = fallback
                    print(f"✅ Using model: {model_name}")
                    break
            else:
                # Use the first available model if no fallbacks work
                if available_models:
                    model_name = available_models[0]
                    print(f"✅ Using first available model: {model_name}")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        return embeddings, llm
    except Exception as e:
        print(f"Google provider initialization failed: {e}")
        return None


