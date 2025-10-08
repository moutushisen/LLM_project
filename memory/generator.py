#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple
import os

try:
    # Preferred new package
    from langchain_ollama import ChatOllama
    OLLAMA_OK = True
except Exception:
    try:
        # Fallback to legacy community import
        from langchain_community.chat_models import ChatOllama  # type: ignore
        OLLAMA_OK = True
    except Exception:
        OLLAMA_OK = False

# Try import Google Generative AI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GEMINI_OK = True
except Exception:
    GOOGLE_GEMINI_OK = False


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def build_summary_prompt(chat_pairs: List[Tuple[str, str]], history_text: str, max_chars: int) -> str:
    chat_excerpt = _truncate("\n\n".join([f"U: {q}\nA: {a}" for q, a in chat_pairs]), 2000)
    return (
        "You are the user's learning partner or teacher. Your task is to help the user remember important learning content. Remember like a human, do not use bullet points or overly formal formats.\n\n"
        "Please condense the following conversation and historical memory points into natural, brief text, key phrases, or keywords. The goal is to create a concise, readable memory summary, not a formal report.\n\n"
        "Remember, you are the user's learning partner, so the tone should be natural and friendly.\n\n"
        "Output in English.\n\n"
        f"Historical Memory (may be empty):\n{_truncate(history_text, max_chars)}\n\n"
        f"Recent Session (marked U/A):\n{chat_excerpt}\n\n"
        "New Memory:"
    )


def build_compress_prompt(text: str, max_chars: int) -> str:
    return (
        "You are the user's learning partner or teacher. Please merge, deduplicate, and further compress the following memory content into a natural, more information-dense short text. Remember like a human, do not use bullet points or overly formal formats. "
        f"Final content must be within {max_chars} characters. Output in English.\n\nMemory to Compress:\n{_truncate(text, max_chars * 2)}\n\nCompressed Memory:"
    )


def _get_local_llm(model_name: str) -> ChatOllama:
    if not OLLAMA_OK:
        raise RuntimeError("Local model dependencies unavailable. Please install Ollama and langchain-ollama/community package")
    return ChatOllama(model=model_name, temperature=0.3)

def _get_google_llm(model_name: str) -> ChatGoogleGenerativeAI:
    if not GOOGLE_GEMINI_OK:
        raise RuntimeError("Google Gemini dependencies unavailable. Please install langchain-google-genai")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.3)


def generate_merged_memory(
    chat_pairs: List[Tuple[str, str]],
    history_text: str,
    max_chars: int = 1200,
    model_name: str = None,
) -> str:
    """Generate memory strictly via local model and merge with history.

    - Uses only ChatOllama; no dependency on RAG app/providers.
    - Prints start/complete logs to console; caller should print save path after persisting.
    """
    # Determine which model to use
    if model_name and "gemini" in model_name:
        print(f"🧠 Starting to generate memory using Google model: {model_name}…")
        llm = _get_google_llm(model_name)
        print(f"✅ Successfully started using Google model: {model_name}")
    else:
        local_model_name = model_name or os.getenv("DEFAULT_LOCAL_MODEL", "phi3:mini")
        print(f"🧠 Starting to generate memory using local model: {local_model_name}…")
        llm = _get_local_llm(local_model_name)
        print(f"✅ Successfully started using local model: {local_model_name}")

    prompt = build_summary_prompt(chat_pairs, history_text, max_chars)
    resp = llm.invoke(prompt)
    new_points = getattr(resp, "content", str(resp)).strip()

    pre_merged = (history_text + "\n" + new_points).strip() if history_text else new_points

    # Always run a deduplicate + compress pass to avoid duplication/accumulation
    compress_prompt = build_compress_prompt(pre_merged, max_chars)
    resp2 = llm.invoke(compress_prompt)
    merged_final = getattr(resp2, "content", str(resp2)).strip()

    print("✅ Local model memory generation completed")
    return merged_final


