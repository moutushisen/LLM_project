#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entity-Aware Memory Generator
Inspired by Cognee's structured memory approach

This module enhances memory compression by:
1. Extracting key entities/terms from conversations
2. Protecting important information during compression
3. Ensuring key concepts are preserved in the final memory
"""

from __future__ import annotations
from typing import List, Tuple, Set
import re
import os

try:
    from langchain_ollama import ChatOllama
    OLLAMA_OK = True
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama
        OLLAMA_OK = True
    except Exception:
        OLLAMA_OK = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_GEMINI_OK = True
except Exception:
    GOOGLE_GEMINI_OK = False


def extract_key_terms(text: str, max_terms: int = 25) -> Set[str]:
    """
    Extract key terms from text using rule-based approach.
    
    Extracts:
    - Capitalized words (likely names, places, concepts)
    - Technical terms (programming, ML, etc.)
    - Code snippets in backticks
    - Important concepts
    
    Args:
        text: Input text to extract terms from
        max_terms: Maximum number of terms to extract
        
    Returns:
        Set of key terms
    """
    terms = set()
    
    # 1. Capitalized words (names, places, concepts)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    terms.update(capitalized)
    
    # 2. Technical/Programming terms
    tech_patterns = [
        r'\b(?:class|function|method|algorithm|API|database|server|client)\b',
        r'\b(?:machine learning|neural network|deep learning|NLP|regression|classification)\b',
        r'\b(?:inheritance|polymorphism|encapsulation|abstraction)\b',
        r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|SQL|HTML|CSS)\b',
        r'\b(?:data structure|array|list|dict|set|tree|graph|hash)\b',
        r'\b(?:model|training|inference|embedding|vector|matrix)\b',
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        terms.update([m.lower() if m.islower() else m for m in matches])
    
    # 3. Code snippets in backticks
    code_snippets = re.findall(r'`([^`]+)`', text)
    # Extract identifiers from code (variable/function names)
    for snippet in code_snippets:
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', snippet)
        terms.update([id for id in identifiers if len(id) > 2])
    
    # 4. Quoted important terms
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    for q in quoted:
        term = q[0] or q[1]
        if term and len(term.split()) <= 3:  # Short phrases only
            terms.add(term)
    
    # Filter and prioritize
    filtered_terms = {
        term for term in terms 
        if len(term) > 2 and not term.lower() in ['the', 'and', 'for', 'with', 'this', 'that']
    }
    
    # Sort by length (longer terms often more specific) and limit
    sorted_terms = sorted(filtered_terms, key=len, reverse=True)[:max_terms]
    
    return set(sorted_terms)


def build_entity_aware_prompt(
    chat_pairs: List[Tuple[str, str]], 
    history_text: str, 
    max_chars: int
) -> str:
    """
    Build a compression prompt that preserves key entities.
    
    Args:
        chat_pairs: List of (user_msg, ai_msg) tuples
        history_text: Previous memory text
        max_chars: Maximum characters for compressed memory
        
    Returns:
        Formatted prompt for LLM
    """
    # Extract key entities from both new conversation and history
    all_text = "\n".join([f"{q} {a}" for q, a in chat_pairs]) + "\n" + history_text
    entities = extract_key_terms(all_text, max_terms=20)
    
    # Format entities for prompt
    if entities:
        entities_str = ", ".join(sorted(entities))
        entity_instruction = f"""
💡 Important Context: The following key terms/concepts are important for learning:
{entities_str}

Please try to naturally include these terms in the summary when relevant, as they represent key concepts being discussed.
"""
    else:
        entity_instruction = ""
    
    # Truncate inputs for prompt
    chat_excerpt = "\n\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_pairs])[:1500]
    history_excerpt = history_text[:max_chars] if history_text else "No previous memory."
    
    return f"""You are a memory compression assistant for a learning system.

{entity_instruction}
Historical Memory:
{history_excerpt}

New Conversation:
{chat_excerpt}

Task: Compress the above into a concise memory summary (max {max_chars} characters).

Guidelines:
1. Naturally incorporate the key terms mentioned above when they fit the context
2. Preserve important facts, concepts, and relationships
3. Write in a natural, flowing narrative style (avoid bullet points)
4. Focus on what the user is learning or needs to remember
5. Prioritize clarity and usefulness over strict term inclusion

Compressed Memory:"""


def build_entity_aware_compress_prompt(text: str, max_chars: int) -> str:
    """
    Build a secondary compression prompt that also preserves entities.
    
    Args:
        text: Pre-merged memory text to compress
        max_chars: Maximum characters for final memory
        
    Returns:
        Formatted prompt for LLM
    """
    entities = extract_key_terms(text, max_terms=15)
    
    if entities:
        entities_str = ", ".join(sorted(entities))
        entity_note = f"\n💡 Key concepts to retain if possible: {entities_str}"
    else:
        entity_note = ""
    
    return f"""You are compressing and deduplicating a memory summary.

Memory to Compress:
{text[:max_chars * 2]}
{entity_note}

Task: Merge, deduplicate, and compress into max {max_chars} characters.

Guidelines:
1. Remove redundant and repetitive information
2. Try to retain key terms/concepts{' mentioned above' if entity_note else ''} when they add value
3. Write in a natural, flowing narrative (avoid bullet points)
4. Maintain important relationships and context
5. Prioritize overall coherence and usefulness

Compressed Memory:"""


def _get_llm(model_name: str = None):
    """Get LLM client (Ollama or Google Gemini)"""
    if model_name and "gemini" in model_name:
        if not GOOGLE_GEMINI_OK:
            raise RuntimeError("Google Gemini dependencies unavailable")
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    else:
        if not OLLAMA_OK:
            raise RuntimeError("Ollama dependencies unavailable")
        local_model = model_name or os.getenv("DEFAULT_LOCAL_MODEL", "phi3:mini")
        return ChatOllama(model=local_model, temperature=0.0)


def generate_entity_aware_memory(
    chat_pairs: List[Tuple[str, str]],
    history_text: str,
    max_chars: int = 1200,
    model_name: str = None,
    verbose: bool = True
) -> str:
    """
    Generate memory with entity preservation.
    
    This is an enhanced version of the standard memory generator that:
    1. Extracts key entities from conversations
    2. Ensures they are preserved during compression
    3. Maintains better information fidelity
    
    Args:
        chat_pairs: List of (user_message, ai_response) tuples
        history_text: Previous memory text
        max_chars: Maximum characters for compressed memory
        model_name: LLM model name (ollama or gemini)
        verbose: Print progress messages
        
    Returns:
        Compressed memory text with preserved entities
    """
    if verbose:
        model_desc = model_name or os.getenv("DEFAULT_LOCAL_MODEL", "phi3:mini")
        print(f"🧠 Generating entity-aware memory using {model_desc}...")
    
    # Get LLM client
    llm = _get_llm(model_name)
    
    # Step 1: Generate initial memory with entity awareness
    prompt = build_entity_aware_prompt(chat_pairs, history_text, max_chars)
    resp = llm.invoke(prompt)
    new_memory = getattr(resp, "content", str(resp)).strip()
    
    # Step 2: Merge with history
    pre_merged = (history_text + "\n" + new_memory).strip() if history_text else new_memory
    
    # Step 3: Compress with entity preservation
    compress_prompt = build_entity_aware_compress_prompt(pre_merged, max_chars)
    resp2 = llm.invoke(compress_prompt)
    final_memory = getattr(resp2, "content", str(resp2)).strip()
    
    # Validation: Check if key entities are present
    if verbose:
        all_text = "\n".join([f"{q} {a}" for q, a in chat_pairs])
        original_entities = extract_key_terms(all_text, max_terms=10)
        preserved_entities = extract_key_terms(final_memory, max_terms=50)
        
        missing = original_entities - preserved_entities
        if missing:
            print(f"📝 Note: Some entities were compressed or rephrased: {', '.join(list(missing)[:5])}")
            print(f"   (This may be OK if the meaning is preserved)")
        else:
            print(f"✅ All key concepts successfully preserved!")
    
    if verbose:
        print("✅ Entity-aware memory generation completed")
    
    return final_memory


# Backward compatibility wrapper
def generate_merged_memory(
    chat_pairs: List[Tuple[str, str]],
    history_text: str,
    max_chars: int = 1200,
    model_name: str = None,
    entity_aware: bool = True  # New parameter
) -> str:
    """
    Wrapper that supports both old and new modes.
    
    Args:
        entity_aware: If True, use entity-aware generation (recommended)
    """
    if entity_aware:
        return generate_entity_aware_memory(
            chat_pairs, history_text, max_chars, model_name, verbose=True
        )
    else:
        # Fall back to original implementation
        from memory.generator import generate_merged_memory as original_generate
        return original_generate(chat_pairs, history_text, max_chars, model_name)


if __name__ == "__main__":
    # Test the entity extraction
    test_text = """
    I'm learning about Python classes and inheritance. 
    The `class` keyword is used to define a class.
    For example: `class Dog(Animal):` creates a Dog class inheriting from Animal.
    Multiple inheritance uses: `class Child(Parent1, Parent2):`
    The MRO (Method Resolution Order) determines which method is called.
    """
    
    print("Testing Entity Extraction:")
    print("-" * 50)
    entities = extract_key_terms(test_text)
    print(f"Extracted entities: {entities}")
    print(f"Total: {len(entities)} entities")

