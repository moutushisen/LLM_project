#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Entity-Aware Memory Generation

This script demonstrates the difference between:
1. Standard memory compression (may lose entities)
2. Entity-aware compression (preserves key terms)

Run: python test_entity_aware_memory.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from memory.entity_aware_generator import (
    extract_key_terms,
    generate_entity_aware_memory,
)
from memory.generator import generate_merged_memory


def test_entity_extraction():
    """Test the entity extraction functionality"""
    print("=" * 80)
    print("TEST 1: Entity Extraction")
    print("=" * 80)
    
    test_cases = [
        {
            "name": "Python Programming",
            "text": """
            I'm learning Python classes and inheritance.
            Use `class Dog(Animal):` to create a subclass.
            Multiple inheritance: `class Child(Parent1, Parent2):`
            MRO means Method Resolution Order.
            """
        },
        {
            "name": "Machine Learning",
            "text": """
            Training a neural network with backpropagation.
            The model uses gradient descent for optimization.
            Loss function: Mean Squared Error (MSE).
            Regularization techniques include L1 and L2.
            """
        },
        {
            "name": "Web Development",
            "text": """
            Building a REST API with Flask and PostgreSQL.
            Using JWT for authentication.
            Frontend: React with TypeScript.
            Deployment on AWS EC2 with Docker.
            """
        }
    ]
    
    for case in test_cases:
        print(f"\n📝 {case['name']}:")
        print(f"   Text: {case['text'][:100]}...")
        entities = extract_key_terms(case['text'])
        print(f"   ✅ Extracted {len(entities)} entities:")
        print(f"      {', '.join(sorted(entities))}")
    
    print()


def test_memory_comparison():
    """Compare standard vs entity-aware memory generation"""
    print("=" * 80)
    print("TEST 2: Memory Compression Comparison")
    print("=" * 80)
    
    # Simulated conversation about Python
    chat_pairs = [
        ("What is inheritance in Python?", 
         "Inheritance allows a class to inherit attributes and methods from another class. Use class Child(Parent): syntax."),
        
        ("How does multiple inheritance work?", 
         "Python supports multiple inheritance with class Child(Parent1, Parent2). The MRO (Method Resolution Order) determines method resolution."),
        
        ("What's the difference between __init__ and __new__?",
         "__new__ creates the instance, __init__ initializes it. __new__ is called first and returns the instance.")
    ]
    
    history = "Previous session: Discussed Python basics, variables, and functions."
    
    print("\n📚 Conversation:")
    for i, (q, a) in enumerate(chat_pairs, 1):
        print(f"   {i}. Q: {q}")
        print(f"      A: {a[:80]}...")
    
    print(f"\n📜 History: {history}")
    
    # Extract entities from conversation
    all_text = "\n".join([f"{q} {a}" for q, a in chat_pairs])
    entities = extract_key_terms(all_text)
    print(f"\n🔑 Key Entities Detected: {', '.join(sorted(entities))}")
    
    print("\n" + "-" * 80)
    print("🔄 Generating Standard Memory...")
    print("-" * 80)
    
    try:
        standard_memory = generate_merged_memory(
            chat_pairs=chat_pairs,
            history_text=history,
            max_chars=500,
            model_name=None,  # Use default local model
        )
        print(f"\n📄 Standard Memory Result:")
        print(f"   {standard_memory}")
        print(f"   Length: {len(standard_memory)} chars")
        
        # Check which entities are preserved (case-insensitive + acronym aware)
        def is_entity_preserved(entity, text):
            text_lower = text.lower()
            # Direct match
            if entity.lower() in text_lower:
                return True
            # Check if it's an acronym of a phrase in text
            words = entity.split()
            if len(words) > 1:
                acronym = ''.join(w[0] for w in words).upper()
                if acronym in text.upper():
                    return True
            # Check if acronym matches a phrase
            if entity.isupper() and len(entity) > 1:
                # Entity is acronym, see if expanded form exists
                pass  # Would need reverse lookup
            return False
        
        preserved_standard = [e for e in entities if is_entity_preserved(e, standard_memory)]
        missing_standard = entities - set(preserved_standard)
        
        print(f"\n   ✅ Preserved entities: {', '.join(preserved_standard) if preserved_standard else 'None'}")
        if missing_standard:
            print(f"   ❌ Missing entities: {', '.join(missing_standard)}")
    
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print(f"   (This is expected if Ollama is not running)")
    
    print("\n" + "-" * 80)
    print("🧠 Generating Entity-Aware Memory...")
    print("-" * 80)
    
    try:
        entity_memory = generate_entity_aware_memory(
            chat_pairs=chat_pairs,
            history_text=history,
            max_chars=500,
            model_name=None,
            verbose=True
        )
        print(f"\n📄 Entity-Aware Memory Result:")
        print(f"   {entity_memory}")
        print(f"   Length: {len(entity_memory)} chars")
        
        # Check which entities are preserved (using same logic as above)
        preserved_entity = [e for e in entities if is_entity_preserved(e, entity_memory)]
        missing_entity = entities - set(preserved_entity)
        
        print(f"\n   ✅ Preserved entities: {', '.join(preserved_entity) if preserved_entity else 'None'}")
        if missing_entity:
            print(f"   ❌ Missing entities: {', '.join(missing_entity)}")
        
        # Summary
        print(f"\n📊 Comparison:")
        print(f"   Standard: {len(preserved_standard)}/{len(entities)} entities preserved")
        print(f"   Entity-Aware: {len(preserved_entity)}/{len(entities)} entities preserved")
        
        if len(preserved_entity) > len(preserved_standard):
            print(f"   ✨ Entity-aware method preserved {len(preserved_entity) - len(preserved_standard)} more entities!")
    
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print(f"   (This is expected if Ollama is not running)")


def test_entity_only():
    """Just test entity extraction without LLM"""
    print("=" * 80)
    print("TEST 3: Entity Extraction Only (No LLM Required)")
    print("=" * 80)
    
    conversations = [
        ("Explain gradient descent", 
         "Gradient descent is an optimization algorithm. It uses the derivative to find the minimum of a loss function."),
        ("What about Adam optimizer?",
         "Adam (Adaptive Moment Estimation) combines momentum and RMSprop. It adapts learning rates for each parameter."),
        ("How to prevent overfitting?",
         "Use regularization (L1, L2), dropout, early stopping, or cross-validation.")
    ]
    
    print("\n💬 Sample Conversation:")
    for q, a in conversations:
        print(f"   Q: {q}")
        print(f"   A: {a}")
    
    # Extract entities
    all_text = "\n".join([f"{q} {a}" for q, a in conversations])
    entities = extract_key_terms(all_text, max_terms=15)
    
    print(f"\n🔍 Extracted Entities:")
    for i, entity in enumerate(sorted(entities), 1):
        print(f"   {i:2d}. {entity}")
    
    print(f"\n✅ Total: {len(entities)} key terms extracted")
    print("\nThese entities would be protected during memory compression.")


def main():
    print("\n🧪 Entity-Aware Memory Testing Suite\n")
    
    # Test 1: Entity Extraction
    test_entity_extraction()
    
    # Test 2: Entity extraction only (no LLM needed)
    test_entity_only()
    
    # Test 3: Full comparison (requires Ollama)
    print("\n" + "=" * 80)
    print("Ready to test memory generation (requires Ollama)?")
    print("Make sure Ollama is running: ollama serve")
    print("=" * 80)
    
    try:
        response = input("\nProceed with memory generation test? (y/n): ")
        if response.lower() == 'y':
            test_memory_comparison()
        else:
            print("Skipped memory generation test.")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    
    print("\n" + "=" * 80)
    print("✅ Testing completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Entity extraction works without any LLM")
    print("2. Entity-aware compression preserves more key information")
    print("3. Useful for learning scenarios where terms/concepts matter")
    print("\nNext Steps:")
    print("- Integrate into your Streamlit app")
    print("- Enable/disable in settings")
    print("- View preserved entities in UI")
    print()


if __name__ == "__main__":
    main()

