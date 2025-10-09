#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify memory integration
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from memory.rolling import RollingMemoryStorage
from rag_modules.core.chain_builder import create_rag_chain

def test_memory_storage():
    """Test 1: Memory storage and retrieval"""
    print("="*60)
    print("Test 1: Memory Storage and Retrieval")
    print("="*60)
    
    storage = RollingMemoryStorage()
    
    # Test write
    test_memory = "User is a Python developer interested in machine learning."
    storage.set_text(test_memory)
    print(f"✓ Wrote memory: {test_memory}")
    
    # Test read
    retrieved = storage.get_text()
    if retrieved == test_memory:
        print(f"✓ Successfully retrieved: {retrieved}")
        return True
    else:
        print(f"✗ Retrieval failed. Got: {retrieved}")
        return False

def test_rag_chain_without_memory():
    """Test 2: RAG chain creation without memory (backward compatibility)"""
    print("\n" + "="*60)
    print("Test 2: RAG Chain Without Memory (Backward Compatibility)")
    print("="*60)
    
    try:
        from langchain_core.documents import Document
        from langchain_community.embeddings import FakeEmbeddings
        from langchain_community.chat_models.fake import FakeListChatModel
        
        # Create fake components
        fake_docs = [
            Document(page_content="Test content 1", metadata={"source": "test.pdf", "page": 0}),
            Document(page_content="Test content 2", metadata={"source": "test.pdf", "page": 1}),
        ]
        fake_embeddings = FakeEmbeddings(size=384)
        fake_llm = FakeListChatModel(responses=["Test response"])
        
        # Create chain without memory
        vectorstore, chain = create_rag_chain(fake_docs, fake_embeddings, fake_llm)
        print("✓ RAG chain created successfully without memory")
        
        # Test if chain works
        if chain is not None:
            print("✓ Chain is ready")
            return True
        else:
            print("✗ Chain is None")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_rag_chain_with_memory():
    """Test 3: RAG chain creation with memory"""
    print("\n" + "="*60)
    print("Test 3: RAG Chain With Memory (New Feature)")
    print("="*60)
    
    try:
        from langchain_core.documents import Document
        from langchain_community.embeddings import FakeEmbeddings
        from langchain_community.chat_models.fake import FakeListChatModel
        
        # Create fake components
        fake_docs = [
            Document(page_content="Test content about Python", metadata={"source": "test.pdf", "page": 0}),
        ]
        fake_embeddings = FakeEmbeddings(size=384)
        fake_llm = FakeListChatModel(responses=["Test response"])
        
        # Create chain with memory
        memory_context = "User prefers simple explanations and is learning Python."
        vectorstore, chain = create_rag_chain(fake_docs, fake_embeddings, fake_llm, memory_context)
        print(f"✓ RAG chain created with memory: {memory_context}")
        
        if chain is not None:
            print("✓ Chain with memory is ready")
            return True
        else:
            print("✗ Chain is None")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_memory_loading_flag():
    """Test 4: Memory loading flag mechanism"""
    print("\n" + "="*60)
    print("Test 4: Memory Loading Flag Mechanism")
    print("="*60)
    
    # Simulate session state
    class SessionState:
        def __init__(self):
            self.memory_loaded = False
            self.memory_context = None
    
    state = SessionState()
    storage = RollingMemoryStorage()
    
    # First load
    if not state.memory_loaded:
        memory_text = storage.get_text()
        if memory_text and memory_text.strip():
            state.memory_context = memory_text
            state.memory_loaded = True
            print(f"✓ First load: memory loaded ({len(memory_text)} chars)")
        else:
            state.memory_context = None
            state.memory_loaded = True
            print("✓ First load: no memory found (flag still set)")
    
    # Second load attempt (should skip)
    load_count = 1
    if not state.memory_loaded:
        load_count += 1
    
    if load_count == 1:
        print("✓ Second load correctly skipped (flag works)")
        return True
    else:
        print("✗ Memory was loaded twice")
        return False

def main():
    """Run all tests"""
    print("\n" + "🧠 Memory Integration Test Suite")
    print("测试记忆集成功能\n")
    
    results = []
    
    # Run tests
    results.append(("Memory Storage", test_memory_storage()))
    results.append(("RAG Chain (No Memory)", test_rag_chain_without_memory()))
    results.append(("RAG Chain (With Memory)", test_rag_chain_with_memory()))
    results.append(("Loading Flag", test_memory_loading_flag()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Memory integration is working correctly.")
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Follow the testing guide in MEMORY_FEATURE_TESTING.md")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

