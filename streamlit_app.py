#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit GUI for the RAG Q&A System
"""

import streamlit as st
import os
import time
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Optional
from datetime import datetime, timedelta
from memory.rolling import RollingMemoryStorage

# Import our existing RAG modules
from rag_modules.app import SimpleRAGApp
from rag_modules.utils import pdf_utils

# Choose memory generator based on environment variable
if os.getenv('USE_ENTITY_AWARE_MEMORY', 'false').lower() == 'true':
    from memory.entity_aware_generator import generate_merged_memory
    MEMORY_MODE = "Entity-Aware"
else:
    from memory.generator import generate_merged_memory
    MEMORY_MODE = "Standard"

# Page configuration
st.set_page_config(
    page_title="Study Pal - Your Reading Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        height: 500px;
        overflow-y: auto;
    }
    
    .pdf-container {
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .status-info {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #5a6fd8;
    }
    
    /* Make the chat container scrollable */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div:nth-child(2) > .st-emotion-cache-1jicfl2 {
        height: 500px;
        overflow-y: auto;
        padding-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'rag_app' not in st.session_state:
        st.session_state.rag_app = SimpleRAGApp()
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'current_pdf_path' not in st.session_state:
        st.session_state.current_pdf_path = None
        
    if 'pdf_doc' not in st.session_state:
        st.session_state.pdf_doc = None
        
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
        
    if 'model_initialized' not in st.session_state:
        st.session_state.model_initialized = False

    # Memory system state
    if 'rolling_memory' not in st.session_state:
        st.session_state.rolling_memory = RollingMemoryStorage()
    
    # Flag to track if memory has been loaded into the current session
    if 'memory_loaded' not in st.session_state:
        st.session_state.memory_loaded = False
    
    # Store the loaded memory context
    if 'memory_context' not in st.session_state:
        st.session_state.memory_context = None

def load_memory_into_session():
    """Load memory into the current session (only once per session)"""
    if not st.session_state.memory_loaded:
        memory_text = st.session_state.rolling_memory.get_text()
        if memory_text and memory_text.strip():
            st.session_state.memory_context = memory_text
            st.session_state.memory_loaded = True
            print(f"📖 Memory loaded into session: {len(memory_text)} characters")
            return True
        else:
            st.session_state.memory_context = None
            st.session_state.memory_loaded = True
            print("📖 No memory found, starting fresh session")
            return False
    return st.session_state.memory_context is not None

def initialize_model():
    """Initialize the RAG model"""
    if not st.session_state.model_initialized:
        with st.spinner("Initializing model..."):
            # Check for API key
            if os.getenv("GOOGLE_API_KEY"):
                default_model = os.getenv("DEFAULT_GOOGLE_MODEL", "gemini-1.5-flash")
                success = st.session_state.rag_app.setup_google_model(default_model)
            else:
                default_model = os.getenv("DEFAULT_LOCAL_MODEL", "phi3:mini")
                success = st.session_state.rag_app.setup_local_model(default_model)
            
            if success:
                st.session_state.model_initialized = True
                st.success(f"Model initialized successfully: {st.session_state.rag_app.current_model}")
            else:
                st.error("Model initialization failed")

def load_pdf_preview(pdf_path: str):
    """Load PDF for preview"""
    try:
        doc = fitz.open(pdf_path)
        st.session_state.pdf_doc = doc
        st.session_state.current_pdf_path = pdf_path
        st.session_state.current_page = 0
        return True
    except Exception as e:
        st.error(f"Unable to load PDF file: {e}")
        return False

def render_pdf_page(page_num: int) -> Optional[Image.Image]:
    """Render a PDF page as an image"""
    if st.session_state.pdf_doc is None:
        return None
    
    try:
        page = st.session_state.pdf_doc[page_num]
        # Render page to image (higher DPI for better quality)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        st.error(f"Unable to render PDF page: {e}")
        return None

def display_chat_message(message: dict):
    """Display a chat message"""
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Reference Sources"):
                    for source in message["sources"]:
                        st.write(f"• {source}")

def summarize_and_store_memory():
    """Summarize and store memory from the current chat history."""
    max_chars = st.session_state.rolling_memory.max_chars
    with st.spinner("🧠 Generating and merging memory..."):
        progress = st.progress(0, text="Starting…")

        # Step 1: get history text and new chat pairs
        history_text = st.session_state.rolling_memory.get_text()
        messages = st.session_state.get("chat_history", [])
        
        # Use all messages to rebuild memory idempotently
        pairs = []
        buf = []
        for m in messages:
            role = m.get("role")
            
            raw_content = m.get("content", "")
            # Ensure content is always a string before stripping
            content_str = "\n".join(map(str, raw_content)) if isinstance(raw_content, list) else str(raw_content)

            if role == "user":
                if buf:
                    pairs.append((buf[0], ""))
                buf = [content_str.strip()]
            elif role == "assistant" and buf:
                pairs.append((buf[0], content_str.strip()))
                buf = []

        if not pairs and not history_text.strip():
            st.warning("No available session context and historical memory is empty. Cannot generate memory. Please chat with the assistant first.")
            progress.empty()
            return

        # Step 2: build inputs for generator
        progress.progress(25, text="Preparing prompts…")

        # Step 3: call decoupled memory generator (strictly local model)
        progress.progress(45, text="Calling model…")
        try:
            merged = generate_merged_memory(
                chat_pairs=pairs,
                history_text=history_text,
                max_chars=max_chars,
                model_name="gemini-2.5-pro-preview-03-25" # Force Gemini for memory generation
            )
        except Exception as gen_err:
            st.error(f"Memory generation failed: {gen_err}")
            raise

        st.session_state.rolling_memory.set_text(merged)
        try:
            save_path = st.session_state.rolling_memory.db_path
            print(f"💾 Saved to: {save_path}")
            # Read-back verification
            read_back = st.session_state.rolling_memory.get_text()
            print(f"🔎 Verification read length: {len(read_back)} characters")
            print(f"📄 Preview: {read_back[:120].replace('\n',' ')}{'…' if len(read_back)>120 else ''}")
        except Exception:
            pass
        
        progress.progress(100, text="Memory generation and merging completed")
        st.success("Memory updated")

        # Force sidebar editor to update
        st.session_state['mem_force_refresh'] = True
        st.rerun()

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Study Pal</h1>
        <p>Your Reading Helper - AI-Powered Document Assistant with Personalized Memory</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize model if not done
    initialize_model()
    
    # Sidebar: Memory management (simplified rolling memory)
    with st.sidebar:
        st.markdown("### 🧠 Memory (Rolling Text)")
        st.caption("Generate and merge on demand; length limited, auto-summarizes when exceeded")
        
        # Display memory mode
        if MEMORY_MODE == "Entity-Aware":
            st.info("🔬 Mode: **Entity-Aware** (preserves key terms)")
        else:
            st.info("📝 Mode: **Standard**")
        
        # Memory status indicator
        if st.session_state.memory_loaded:
            if st.session_state.memory_context:
                st.success(f"✅ Memory loaded ({len(st.session_state.memory_context)} chars)")
            else:
                st.info("ℹ️ No memory in current session")
        else:
            st.warning("⏳ Memory will load on first question")

        max_chars = st.number_input("Memory Length Limit (characters)", min_value=200, max_value=5000, value=1200, step=100)
        st.session_state.rolling_memory.max_chars = max_chars # Update max_chars in session state
        current_memory = st.session_state.rolling_memory.get_text()
        # If a refresh was requested, clear widget state so it picks up DB value
        if st.session_state.get('mem_force_refresh'):
            st.session_state.pop('rolling_mem_editor', None)
            st.session_state['mem_force_refresh'] = False
        st.text_area("Current Memory (Read-only)", value=current_memory, key="rolling_mem_editor", height=180, disabled=True)
        cols_mem = st.columns(3)
        with cols_mem[0]:
            pass # Removed "Save Memory" button
        with cols_mem[1]:
            if st.button("🗑️ Clear Memory", use_container_width=True):
                st.session_state.rolling_memory.clear()
                st.session_state['mem_force_refresh'] = True
                st.rerun()
        with cols_mem[2]:
            pass

        st.markdown("#### Generate/Merge Memory (Model-based)")
        st.caption("Compress current session Q&A into key points and merge with historical memory. Summarizes if too long.")
        if st.button("🧩 Generate & Merge", use_container_width=True):
            summarize_and_store_memory()

    # Main layout: two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 PDF Document Preview")
        
        # PDF upload section
        uploaded_file = st.file_uploader(
            "Upload PDF File", 
            type=['pdf'],
            help="Select a PDF file for Q&A"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily (cross-platform)
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load PDF into RAG system
            if st.session_state.current_pdf_path != temp_path:
                with st.spinner("Processing PDF file..."):
                    # Load memory on first interaction (if not already loaded)
                    load_memory_into_session()
                    
                    splits, pdf_path = pdf_utils.load_pdf(temp_path)
                    if splits:
                        st.session_state.rag_app.splits = splits
                        st.session_state.rag_app.current_pdf = pdf_path
                        
                        # Reinitialize model with new PDF and memory context
                        if st.session_state.rag_app.model_type == "google":
                            st.session_state.rag_app.setup_google_model(
                                st.session_state.rag_app.current_model or "gemini-1.5-flash",
                                st.session_state.memory_context
                            )
                        else:
                            st.session_state.rag_app.setup_local_model(
                                st.session_state.rag_app.current_model or "phi3:mini",
                                st.session_state.memory_context
                            )
                        
                        load_pdf_preview(temp_path)
                        st.success("PDF file loaded successfully!")
                        st.rerun()
        
        # PDF preview section
        if st.session_state.pdf_doc is not None:
            total_pages = len(st.session_state.pdf_doc)
            
            # Page navigation
            col_prev, col_info, col_next = st.columns([1, 2, 1])
            
            with col_prev:
                if st.button("⬅️ Previous") and st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with col_info:
                st.markdown(
                    f"<div style='text-align: center;'>Page {st.session_state.current_page + 1} / {total_pages}</div>",
                    unsafe_allow_html=True
                )
            
            with col_next:
                if st.button("Next ➡️") and st.session_state.current_page < total_pages - 1:
                    st.session_state.current_page += 1
                    st.rerun()
            
            # Display current page
            img = render_pdf_page(st.session_state.current_page)
            if img:
                st.image(img, use_container_width=True)
                
            # PDF info
            st.markdown(f"""
            <div class="status-info">
                📁 Filename: {os.path.basename(st.session_state.current_pdf_path)}<br>
                📄 Total Pages: {total_pages}<br>
                📊 Document Chunks: {len(st.session_state.rag_app.splits) if st.session_state.rag_app.splits else 0}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Please upload a PDF file to start using RAG functionality")
    
    with col2:
        st.markdown("### 💬 AI Chat Assistant")
        
        # Model status and controls
        with st.expander("🛠️ Model Settings", expanded=False):
            # Current model status
            current_model = st.session_state.rag_app.current_model or "Not initialized"
            model_type = st.session_state.rag_app.model_type or "Unknown"
            mode = "RAG Mode" if st.session_state.rag_app.retrieval_chain else "Chat Mode"
            
            st.markdown(f"""
            <div class="status-info">
                🤖 Current Model: <strong>{current_model}</strong> ({model_type})<br>
                ⚙️ Working Mode: <strong>{mode}</strong><br>
                🔗 Status: <strong>{'Ready' if st.session_state.model_initialized else 'Not Ready'}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Model switching buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🌐 Google Model", disabled=(st.session_state.rag_app.model_type == "google")):
                    if os.getenv("GOOGLE_API_KEY"):
                        with st.spinner("Switching to Google model..."):
                            # Preserve memory when switching models
                            success = st.session_state.rag_app.setup_google_model(
                                memory_context=st.session_state.memory_context
                            )
                            if success:
                                st.success("✅ Switched to Google model")
                            else:
                                st.error("❌ Google model switch failed")
                    else:
                        st.error("❌ Google API key not found")
                    st.rerun()
            
            with col2:
                if st.button("🏠 Local Model", disabled=(st.session_state.rag_app.model_type == "local")):
                    with st.spinner("Switching to local model..."):
                        # Preserve memory when switching models
                        success = st.session_state.rag_app.setup_local_model(
                            memory_context=st.session_state.memory_context
                        )
                        if success:
                            st.success("✅ Switched to local model")
                        else:
                            st.error("❌ Local model switch failed, please ensure Ollama is running")
                    st.rerun()
            
            # Advanced model selection (for Google models)
            if st.session_state.rag_app.model_type == "google":
                st.write("**Advanced Settings:**")
                available_models = st.session_state.rag_app.query_google_models()
                if available_models:
                    selected_model = st.selectbox(
                        "Select Google Model:", 
                        available_models,
                        index=available_models.index(current_model) if current_model in available_models else 0
                    )
                    if st.button("Apply Model"):
                        with st.spinner(f"Switching to {selected_model}..."):
                            # Preserve memory when changing model
                            success = st.session_state.rag_app.setup_google_model(
                                selected_model,
                                memory_context=st.session_state.memory_context
                            )
                            if success:
                                st.success(f"✅ Switched to {selected_model}")
                            else:
                                st.error("❌ Model switch failed")
                        st.rerun()
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                display_chat_message(message)
        
        # Chat input
        if prompt := st.chat_input("Enter your question..."):
            # Load memory on first question (if not already loaded)
            if not st.session_state.memory_loaded:
                load_memory_into_session()
                # If we have memory, we need to reinitialize the RAG chain with memory context
                if st.session_state.memory_context and st.session_state.rag_app.splits:
                    if st.session_state.rag_app.model_type == "google":
                        st.session_state.rag_app.setup_google_model(
                            st.session_state.rag_app.current_model or "gemini-1.5-flash",
                            st.session_state.memory_context
                        )
                    else:
                        st.session_state.rag_app.setup_local_model(
                            st.session_state.rag_app.current_model or "phi3:mini",
                            st.session_state.memory_context
                        )
            
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("AI is thinking..."):
                    try:
                        if st.session_state.rag_app.retrieval_chain:
                            # RAG mode
                            response = st.session_state.rag_app.retrieval_chain.invoke({"input": prompt})
                            answer = response.get("answer", "Sorry, I cannot answer this question.")
                            
                            # Extract sources
                            sources = []
                            for doc in response.get("context", []):
                                page_num = doc.metadata.get('page', doc.metadata.get('page_number', 'N/A'))
                                if page_num != 'N/A':
                                    page_num += 1
                                source_info = f"{os.path.basename(doc.metadata.get('source', 'Unknown file'))}, Page {page_num}"
                                sources.append(source_info)
                            
                            st.write(answer)
                            if sources:
                                with st.expander("📚 Reference Sources"):
                                    for source in set(sources):
                                        st.write(f"• {source}")
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": list(set(sources))
                            })
                        
                        elif st.session_state.rag_app.llm:
                            # Chat-only mode (with optional memory)
                            # Build prompt with memory if available
                            if st.session_state.memory_context and st.session_state.memory_context.strip():
                                chat_prompt = f"""You are a personalized AI assistant. You have the following memory about the user:

<memory>
{st.session_state.memory_context}
</memory>

Use this memory to provide personalized responses. Answer the user's question taking into account their background and preferences.

User question: {prompt}

Please answer in a natural, helpful way."""
                            else:
                                chat_prompt = prompt
                            
                            llm_response = st.session_state.rag_app.llm.invoke(chat_prompt)
                            content = getattr(llm_response, 'content', str(llm_response))
                            
                            st.write(content)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": content
                            })
                        
                        else:
                            st.error("Model not initialized, please check configuration")
                            
                    except Exception as e:
                        error_msg = f"Error processing question: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        # New session button (reset memory loading flag and reinject memory)
        if st.button("🔄 New Session (Reload Memory)"):
            # Clear chat history
            st.session_state.chat_history = []
            
            # Reset memory loading flags
            st.session_state.memory_loaded = False
            st.session_state.memory_context = None
            
            # Immediately load memory for the new session
            memory_loaded = load_memory_into_session()
            
            # Reinitialize model with new memory context (even in chat-only mode)
            if st.session_state.rag_app.model_type == "google":
                st.session_state.rag_app.setup_google_model(
                    st.session_state.rag_app.current_model or "gemini-1.5-flash",
                    st.session_state.memory_context
                )
            elif st.session_state.rag_app.model_type == "local":
                st.session_state.rag_app.setup_local_model(
                    st.session_state.rag_app.current_model or "phi3:mini",
                    st.session_state.memory_context
                )
            
            if memory_loaded:
                st.success(f"New session started! Memory loaded ({len(st.session_state.memory_context)} chars)")
            else:
                st.info("New session started! No memory found.")
            st.rerun()

if __name__ == "__main__":
    main()
