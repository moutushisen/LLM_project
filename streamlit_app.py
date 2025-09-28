#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit GUI for the RAG Q&A System
"""

import streamlit as st
import os
import time
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Optional

# Import our existing RAG modules
from rag_modules.app import SimpleRAGApp
from rag_modules.utils import pdf_utils

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
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

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Q&A System</h1>
        <p>Intelligent Document Q&A System - PDF Preview + AI Chat Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize model if not done
    initialize_model()
    
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
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load PDF into RAG system
            if st.session_state.current_pdf_path != temp_path:
                with st.spinner("Processing PDF file..."):
                    splits, pdf_path = pdf_utils.load_pdf(temp_path)
                    if splits:
                        st.session_state.rag_app.splits = splits
                        st.session_state.rag_app.current_pdf = pdf_path
                        
                        # Reinitialize model with new PDF
                        if st.session_state.rag_app.model_type == "google":
                            st.session_state.rag_app.setup_google_model(
                                st.session_state.rag_app.current_model or "gemini-1.5-flash"
                            )
                        else:
                            st.session_state.rag_app.setup_local_model(
                                st.session_state.rag_app.current_model or "phi3:mini"
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
                            success = st.session_state.rag_app.setup_google_model()
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
                        success = st.session_state.rag_app.setup_local_model()
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
                            success = st.session_state.rag_app.setup_google_model(selected_model)
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
                            # Chat-only mode
                            llm_response = st.session_state.rag_app.llm.invoke(prompt)
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

if __name__ == "__main__":
    main()
