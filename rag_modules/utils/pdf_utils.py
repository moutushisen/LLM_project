import os
from typing import List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def find_pdf_files(directory: str = '.') -> List[str]:
    pdf_files: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file_name))
    return pdf_files


def show_pdf_files() -> List[str]:
    pdf_files = find_pdf_files()
    if not pdf_files:
        print("No PDF files found in current directory and subdirectories")
        return []

    print(f"\nFound {len(pdf_files)} PDF files:")
    for i, pdf_file in enumerate(pdf_files, 1):
        display_path = os.path.relpath(pdf_file) if pdf_file.startswith(os.getcwd()) else pdf_file
        print(f"  {i}. {display_path}")
    return pdf_files


def select_pdf_interactive() -> Optional[str]:
    pdf_files = show_pdf_files()
    if not pdf_files:
        return None
    try:
        choice = input("\nSelect PDF file to load (enter number): ").strip()
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(pdf_files):
            return pdf_files[choice_idx]
        print("Invalid selection")
        return None
    except ValueError:
        print("Please enter a valid number")
        return None


def load_pdf(pdf_path: Optional[str] = None) -> Tuple[Optional[list], Optional[str]]:
    """Load a PDF into split documents, returning (splits, pdf_path)."""
    if not pdf_path:
        print("No PDF path provided. Use /files command to view available PDFs, then /load to select one.")
        return None, None

    print(f"Loading PDF: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print(f"PDF loaded, {len(splits)} document chunks")
        return splits, pdf_path
    except Exception as e:
        print(f"Failed to load PDF: {e}")
        return None, None


