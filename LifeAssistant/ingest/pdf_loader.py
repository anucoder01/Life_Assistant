# ingest/pdf_loader.py
# ============================================================
# PDF LOADER — Extracts and chunks text from PDF files.
#
# HOW IT WORKS:
# 1. Uses PyMuPDF (fitz) to extract text page-by-page
# 2. Splits the full text into overlapping chunks
#    - Why chunks? LLMs have context limits. We embed each
#      chunk separately so we can retrieve just the relevant part.
#    - Why overlap? So sentences at chunk boundaries aren't
#      split apart and lose meaning.
# 3. Returns list of {text, metadata} dicts ready for embedding
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n[Page {page_num + 1}]\n{text}"
    doc.close()
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.

    Example with chunk_size=20, overlap=5:
    Text: "Hello world this is a test of chunking"
    Chunk 1: "Hello world this is a"  (chars 0-20)
    Chunk 2: "is a test of chunking"  (chars 15-35)
                ^^^^^ overlapping part
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap  # Step forward, keeping overlap
    return chunks


def load_pdf(pdf_path: str, username: str) -> List[Dict]:
    """
    Load a PDF and return chunked documents with metadata.

    Returns list of:
    {
        "text": "chunk content...",
        "metadata": {
            "source": "filename.pdf",
            "type": "pdf",
            "user": "alice",
            "chunk_id": 3
        }
    }
    """
    path = Path(pdf_path)
    print(f"📄 Loading PDF: {path.name}")

    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "text": chunk,
            "metadata": {
                "source": path.name,
                "type": "pdf",
                "user": username,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
        })

    print(f"   → {len(documents)} chunks extracted from {path.name}")
    return documents


def load_pdfs_from_directory(directory: str, username: str) -> List[Dict]:
    """Load all PDFs from a directory."""
    all_documents = []
    pdf_files = list(Path(directory).glob("**/*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {directory}")
        return []

    for pdf_file in pdf_files:
        docs = load_pdf(str(pdf_file), username)
        all_documents.extend(docs)

    print(f"\n✅ Loaded {len(all_documents)} total chunks from {len(pdf_files)} PDFs")
    return all_documents
