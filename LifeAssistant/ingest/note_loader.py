# ingest/note_loader.py
# ============================================================
# NOTE LOADER — Loads plain text and Markdown notes.
#
# Supports: .txt, .md files
# Same chunking logic as PDF loader.
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP
from ingest.pdf_loader import chunk_text


def load_note(file_path: str, username: str) -> List[Dict]:
    """Load a .txt or .md file and return chunked documents."""
    path = Path(file_path)
    print(f"📝 Loading note: {path.name}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()

    chunks = chunk_text(raw_text)
    documents = []

    for i, chunk in enumerate(chunks):
        documents.append({
            "text": chunk,
            "metadata": {
                "source": path.name,
                "type": "note",
                "user": username,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
        })

    print(f"   → {len(documents)} chunks from {path.name}")
    return documents


def load_notes_from_directory(directory: str, username: str) -> List[Dict]:
    """Load all .txt and .md files from a directory."""
    all_docs = []
    note_files = list(Path(directory).glob("**/*.txt")) + \
                 list(Path(directory).glob("**/*.md"))

    for note_file in note_files:
        docs = load_note(str(note_file), username)
        all_docs.extend(docs)

    return all_docs
