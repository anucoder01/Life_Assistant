# embeddings/embedder.py
# ============================================================
# EMBEDDING MODEL — Converts text to numerical vectors.
#
# HOW IT WORKS:
# - Uses sentence-transformers (all-MiniLM-L6-v2 model)
# - This model converts any text into a 384-dimensional vector
# - Similar texts end up with similar vectors (close in space)
# - This is what makes semantic search work:
#   "What's my study plan?" finds "Review ML notes at 3pm"
#   even though the words don't match exactly.
#
# WHY THIS MODEL:
# - Free, runs locally (no API cost)
# - Fast (~14,000 sentences/sec on CPU)
# - Good quality for RAG use cases
# - Small: ~90MB download
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from typing import List, Union
from config import EMBEDDING_MODEL
import numpy as np


class Embedder:
    """Wrapper around sentence-transformers for text embedding."""

    def __init__(self):
        print(f"🔢 Loading embedding model: {EMBEDDING_MODEL}")
        # This downloads the model on first run (~90MB), then caches it
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("   → Embedding model ready")

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text(s) to embedding vector(s).

        Args:
            text: Single string or list of strings

        Returns:
            numpy array of shape (384,) for single text,
            or (N, 384) for list of N texts
        """
        if isinstance(text, str):
            text = [text]

        # encode() returns numpy array
        embeddings = self.model.encode(
            text,
            show_progress_bar=len(text) > 10,  # Show progress for large batches
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        # Return single vector if single input was given
        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a large list of texts in batches."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )


# Singleton — load model once, reuse everywhere
_embedder_instance = None

def get_embedder() -> Embedder:
    """Get or create the global embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
