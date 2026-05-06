# embeddings/vector_store.py
# ============================================================
# VECTOR STORE — ChromaDB interface for storing/querying vectors.
#
# HOW IT WORKS:
# - ChromaDB is a local vector database (like SQLite but for vectors)
# - We store each text chunk + its embedding vector + metadata
# - At query time, ChromaDB finds the N most similar vectors
#   using approximate nearest neighbor search
# - "Most similar" = smallest angle between vectors (cosine similarity)
#
# DATA ISOLATION:
# - Each user's data is stored in a separate ChromaDB collection
# - User "alice" cannot see user "bob"'s data
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from config import CHROMA_DIR, TOP_K_RESULTS
from embeddings.embedder import get_embedder


class VectorStore:
    """Interface to ChromaDB for storing and querying document embeddings."""

    def __init__(self, username: str):
        """
        Each user gets their own collection (isolated data).

        Args:
            username: The authenticated user's name
        """
        self.username = username

        # Initialize ChromaDB with persistent storage
        # This means data survives program restarts
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )

        # Collection name is user-specific for isolation
        collection_name = f"user_{username}"

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # Use cosine distance for semantic similarity
            metadata={"hnsw:space": "cosine"}
        )

        self.embedder = get_embedder()
        print(f"📦 Vector store ready for user: {username}")
        print(f"   → {self.collection.count()} documents currently stored")

    def add_documents(self, documents: List[Dict]) -> int:
        """
        Add documents (with their embeddings) to the vector store.

        Args:
            documents: List of {"text": "...", "metadata": {...}}

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        # Create unique IDs for each chunk
        # Format: user_source_chunkid
        ids = [
            f"{self.username}_{meta.get('source', 'unknown')}_{meta.get('chunk_id', i)}"
            for i, meta in enumerate(metadatas)
        ]

        print(f"   🔢 Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed_batch(texts)

        # Add to ChromaDB
        # ChromaDB expects lists, not numpy arrays
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

        print(f"   ✅ Added {len(texts)} chunks to vector store")
        return len(texts)

    def query(self, query_text: str, top_k: int = TOP_K_RESULTS,
              source_filter: Optional[str] = None) -> List[Dict]:
        """
        Find the most relevant documents for a query.

        Args:
            query_text: The user's question
            top_k: Number of results to return
            source_filter: Optional — only search in specific file

        Returns:
            List of {"text": "...", "metadata": {...}, "distance": 0.12}
        """
        # Embed the query using the same model
        query_embedding = self.embedder.embed(query_text)

        # Build optional filter
        where = None
        if source_filter:
            where = {"source": {"$eq": source_filter}}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                formatted.append({
                    "text": text,
                    "metadata": meta,
                    "relevance_score": 1 - dist  # Convert distance to similarity
                })

        return formatted

    def clear(self):
        """Remove all documents for this user."""
        self.client.delete_collection(f"user_{self.username}")
        print(f"🗑️  Cleared all data for user: {self.username}")

    def get_sources(self) -> List[str]:
        """List all ingested sources for this user."""
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if "source" in meta:
                sources.add(meta["source"])
        return sorted(list(sources))
