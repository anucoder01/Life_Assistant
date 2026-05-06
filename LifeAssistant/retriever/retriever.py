# retriever/retriever.py — optimized, single retrieval call

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from embeddings.vector_store import VectorStore
from memory.long_term_memory import MemoryStore
from config import TOP_K_RESULTS
import time


class Retriever:
    def __init__(self, username: str, vector_store: VectorStore,
                 memory_store: MemoryStore):
        self.username = username
        self.vector_store = vector_store
        self.memory_store = memory_store

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> Dict:
        """
        Single retrieval call — returns everything needed.
        Call this ONCE per request, then pass result to build_context_from_retrieved.
        """
        t0 = time.time()

        # 1. Vector store search
        doc_results = self.vector_store.query(query, top_k=top_k)

        # 2. Memory search
        memory_results = self.memory_store.get_relevant_memories(
            query, username=self.username, top_k=3
        )

        # 3. Format document context — accept all results
        doc_context_parts = []
        sources_used = set()

        for result in doc_results:
            if result["relevance_score"] > 0.0:
                source = result["metadata"].get("source", "unknown")
                sources_used.add(source)
                doc_context_parts.append(f"[From {source}]\n{result['text']}")

        document_context = "\n\n---\n\n".join(doc_context_parts)

        # 4. Format memory context
        memory_parts = []
        for mem in memory_results:
            memory_parts.append(
                f"[Past conversation]\nYou asked: {mem['query']}\nI answered: {mem['response']}"
            )
        memory_context = "\n\n".join(memory_parts)

        elapsed = round((time.time() - t0) * 1000)
        print(f"🔍 Retrieved {len(doc_results)} chunks in {elapsed}ms | "
              f"Sources: {list(sources_used)}")

        return {
            "document_context": document_context,
            "memory_context": memory_context,
            "sources": list(sources_used),
            "raw_results": doc_results,
            "has_context": bool(document_context or memory_context)
        }

    def build_context_from_retrieved(self, retrieved: Dict) -> str:
        """
        Build prompt context string from already-retrieved results.
        Call this AFTER retrieve() — never call retrieve() twice.
        """
        parts = []
        if retrieved["document_context"]:
            parts.append(
                "=== RELEVANT DOCUMENTS FROM YOUR PERSONAL DATA ===\n"
                + retrieved["document_context"]
            )
        if retrieved["memory_context"]:
            parts.append(
                "=== RELEVANT PAST CONVERSATIONS ===\n"
                + retrieved["memory_context"]
            )
        return "\n\n".join(parts)

    def build_prompt_context(self, query: str) -> str:
        """Legacy method — calls retrieve() internally. Use retrieve() directly for speed."""
        retrieved = self.retrieve(query)
        return self.build_context_from_retrieved(retrieved)