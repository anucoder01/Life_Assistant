# retriever/retriever.py — optimized, single retrieval call

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from embeddings.vector_store import VectorStore
from memory.long_term_memory import MemoryStore
from config import TOP_K_RESULTS, RERANK_CANDIDATE_MULTIPLIER, RERANKER_MODEL
import time
import datetime
import re
import threading
import math


class CrossEncoderReranker:
    """Locally rerank vector-search candidates with a pretrained cross-encoder."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(RERANKER_MODEL)

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def scores(self, query: str, results: list) -> list:
        pairs = [(query, item.get("text", "")) for item in results]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)
        scores = []
        for raw in raw_scores:
            value = float(raw)
            # MS MARCO CrossEncoder emits logits; map them to [0, 1].
            value = max(-50.0, min(50.0, value))
            scores.append(1.0 / (1.0 + math.exp(-value)))
        return scores


def extract_date_from_query(query: str) -> str:
    """Helper to extract a target date from the query string (returns YYYY-MM-DD)."""
    query_lower = query.lower()
    
    # Check for direct date match like YYYY-MM-DD
    match = re.search(r'\b\d{4}-\d{2}-\d{2}\b', query_lower)
    if match:
        return match.group(0)
        
    today = datetime.date.today()
    
    if 'today' in query_lower:
        return today.strftime('%Y-%m-%d')
    elif 'tomorrow' in query_lower:
        tomorrow = today + datetime.timedelta(days=1)
        return tomorrow.strftime('%Y-%m-%d')
    elif 'yesterday' in query_lower:
        yesterday = today - datetime.timedelta(days=1)
        return yesterday.strftime('%Y-%m-%d')
        
    days_of_week = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    for day, target_weekday in days_of_week.items():
        if day in query_lower:
            current_weekday = today.weekday()
            days_ahead = target_weekday - current_weekday
            if days_ahead < 0:
                days_ahead += 7
            target_date = today + datetime.timedelta(days=days_ahead)
            return target_date.strftime('%Y-%m-%d')
            
    return None


class Retriever:
    def __init__(self, username: str, vector_store: VectorStore,
                 memory_store: MemoryStore):
        self.username = username
        self.vector_store = vector_store
        self.memory_store = memory_store

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> Dict:
        """
        Single retrieval call — returns chronologically reranked and boosted context.
        """
        t0 = time.time()

        # Fetch a wider candidate set, then use the cross-encoder to score the
        # query and each chunk together. Keep vector search as a fallback.
        candidate_k = max(top_k, top_k * RERANK_CANDIDATE_MULTIPLIER)
        doc_results = self.vector_store.query(query, top_k=candidate_k)
        try:
            if doc_results:
                neural_scores = CrossEncoderReranker.get().scores(query, doc_results)
                for result, neural_score in zip(doc_results, neural_scores):
                    result["reranker_score"] = neural_score
                    result["relevance_score"] = (
                        0.75 * neural_score + 0.25 * result["relevance_score"]
                    )
        except Exception as exc:
            print(f"Cross-encoder reranker unavailable; using vector ranking: {exc}")

        # 2. Memory search
        memory_results = self.memory_store.get_relevant_memories(
            query, username=self.username, top_k=3
        )

        # Rerank document results based on time decay and query date-boost
        current_time = time.time()
        reranked_results = []
        target_date = extract_date_from_query(query)
        
        for result in doc_results:
            score = result["relevance_score"]
            meta = result.get("metadata", {})
            mtime = meta.get("timestamp")
            text = result.get("text", "")
            
            # Time Decay (0.5% score decay per day)
            if mtime:
                age_days = max(0.0, (current_time - mtime) / 86400.0)
                decay_factor = 2.71828 ** (-0.005 * age_days)
                decay_factor = max(0.2, decay_factor)  # Bounded
                score = score * decay_factor
                
            # Date Boost (+0.3 relevance score if date matches query)
            if target_date:
                try:
                    dt = datetime.datetime.strptime(target_date, "%Y-%m-%d")
                    date_formats = [
                        target_date,
                        dt.strftime("%Y-%m-%d"),
                        dt.strftime("%d-%m-%Y"),
                        dt.strftime("%b %d"),
                        dt.strftime("%B %d"),
                        dt.strftime("%d %b"),
                        dt.strftime("%d %B"),
                        f"{dt.month}/{dt.day}",
                        f"{dt.day}/{dt.month}",
                    ]
                    if any(fmt.lower() in text.lower() for fmt in date_formats):
                        score += 0.3
                except Exception:
                    pass
                    
            result["adjusted_score"] = score
            reranked_results.append(result)
            
        # Re-sort results by adjusted score
        reranked_results.sort(key=lambda x: x["adjusted_score"], reverse=True)
        reranked_results = reranked_results[:top_k]

        # 3. Format document context
        doc_context_parts = []
        sources_used = []

        for result in reranked_results:
            if result.get("adjusted_score", result["relevance_score"]) > 0.0:
                source = result["metadata"].get("source", "unknown")
                if source not in sources_used:
                    sources_used.append(source)
                
                # Include page number in citation if available
                page_str = ""
                if "page" in result["metadata"]:
                    page_str = f" Page {result['metadata']['page']}"
                
                doc_context_parts.append(f"[From {source}{page_str}]\n{result['text']}")

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
              f"Sources: {sources_used}")

        return {
            "document_context": document_context,
            "memory_context": memory_context,
            "sources": sources_used,
            "raw_results": reranked_results,
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
