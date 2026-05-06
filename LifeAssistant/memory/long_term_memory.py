# memory/long_term_memory.py
# ============================================================
# LONG-TERM MEMORY — Persists past Q&A pairs in SQLite.
#
# HOW IT WORKS:
# - Every time you ask a question and get an answer,
#   it's saved to a SQLite database.
# - Future queries are matched against past Q&A pairs
#   using simple text similarity.
# - This gives the assistant "memory" across sessions.
#
# WHY SQLITE:
# - Built into Python (no extra install)
# - Persistent across restarts
# - Per-user tables for privacy isolation
#
# WHAT'S STORED:
# - timestamp, username, query, response, tags
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from config import MEMORY_DB_PATH


class MemoryStore:
    """SQLite-based long-term memory for the life assistant."""

    def __init__(self):
        self.db_path = str(MEMORY_DB_PATH)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    session_id TEXT DEFAULT ''
                )
            """)
            # Index for fast user-specific queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_username
                ON memories(username)
            """)
            conn.commit()

    def save_interaction(self, username: str, query: str,
                        response: str, tags: List[str] = None,
                        session_id: str = "") -> int:
        """
        Save a Q&A pair to memory.

        Returns:
            ID of the saved memory
        """
        tags = tags or []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO memories (username, timestamp, query, response, tags, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                username,
                datetime.now().isoformat(),
                query,
                response,
                json.dumps(tags),
                session_id
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_memories(self, username: str, limit: int = 10) -> List[Dict]:
        """Get the N most recent interactions for a user."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM memories
                WHERE username = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (username, limit)).fetchall()

        return [dict(row) for row in rows]

    def get_relevant_memories(self, query: str, username: str,
                               top_k: int = 3) -> List[Dict]:
        """
        Find past memories relevant to the current query.
        Uses simple keyword matching (can be upgraded to embedding search).

        For a production system, you'd embed queries and do vector search here.
        For simplicity, we do keyword overlap scoring.
        """
        all_memories = self.get_recent_memories(username, limit=50)
        if not all_memories:
            return []

        query_words = set(query.lower().split())

        # Score each memory by keyword overlap
        scored = []
        for mem in all_memories:
            mem_words = set(mem["query"].lower().split())
            overlap = len(query_words & mem_words)
            if overlap > 0:
                scored.append((overlap, mem))

        # Sort by overlap score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def get_memory_summary(self, username: str) -> Dict:
        """Get statistics about stored memories for a user."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE username = ?",
                (username,)
            ).fetchone()[0]

            first = conn.execute(
                "SELECT MIN(timestamp) FROM memories WHERE username = ?",
                (username,)
            ).fetchone()[0]

            last = conn.execute(
                "SELECT MAX(timestamp) FROM memories WHERE username = ?",
                (username,)
            ).fetchone()[0]

        return {
            "total_interactions": count,
            "first_interaction": first,
            "last_interaction": last,
            "username": username
        }

    def clear_user_memory(self, username: str):
        """Delete all memories for a user."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memories WHERE username = ?", (username,))
            conn.commit()
        print(f"🗑️  Cleared memory for user: {username}")
