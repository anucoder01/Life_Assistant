# ingest/schedule_loader.py
# ============================================================
# SCHEDULE LOADER — Parses schedule/todo files.
#
# Supports JSON schedules in this format:
# {
#   "schedule": [
#     {"date": "2024-01-15", "time": "09:00", "task": "Study ML", "priority": "high"},
#     {"date": "2024-01-15", "time": "14:00", "task": "Gym", "priority": "medium"}
#   ]
# }
#
# Also supports plain .txt schedule files.
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import List, Dict


def load_json_schedule(file_path: str, username: str) -> List[Dict]:
    """Load a JSON schedule file."""
    path = Path(file_path)
    print(f"📅 Loading schedule: {path.name}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    documents = []
    # Convert each schedule item into a text chunk
    items = data.get("schedule", data if isinstance(data, list) else [])

    mtime = os.path.getmtime(file_path)
    for i, item in enumerate(items):
        # Convert the schedule item to natural language text
        if isinstance(item, dict):
            text_parts = []
            for key, value in item.items():
                text_parts.append(f"{key}: {value}")
            text = "Schedule entry - " + ", ".join(text_parts)
        else:
            text = str(item)

        documents.append({
            "text": text,
            "metadata": {
                "source": path.name,
                "type": "schedule",
                "user": username,
                "chunk_id": i,
                "timestamp": mtime
            }
        })

    print(f"   → {len(documents)} schedule entries loaded")
    return documents


def load_schedule(file_path: str, username: str) -> List[Dict]:
    """Auto-detect format and load schedule."""
    path = Path(file_path)
    if path.suffix == '.json':
        return load_json_schedule(file_path, username)
    else:
        # Treat as plain text
        from ingest.note_loader import load_note
        return load_note(file_path, username)
