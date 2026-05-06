# 🧠 RAG-Based Life Assistant with Face Recognition

## What This System Does

This is a **personal intelligence assistant** — not just a chatbot. It:
1. **Recognizes your face** before giving access (privacy-first)
2. **Ingests your personal data** — PDFs, notes, schedules, past chats
3. **Converts everything to vectors** and stores in a local vector DB
4. **Retrieves the most relevant context** when you ask a question
5. **Generates grounded answers** using Claude AI + your personal data
6. **Remembers past interactions** across sessions (long-term memory)
7. **Helps with decisions** — study plans, task management, daily routines

---

## System Architecture

```
[Your Face] → [OpenCV Face Auth] → [Access Granted]
                                          ↓
[Your Data: PDFs, Notes, Schedules]
        ↓
[Embedding Model: sentence-transformers]
        ↓
[Vector DB: ChromaDB (local)]
        ↓
[Query] → [Retriever] → [Top-K Relevant Chunks]
                                ↓
                    [LLM: Claude API] → [Response]
                                ↑
                    [Long-Term Memory: SQLite]
```

---

## Folder Structure

```
LifeAssistant/
├── face_auth/
│   ├── register_face.py     # One-time: register your face
│   └── authenticate.py      # Runtime: verify face before access
├── ingest/
│   ├── pdf_loader.py        # Extract text from PDFs
│   ├── note_loader.py       # Load .txt / .md notes
│   └── schedule_loader.py   # Parse schedule files
├── embeddings/
│   ├── embedder.py          # Sentence-transformer embedding
│   └── vector_store.py      # ChromaDB interface
├── retriever/
│   └── retriever.py         # Fetch top-K relevant chunks
├── memory/
│   └── long_term_memory.py  # SQLite: store past Q&A pairs
├── llm/
│   └── generator.py         # Claude API call with context
├── decision/
│   └── decision_support.py  # Study/task/routine planning layer
├── api/
│   └── main.py              # FastAPI backend (REST endpoints)
├── frontend/
│   └── index.html           # Web UI
├── data/
│   ├── faces/               # Stored face encodings per user
│   ├── uploads/             # User-uploaded PDFs, notes
│   └── memory.db            # SQLite memory database
├── config.py                # All configuration
├── requirements.txt         # Python dependencies
└── run.sh                   # One-command startup script
```
