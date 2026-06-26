# 🧠 LifeAssistant: RAG-Based Personal AI with Face Recognition

## 🌟 What is LifeAssistant?

LifeAssistant is a highly personalized, privacy-first **Artificial Intelligence Assistant**. It is not just another wrapper around an LLM; it is a full-fledged intelligent system designed to securely ingest your personal data, remember your past interactions, and provide grounded, context-aware assistance for your daily life, studies, and decision-making.

By combining **Retrieval-Augmented Generation (RAG)** with **local Face Recognition authentication**, LifeAssistant ensures your private life remains private while granting you a second brain that knows your schedules, notes, and documents inside out.

---

## 🏆 Why LifeAssistant? (Pros & Advantages over Existing Systems)

Compared to standard chatbot interfaces (like ChatGPT, standard Claude) or existing generic RAG implementations, LifeAssistant offers significant advantages:

1. **Uncompromised Privacy (Face Auth):** Standard apps rely on basic passwords or cloud OAuth. LifeAssistant uses local OpenCV face recognition to authenticate you before granting access to your sensitive personal data.
2. **Local Embedding & Vector Storage:** Unlike cloud-based vector databases that require uploading your private documents to third-party servers, LifeAssistant uses local `sentence-transformers` and a local `ChromaDB` instance. Your data is embedded and stored strictly on your local machine.
3. **True Long-Term Memory:** Standard AI chats forget you once the session ends or the context window fills up. LifeAssistant features a dedicated SQLite long-term memory layer, allowing it to recall past conversations and build a continuous understanding of your life over time.
4. **Agentic Decision Support:** Rather than just answering questions, LifeAssistant includes a dedicated Decision Support module tailored for actionable outputs like generating study plans, managing tasks, and optimizing daily routines based on your ingested schedules.
5. **Holistic Data Ingestion:** Natively handles multiple file formats (PDFs, Markdown/TXT notes, Schedule files) providing a unified "second brain."

---

## ✨ Key Features

- 🔐 **Biometric Security:** One-time face registration ensures only you can interact with your assistant.
- 📚 **Multi-Format Ingestion:** Seamlessly parse and load PDFs, text notes, and schedule formats.
- 🧠 **Retrieval-Augmented Generation (RAG):** Answers are grounded in *your* documents. It doesn't hallucinate; it reads your notes before speaking.
- 💾 **Persistent SQLite Memory:** Remembers past Q&A pairs to provide highly contextual follow-up answers across different days and sessions.
- 🤖 **Claude-Powered Generation:** Leverages the advanced reasoning capabilities of the Claude API for the final answer generation.
- 🌐 **Modern Web UI:** A clean, easy-to-use frontend (HTML/JS) interacting with a robust FastAPI backend.

---

## 🏗️ System Architecture

LifeAssistant operates on a sophisticated pipeline to ensure speed, accuracy, and security:

```text
[Your Face] → [OpenCV Face Auth] → [Access Granted]
                                          ↓
[Your Data: PDFs, Notes, Schedules]
        ↓
[Local Embedding Model: sentence-transformers]
        ↓
[Local Vector DB: ChromaDB]
        ↓
[User Query] → [Retriever] → [Top-K Relevant Chunks]
                                ↓
                    [LLM: Claude API] → [Response]
                                ↑
                    [Long-Term Memory: SQLite]
```

---

## 📁 Folder Structure

```text
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

---

## 🚀 Getting Started

1. **Install Dependencies:**
   Make sure you have Python 3.9+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configuration:**
   Set up your Claude API key and other local paths in `config.py`.
3. **Register Your Face:**
   Run `python face_auth/register_face.py` to securely store your facial encoding locally.
4. **Ingest Data:**
   Upload your PDFs and notes to `data/uploads/`, then run the ingestion scripts in the `ingest/` folder.
5. **Start the System:**
   Simply run the startup script to boot the backend and serve the UI:
   ```bash
   bash run.sh
   ```

---

## 🛠️ Technologies Used

- **Backend API:** FastAPI (Python)
- **Vector Database:** ChromaDB (Local)
- **Embeddings:** Sentence-Transformers (HuggingFace)
- **LLM API:** Anthropic Claude API
- **Biometrics:** OpenCV & face_recognition
- **Memory Storage:** SQLite
- **Frontend:** HTML, CSS, Vanilla JavaScript

---

## ©️ License & Copyright

**Copyright (c) 2026 Anu (anucoder01). All Rights Reserved.**

This project is proprietary and closed-source. You may not copy, modify, distribute, or use this code for any purpose without explicit written permission from the copyright holder.
