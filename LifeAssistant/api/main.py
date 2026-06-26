# api/main.py — optimized for low latency

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List
import shutil
import uuid
import traceback
import json

from config import UPLOADS_DIR
from embeddings.vector_store import VectorStore
from memory.long_term_memory import MemoryStore
from retriever.retriever import Retriever
from llm.generator import LLMGenerator
from decision.decision_support import DecisionSupport
from ingest.pdf_loader import load_pdf
from ingest.note_loader import load_note
from ingest.schedule_loader import load_schedule

app = FastAPI(title="Life Assistant API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_store = MemoryStore()
llm = LLMGenerator()
_vector_stores = {}


def normalize(username: str) -> str:
    return username.strip().lower()


def get_vector_store(username: str) -> VectorStore:
    username = normalize(username)
    if username not in _vector_stores:
        _vector_stores[username] = VectorStore(username)
    return _vector_stores[username]


# ── Pre-warm DeepFace at startup ──────────────────────────
# This loads FaceNet model into memory when server starts,
# so the FIRST face scan is fast instead of taking 3+ seconds.
@app.on_event("startup")
async def startup_event():
    print("\n🚀 Pre-warming face recognition model...")
    try:
        from face_auth.authenticate import get_model, load_registered_faces
        get_model()           # Load FaceNet into memory
        load_registered_faces()  # Cache face encodings
        print("✅ Face model ready\n")
    except Exception as e:
        print(f"⚠️  Face model pre-warm failed: {e}\n")


# ── Request models ────────────────────────────────────────

class ChatRequest(BaseModel):
    username: str
    message: str
    conversation_history: List[dict] = []

class SourceChunk(BaseModel):
    filename: str
    page: int = None
    text: str
    score: float

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    source_chunks: List[SourceChunk] = []
    intent: str
    interaction_id: int

class FaceAuthRequest(BaseModel):
    image_base64: str

class FaceRegisterRequest(BaseModel):
    username: str
    images_base64: List[str]


# ── Face Auth Endpoints ───────────────────────────────────

@app.post("/auth/verify-face")
async def verify_face(request: FaceAuthRequest):
    """Fast browser-based face auth — model pre-loaded at startup."""
    try:
        import time
        t0 = time.time()
        from face_auth.authenticate import authenticate_from_image
        username = authenticate_from_image(request.image_base64)
        elapsed = round((time.time() - t0) * 1000)
        print(f"   Auth took: {elapsed}ms")

        if username:
            return {"status": "authenticated", "username": username, "latency_ms": elapsed}
        raise HTTPException(status_code=401, detail="Face not recognized.")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/register-browser")
async def register_face_browser(request: FaceRegisterRequest):
    try:
        from face_auth.register_face import register_from_images
        username = normalize(request.username)
        result = register_from_images(username, request.images_base64)
        if result["success"]:
            return result
        raise HTTPException(status_code=400, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/registered-users")
async def get_registered_users():
    from config import FACES_DIR
    users = [f.stem for f in FACES_DIR.glob("*.npy")]
    return {"registered_users": users}


@app.post("/auth/authenticate")
async def authenticate():
    """Legacy webcam auth."""
    try:
        from face_auth.authenticate import authenticate as face_auth
        username = face_auth(timeout_seconds=15)
        if username:
            return {"status": "authenticated", "username": username}
        raise HTTPException(status_code=401, detail="Face not recognized.")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve Uploaded Files ──────────────────────────────────

@app.get("/files/{username}/{filename}")
async def get_uploaded_file(username: str, filename: str):
    username = normalize(username)
    filename = os.path.basename(filename)  # security: prevent path traversal
    file_path = UPLOADS_DIR / username / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# ── Ingest ────────────────────────────────────────────────

@app.post("/ingest")
async def ingest_file(username: str = Form(...), file: UploadFile = File(...)):
    username = normalize(username)
    user_dir = UPLOADS_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    suffix = file.filename.lower().split(".")[-1]
    try:
        if suffix == "pdf":
            documents = load_pdf(str(file_path), username)
        elif suffix in ["txt", "md"]:
            documents = load_note(str(file_path), username)
        elif suffix == "json":
            documents = load_schedule(str(file_path), username)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported: {suffix}")

        vs = get_vector_store(username)
        count = vs.add_documents(documents)
        return {"status": "success", "filename": file.filename, "chunks_ingested": count}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Streaming Chat — fixed double retrieval ───────────────

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        import time
        t0 = time.time()
        try:
            username = normalize(request.username)
            query = request.message

            # Decision support
            decision = DecisionSupport(username)
            decision_result = decision.process(query)
            enriched_query = decision_result["enriched_query"]

            # Retrieve ONCE — fixed the double retrieval bug
            vs = get_vector_store(username)
            retriever = Retriever(username, vs, memory_store)
            retrieved = retriever.retrieve(enriched_query)  # single call
            context = retriever.build_context_from_retrieved(retrieved)
            sources = retrieved["sources"]

            print(f"   Retrieval: {round((time.time()-t0)*1000)}ms | "
                  f"Context: {len(context)} chars | Sources: {sources}")

            # Extract source chunks for frontend preview
            source_chunks = []
            for r in retrieved.get("raw_results", []):
                if r.get("adjusted_score", r["relevance_score"]) > 0.0:
                    source_chunks.append({
                        "filename": r["metadata"].get("source", "unknown"),
                        "page": r["metadata"].get("page"),
                        "text": r.get("text", ""),
                        "score": round(r.get("adjusted_score", r["relevance_score"]), 3)
                    })

            # Send sources immediately
            yield f"data: {json.dumps({'type': 'sources', 'data': sources, 'chunks': source_chunks, 'intent': decision_result['intent']})}\n\n"
            history = [dict(m) for m in request.conversation_history]
            full_response = ""

            for chunk in llm.generate_stream(
                query=enriched_query, context=context,
                username=username, conversation_history=history
            ):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

            # Save to memory
            interaction_id = memory_store.save_interaction(
                username=username, query=query, response=full_response,
                tags=[decision_result["intent"]], session_id=str(uuid.uuid4())[:8]
            )

            total = round((time.time() - t0) * 1000)
            print(f"   Total response time: {total}ms")

            yield f"data: {json.dumps({'type': 'done', 'interaction_id': interaction_id})}\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        username = normalize(request.username)
        query = request.message

        decision = DecisionSupport(username)
        decision_result = decision.process(query)
        enriched_query = decision_result["enriched_query"]

        vs = get_vector_store(username)
        retriever = Retriever(username, vs, memory_store)
        retrieved = retriever.retrieve(enriched_query)
        context = retriever.build_context_from_retrieved(retrieved)
        sources = retrieved["sources"]

        history = [dict(m) for m in request.conversation_history]
        response_text = llm.generate(
            query=enriched_query, context=context,
            username=username, conversation_history=history
        )

        interaction_id = memory_store.save_interaction(
            username=username, query=query, response=response_text,
            tags=[decision_result["intent"]], session_id=str(uuid.uuid4())[:8]
        )

        source_chunks = []
        for r in retrieved.get("raw_results", []):
            if r.get("adjusted_score", r["relevance_score"]) > 0.0:
                source_chunks.append({
                    "filename": r["metadata"].get("source", "unknown"),
                    "page": r["metadata"].get("page"),
                    "text": r.get("text", ""),
                    "score": round(r.get("adjusted_score", r["relevance_score"]), 3)
                })

        return ChatResponse(
            response=response_text, sources=sources,
            source_chunks=source_chunks,
            intent=decision_result["intent"], interaction_id=interaction_id
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ── Other endpoints ───────────────────────────────────────

@app.get("/debug/{username}")
async def debug_user(username: str):
    username = normalize(username)
    vs = get_vector_store(username)
    count = vs.collection.count()
    test_results = vs.query("summary overview experience skills", top_k=3)
    return {
        "username": username, "total_chunks": count,
        "sources": vs.get_sources(),
        "test_retrieval": [
            {"score": round(r["relevance_score"], 3),
             "source": r["metadata"].get("source"),
             "text_preview": r["text"][:100]}
            for r in test_results
        ]
    }

@app.get("/memory/{username}")
async def get_memory(username: str, limit: int = 20):
    username = normalize(username)
    return {
        "memories": memory_store.get_recent_memories(username, limit=limit),
        "summary": memory_store.get_memory_summary(username)
    }

@app.delete("/memory/{username}")
async def clear_memory(username: str):
    memory_store.clear_user_memory(normalize(username))
    return {"status": "success"}

@app.get("/sources/{username}")
async def get_sources(username: str):
    username = normalize(username)
    vs = get_vector_store(username)
    return {"username": username, "sources": vs.get_sources()}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "groq/llama-3.3-70b",
            "streaming": True, "browser_face_auth": True}

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT)