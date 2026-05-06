# ============================================================
# COMPLETE SETUP & EXECUTION GUIDE
# Life Assistant — RAG + Face Recognition
# ============================================================

# ══════════════════════════════════════════════════════════
# PART 1: SYSTEM PREREQUISITES
# ══════════════════════════════════════════════════════════

# ── Step 1: Install Python 3.10+ ──────────────────────────
# Check your version first:
python3 --version
# If < 3.10, install from https://www.python.org/downloads/

# ── Step 2: Install CMake and dlib dependencies ───────────
# dlib is required by face_recognition. This is the trickiest part.

# On macOS:
brew install cmake
brew install boost
brew install boost-python3

# On Ubuntu/Debian Linux:
sudo apt-get update
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev
sudo apt-get install -y libx11-dev libgtk-3-dev python3-dev

# On Windows:
# 1. Install Visual Studio Build Tools from:
#    https://visualstudio.microsoft.com/visual-cpp-build-tools/
# 2. Install CMake from https://cmake.org/download/
# 3. Add CMake to your PATH

# ── Step 3: Install a webcam ──────────────────────────────
# Face recognition requires a physical webcam connected to
# the machine where the Python server runs.
# If running on a remote server, you'll need to run
# face_auth scripts locally (see alternative setup below).


# ══════════════════════════════════════════════════════════
# PART 2: PROJECT SETUP
# ══════════════════════════════════════════════════════════

# ── Step 4: Navigate to project folder ────────────────────
cd LifeAssistant

# ── Step 5: Create a virtual environment ──────────────────
# This isolates your project dependencies from system Python.
python3 -m venv venv

# Activate it:
source venv/bin/activate          # macOS / Linux
# OR on Windows:
# venv\Scripts\activate

# You should now see (venv) in your terminal prompt.

# ── Step 6: Install all dependencies ──────────────────────
pip install --upgrade pip
pip install -r requirements.txt

# ⏱️  This takes 3-10 minutes. What's happening:
# - face_recognition: compiles dlib from source (longest step)
# - sentence-transformers: downloads ~90MB model
# - chromadb: sets up local vector database
# - anthropic: Claude API client
# - fastapi + uvicorn: web server

# If face_recognition fails on macOS with Apple Silicon (M1/M2/M3):
pip install face_recognition --no-binary :all:

# If dlib compilation fails on Linux, try:
pip install dlib --verbose
# If still failing: sudo apt-get install libdlib-dev

# ── Step 7: Set your Anthropic API key ────────────────────
# Get your key from: https://console.anthropic.com/keys
# Then set it as an environment variable:

export ANTHROPIC_API_KEY="sk-ant-api03-YOUR-KEY-HERE"

# To make this permanent (so you don't need to re-set every session):
# Add the above line to your ~/.bashrc or ~/.zshrc:
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-YOUR-KEY-HERE"' >> ~/.zshrc
source ~/.zshrc


# ══════════════════════════════════════════════════════════
# PART 3: REGISTER YOUR FACE (One-time setup per user)
# ══════════════════════════════════════════════════════════

# ── Step 8: Register your face ────────────────────────────
# Make sure you're in the LifeAssistant/ directory with venv active.
# Replace "yourname" with your actual username (lowercase, no spaces).

python face_auth/register_face.py yourname

# What happens:
# 1. Your webcam opens in a new window
# 2. A green rectangle appears around your face
# 3. Press SPACEBAR to start capturing
# 4. It captures 10 frames of your face
# 5. Averages them into a single 128-number vector
# 6. Saves it to data/faces/yourname.npy
# 7. This file is YOUR face — keep it private

# Tips for good registration:
# - Sit in good, even lighting (no strong backlight)
# - Look directly at the camera
# - Normal expression (how you'd normally look at your screen)
# - Remove glasses if you sometimes wear them and register twice:
#   python face_auth/register_face.py yourname_glasses
#   python face_auth/register_face.py yourname_noglasses

# To register multiple users:
python face_auth/register_face.py alice
python face_auth/register_face.py bob
# Each gets a completely separate data partition in the system.


# ══════════════════════════════════════════════════════════
# PART 4: TEST FACE AUTHENTICATION
# ══════════════════════════════════════════════════════════

# ── Step 9: Test authentication standalone ────────────────
python face_auth/authenticate.py

# Expected output:
# 🔒 Life Assistant — Face Authentication
#    Registered users: ['yourname']
#    Look at the camera... (15s timeout)
# ✅ Authenticated as: yourname
# Welcome, yourname! Access granted.

# If you see "Unknown (0.6+)" — the distance is too high.
# Try re-registering in better lighting:
# python face_auth/register_face.py yourname
# You can adjust sensitivity in config.py: FACE_TOLERANCE = 0.5
# Lower = stricter (0.4), Higher = more lenient (0.6)


# ══════════════════════════════════════════════════════════
# PART 5: START THE SERVER
# ══════════════════════════════════════════════════════════

# ── Step 10: Start the FastAPI backend ────────────────────
python api/main.py

# Expected output:
# 🔢 Loading embedding model: all-MiniLM-L6-v2
#    → Embedding model ready
# 🤖 LLM ready: claude-sonnet-4-20250514
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

# The server is now running at http://localhost:8000
# API documentation auto-generated at: http://localhost:8000/docs

# ── Step 11: Open the frontend ────────────────────────────
# Open a NEW terminal tab (keep the server running in the first one)
# Then simply open the HTML file in your browser:

# macOS:
open frontend/index.html

# Linux:
xdg-open frontend/index.html

# Windows:
start frontend/index.html

# OR just drag-and-drop frontend/index.html into Chrome/Firefox.

# The UI opens. Click "Scan My Face" → webcam opens →
# look at camera → authenticated → you're in!


# ══════════════════════════════════════════════════════════
# PART 6: INGEST YOUR DATA
# ══════════════════════════════════════════════════════════

# ── Step 12: Upload files via the UI ──────────────────────
# In the sidebar, click the upload area or use the API directly.
# Supported file types:
#   .pdf  — textbooks, notes, documents
#   .txt  — plain text notes
#   .md   — Markdown notes
#   .json — schedules in the format shown in data/sample_schedule.json

# Try with the sample files first:
# Upload: data/sample_notes.md
# Upload: data/sample_schedule.json

# ── Step 12b: Ingest via API directly (alternative) ───────
curl -X POST http://localhost:8000/ingest \
  -F "username=yourname" \
  -F "file=@data/sample_notes.md"

curl -X POST http://localhost:8000/ingest \
  -F "username=yourname" \
  -F "file=@data/sample_schedule.json"


# ══════════════════════════════════════════════════════════
# PART 7: CHAT WITH YOUR ASSISTANT
# ══════════════════════════════════════════════════════════

# ── Step 13: Ask questions ────────────────────────────────
# Type in the chat UI. Try these test queries:

# "What's on my schedule for Monday?"
# → Retrieves from sample_schedule.json

# "What LeetCode problems have I solved?"
# → Retrieves from sample_notes.md

# "Help me plan my study week"
# → Triggers decision_support planning mode

# "What are my goals and how am I tracking?"
# → Combines schedule + notes context

# "Summarize my ML notes on backpropagation"
# → Retrieves from notes, summarizes with Claude


# ══════════════════════════════════════════════════════════
# PART 8: VERIFY EVERYTHING IS WORKING
# ══════════════════════════════════════════════════════════

# Check API health:
curl http://localhost:8000/health

# Check ingested sources for a user:
curl http://localhost:8000/sources/yourname

# Check memory (past conversations):
curl http://localhost:8000/memory/yourname

# View full API docs (in browser):
# http://localhost:8000/docs


# ══════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ══════════════════════════════════════════════════════════

# ── Problem: face_recognition won't install ───────────────
# Solution: Install dlib wheel directly
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.0-cp310-cp310-linux_x86_64.whl
# (Find the right wheel for your Python version at the above repo)

# ── Problem: Webcam not opening ───────────────────────────
# Test webcam with:
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"
# If FAIL: try index 1 or 2 instead of 0 in face_auth scripts

# ── Problem: chromadb error on first run ──────────────────
# Solution:
pip install --upgrade chromadb
# If using Python 3.12: pip install chromadb==0.5.3

# ── Problem: "No module named 'fitz'" ─────────────────────
pip install PyMuPDF

# ── Problem: ANTHROPIC_API_KEY error ─────────────────────
echo $ANTHROPIC_API_KEY   # Should print your key
# If empty: export ANTHROPIC_API_KEY="sk-ant-..."

# ── Problem: Frontend shows "server offline" ─────────────
# Make sure api/main.py is running in a terminal
# Check port 8000 isn't blocked: lsof -i :8000

# ── Problem: Face not recognized (distance > 0.5) ─────────
# 1. Re-register in better lighting
# 2. Increase tolerance in config.py: FACE_TOLERANCE = 0.6
# 3. Make sure same camera angle as registration


# ══════════════════════════════════════════════════════════
# PART 9: ADDING YOUR OWN DATA
# ══════════════════════════════════════════════════════════

# ── Your own PDFs ─────────────────────────────────────────
# Any PDF: textbooks, lecture notes, research papers, documents
# Just upload via the UI or API.

# ── Your own notes ────────────────────────────────────────
# Any .txt or .md file. Export from Notion/Obsidian if needed.

# ── Your own schedule ─────────────────────────────────────
# Create a JSON file matching data/sample_schedule.json format.
# You can also export from Google Calendar and convert to JSON.

# ── Multiple PDFs at once ─────────────────────────────────
# Use the Python API directly to batch ingest:
python -c "
import sys; sys.path.insert(0, '.')
from ingest.pdf_loader import load_pdfs_from_directory
from embeddings.vector_store import VectorStore

vs = VectorStore('yourname')
docs = load_pdfs_from_directory('path/to/your/pdfs', 'yourname')
vs.add_documents(docs)
print('Done!')
"


# ══════════════════════════════════════════════════════════
# QUICK REFERENCE: COMPLETE STARTUP SEQUENCE
# ══════════════════════════════════════════════════════════

# Every time you want to use Life Assistant:

# Terminal 1:
cd LifeAssistant
source venv/bin/activate
export ANTHROPIC_API_KEY="sk-ant-..."   # (skip if in .zshrc/.bashrc)
python api/main.py

# Then open frontend/index.html in your browser.
# That's it.
