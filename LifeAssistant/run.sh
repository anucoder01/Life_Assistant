#!/bin/bash
# run.sh — Start the Life Assistant
# ============================================================
# This script:
# 1. Checks your API key is set
# 2. Starts the FastAPI backend server
# 3. Opens the frontend in your browser
#
# Usage: ./run.sh
# ============================================================

set -e  # Exit on error

echo ""
echo "🧠 Life Assistant — Starting Up"
echo "================================"

# Check that at least one configured provider has a key
if [ -z "$GROQ_API_KEY" ] && [ -z "$GEMINI_API_KEY" ]; then
  echo ""
  echo "❌ ERROR: No LLM provider key is set!"
  echo ""
  echo "   Set GROQ_API_KEY and/or GEMINI_API_KEY before starting."
  echo "   For example: export GROQ_API_KEY='...'"
  echo "   ./run.sh"
  echo ""
  exit 1
fi

echo "✅ LLM provider key found"
echo ""

# Start API server
echo "🚀 Starting FastAPI server on http://localhost:8000"
echo "   (Press Ctrl+C to stop)"
echo ""

# Open browser after a short delay
(sleep 2 && open frontend/index.html 2>/dev/null || \
           xdg-open frontend/index.html 2>/dev/null || \
           echo "Open frontend/index.html in your browser") &

# Start server
python api/main.py
