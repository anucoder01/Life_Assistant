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

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "❌ ERROR: ANTHROPIC_API_KEY is not set!"
  echo ""
  echo "   Get your key from: https://console.anthropic.com"
  echo "   Then run:"
  echo "   export ANTHROPIC_API_KEY='sk-ant-...'"
  echo "   ./run.sh"
  echo ""
  exit 1
fi

echo "✅ API key found"
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
