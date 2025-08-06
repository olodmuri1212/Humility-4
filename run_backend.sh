#!/bin/bash
echo "🚀 Starting Humility Interview Backend..."
echo "📋 Make sure you have installed dependencies:"
echo "   pip install fastapi uvicorn faster-whisper soundfile numpy"
echo ""

cd "$(dirname "$0")"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
