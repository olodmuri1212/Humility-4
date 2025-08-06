#!/bin/bash
echo "🎤 Starting Humility Interview Frontend..."
echo "📋 Make sure backend is running on http://127.0.0.1:8000"
echo ""

cd "$(dirname "$0")"
python gradio_app_enhanced.py
