#!/bin/bash
# Render startup script

echo "🚀 Starting deployment..."

# Download pre-built ChromaDB if needed
echo "📦 Checking for ChromaDB..."
python download_chromadb.py

# Start the FastAPI app
echo "▶️  Starting FastAPI server..."
uvicorn qa_agent:app --host 0.0.0.0 --port $PORT
