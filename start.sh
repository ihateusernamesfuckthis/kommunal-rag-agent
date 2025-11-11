#!/bin/bash
# Render startup script

echo "🚀 Starting deployment..."

# Initialize database if needed
echo "🔄 Checking ChromaDB..."
python init_db.py

# Start the FastAPI app
echo "▶️  Starting FastAPI server..."
uvicorn qa_agent:app --host 0.0.0.0 --port $PORT
