#!/bin/bash
# Render startup script
# Database initialization happens in background thread

echo "🚀 Starting FastAPI server..."
uvicorn qa_agent:app --host 0.0.0.0 --port $PORT
