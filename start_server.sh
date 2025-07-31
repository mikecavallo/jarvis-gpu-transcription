#!/bin/bash
# Jarvis GPU Transcription Server Startup Script

echo "🚀 Starting Jarvis GPU Transcription Server..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start server
uvicorn transcription_server:app --host 0.0.0.0 --port 8000
