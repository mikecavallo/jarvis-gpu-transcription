#!/usr/bin/env python3
"""
GPU Transcription Server for Jarvis Assistant.

This server runs on your gaming rig with NVIDIA GPU and provides
fast transcription services via HTTP API. The main Jarvis system
sends audio files and receives transcriptions.

Features:
- GPU-accelerated Whisper transcription
- REST API with FastAPI
- Multiple model size support
- Batch processing capability
- Health checks and monitoring

Setup:
1. pip install -r requirements.txt
2. python transcription_server.py
3. Update Jarvis config with server IP
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import tempfile
import os
import time
import logging
from typing import Optional, Dict, Any
from faster_whisper import WhisperModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jarvis GPU Transcription Server",
    description="High-speed GPU transcription service for Jarvis Assistant",
    version="1.0.0"
)

# Global model storage
models: Dict[str, WhisperModel] = {}
server_stats = {
    "requests_processed": 0,
    "total_processing_time": 0.0,
    "server_start_time": time.time(),
    "gpu_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
}

def initialize_models():
    """Initialize Whisper models with GPU fallback to CPU."""
    global server_stats
    
    try:
        # Try GPU first, fallback to CPU if cuDNN issues
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Attempting to initialize models on {device}")
        logger.info(f"GPU: {server_stats['gpu_name']}")
        
        # Load multiple model sizes for different speed/accuracy tradeoffs
        model_configs = {
            "tiny": {"size": "tiny", "description": "Fastest, least accurate"},
            "base": {"size": "base", "description": "Good balance"},
            "small": {"size": "small", "description": "Better accuracy"},
            "medium": {"size": "medium", "description": "High accuracy (slower)"}
        }
        
        # Try loading models with GPU first
        gpu_failed = False
        for name, config in model_configs.items():
            try:
                logger.info(f"Loading {name} model ({config['description']}) on {device}...")
                models[name] = WhisperModel(
                    config["size"], 
                    device=device, 
                    compute_type=compute_type
                )
                logger.info(f"✅ {name} model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"❌ Failed to load {name} model on {device}: {e}")
                if device == "cuda" and "cudnn" in str(e).lower():
                    logger.warning("cuDNN compatibility issue detected, will try CPU fallback")
                    gpu_failed = True
                    break
        
        # If GPU failed due to cuDNN, retry with CPU
        if gpu_failed and device == "cuda":
            logger.info("🔄 Falling back to CPU due to GPU/cuDNN issues...")
            models.clear()  # Clear any partially loaded models
            device = "cpu"
            compute_type = "int8"
            server_stats["gpu_available"] = False
            
            for name, config in model_configs.items():
                try:
                    logger.info(f"Loading {name} model ({config['description']}) on CPU...")
                    models[name] = WhisperModel(
                        config["size"], 
                        device=device, 
                        compute_type=compute_type
                    )
                    logger.info(f"✅ {name} model loaded successfully on CPU")
                except Exception as e:
                    logger.error(f"❌ Failed to load {name} model on CPU: {e}")
        
        if not models:
            raise Exception("No models could be loaded on either GPU or CPU")
            
        final_device = "CPU" if device == "cpu" else "GPU"
        logger.info(f"🚀 Server ready with {len(models)} models on {final_device}")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize models when server starts."""
    initialize_models()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Server dashboard with status and information."""
    uptime = time.time() - server_stats["server_start_time"]
    avg_processing_time = (
        server_stats["total_processing_time"] / server_stats["requests_processed"]
        if server_stats["requests_processed"] > 0 else 0
    )
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jarvis GPU Transcription Server</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
            .header {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            .status {{ background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
            .stat {{ background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
            .models {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .endpoint {{ background: #d1ecf1; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="header">🎮 Jarvis GPU Transcription Server</h1>
            
            <div class="status">
                <h2>🟢 Server Status: Running</h2>
                <p><strong>GPU:</strong> {server_stats['gpu_name']}</p>
                <p><strong>Models Loaded:</strong> {', '.join(models.keys())}</p>
                <p><strong>Uptime:</strong> {uptime/3600:.1f} hours</p>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <h3>📊 Requests</h3>
                    <p>{server_stats['requests_processed']}</p>
                </div>
                <div class="stat">
                    <h3>⚡ Avg Speed</h3>
                    <p>{avg_processing_time:.2f}s</p>
                </div>
                <div class="stat">
                    <h3>🎮 GPU</h3>
                    <p>{'✅ Available' if server_stats['gpu_available'] else '❌ Not Available'}</p>
                </div>
            </div>
            
            <div class="models">
                <h2>🤖 Available Models</h2>
                <ul>
                    <li><strong>tiny</strong> - Fastest (~0.1s), good for real-time</li>
                    <li><strong>base</strong> - Balanced (~0.3s), recommended</li>
                    <li><strong>small</strong> - High accuracy (~0.5s)</li>
                    <li><strong>medium</strong> - Best accuracy (~0.8s)</li>
                </ul>
            </div>
            
            <h2>🔌 API Endpoints</h2>
            <div class="endpoint">
                <strong>Health Check:</strong> <code>GET /health</code>
            </div>
            <div class="endpoint">
                <strong>Fast Transcription:</strong> <code>POST /transcribe/fast</code>
            </div>
            <div class="endpoint">
                <strong>Custom Transcription:</strong> <code>POST /transcribe</code>
            </div>
            <div class="endpoint">
                <strong>Statistics:</strong> <code>GET /stats</code>
            </div>
            
            <p><em>Perfect for Jarvis Voice Assistant and other real-time applications!</em></p>
        </div>
    </body>
    </html>
    """
    return html

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_available": len(models),
        "gpu_available": server_stats["gpu_available"],
        "requests_processed": server_stats["requests_processed"]
    }

@app.get("/stats")
async def get_stats():
    """Get detailed server statistics."""
    uptime = time.time() - server_stats["server_start_time"]
    avg_processing_time = (
        server_stats["total_processing_time"] / server_stats["requests_processed"]
        if server_stats["requests_processed"] > 0 else 0
    )
    
    return {
        "requests_processed": server_stats["requests_processed"],
        "uptime_seconds": uptime,
        "average_processing_time": avg_processing_time,
        "gpu_info": {
            "available": server_stats["gpu_available"],
            "name": server_stats["gpu_name"]
        },
        "models": {name: "loaded" for name in models.keys()}
    }

@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model: str = "base",
    language: str = "en",
    task: str = "transcribe"
):
    """
    Transcribe audio file using GPU-accelerated Whisper.
    
    Args:
        audio_file: Audio file (WAV, MP3, etc.)
        model: Model size (tiny, base, small, medium)
        language: Language code (en, es, fr, etc.)
        task: transcribe or translate
    
    Returns:
        JSON with transcription result
    """
    start_time = time.time()
    
    try:
        # Validate model
        if model not in models:
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model}' not available. Available: {list(models.keys())}"
            )
        
        # Validate file
        if not audio_file.content_type.startswith(('audio/', 'video/')):
            raise HTTPException(
                status_code=400,
                detail="File must be audio or video format"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Transcribe with selected model
            whisper_model = models[model]
            
            logger.info(f"Transcribing with {model} model, language: {language}")
            
            try:
                segments, info = whisper_model.transcribe(
                    tmp_file_path,
                    language=language,
                    task=task,
                    beam_size=5 if model in ["small", "medium"] else 3,  # Better quality for larger models
                    temperature=0.0,  # Deterministic
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False
                )
            except Exception as transcribe_error:
                # Check if it's a cuDNN error during transcription
                if "cudnn" in str(transcribe_error).lower():
                    logger.warning(f"cuDNN error during transcription, reinitializing models on CPU: {transcribe_error}")
                    # Reinitialize all models on CPU
                    models.clear()
                    global server_stats
                    server_stats["gpu_available"] = False
                    
                    model_configs = {
                        "tiny": {"size": "tiny"},
                        "base": {"size": "base"}, 
                        "small": {"size": "small"},
                        "medium": {"size": "medium"}
                    }
                    
                    for name, config in model_configs.items():
                        models[name] = WhisperModel(config["size"], device="cpu", compute_type="int8")
                    
                    # Retry transcription with CPU model
                    whisper_model = models[model]
                    segments, info = whisper_model.transcribe(
                        tmp_file_path,
                        language=language,
                        task=task,
                        beam_size=5 if model in ["small", "medium"] else 3,
                        temperature=0.0,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=False
                    )
                    logger.info("✅ Successfully transcribed using CPU fallback")
                else:
                    raise transcribe_error
            
            # Extract text and metadata
            text = " ".join([segment.text for segment in segments]).strip()
            
            processing_time = time.time() - start_time
            
            # Update stats
            server_stats["requests_processed"] += 1
            server_stats["total_processing_time"] += processing_time
            
            result = {
                "text": text,
                "language": info.language if hasattr(info, 'language') else language,
                "language_probability": getattr(info, 'language_probability', 0.0),
                "processing_time": processing_time,
                "model_used": model,
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    }
                    for segment in segments
                ]
            }
            
            logger.info(f"Transcription completed in {processing_time:.2f}s: '{text[:50]}...'")
            
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe/fast")
async def transcribe_fast(
    audio_file: UploadFile = File(...),
    language: str = "en"
):
    """
    Fast transcription using the smallest model for real-time applications.
    
    Optimized for speed over accuracy - perfect for voice commands.
    """
    return await transcribe_audio(
        audio_file=audio_file,
        model="tiny",  # Fastest model
        language=language,
        task="transcribe"
    )

@app.post("/transcribe/accurate")
async def transcribe_accurate(
    audio_file: UploadFile = File(...),
    language: str = "en"
):
    """
    High-accuracy transcription using larger model.
    
    Optimized for accuracy over speed - better for longer content.
    """
    return await transcribe_audio(
        audio_file=audio_file,
        model="medium",  # Most accurate model we load
        language=language,
        task="transcribe"
    )

