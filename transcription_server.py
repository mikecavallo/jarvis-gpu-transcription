#!/usr/bin/env python3
"""
GPU Transcription Server for Jarvis Assistant.

Runs on a machine with an NVIDIA GPU and serves fast Whisper transcription
over HTTP. The main Jarvis system sends audio and receives text back.

Features:
- GPU-accelerated Whisper (faster-whisper / CTranslate2) with CPU fallback
- REST API with FastAPI
- Multiple model sizes, configurable at startup
- Optional API-key authentication
- Health, JSON stats and Prometheus metrics endpoints

Setup:
1. pip install -r requirements.txt
2. python transcription_server.py
3. Point Jarvis at this server's IP and port
"""

import argparse
import logging
import os
import secrets
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, PlainTextResponse

logging.basicConfig(level=os.getenv("TRANSCRIPTION_LOG_LEVEL", "info").upper())
logger = logging.getLogger(__name__)

# Ordered smallest/fastest to largest/most accurate.
MODEL_DESCRIPTIONS = {
    "tiny": "Fastest, least accurate",
    "base": "Good balance",
    "small": "Better accuracy",
    "medium": "High accuracy (slower)",
    "large-v2": "Best accuracy (slowest)",
    "large-v3": "Best accuracy (slowest)",
}
DEFAULT_MODELS = "tiny,base,small,medium"


def configured_models() -> List[str]:
    """Model sizes to load, from TRANSCRIPTION_MODELS (comma-separated)."""
    raw = os.getenv("TRANSCRIPTION_MODELS", DEFAULT_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def model_rank(name: str) -> int:
    order = list(MODEL_DESCRIPTIONS)
    return order.index(name) if name in order else len(order)


def cuda_device_count() -> int:
    """Number of CUDA devices CTranslate2 can see (0 if none or no CUDA build)."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def gpu_name() -> str:
    """Best-effort GPU name via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = out.stdout.strip().splitlines()
        return lines[0] if out.returncode == 0 and lines else "None"
    except Exception:
        return "None"


def default_model_factory(size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device=device, compute_type=compute_type)


# Swappable so tests can inject a fake model without downloading weights.
model_factory: Callable = default_model_factory

models: Dict[str, object] = {}
server_stats = {
    "requests_processed": 0,
    "requests_failed": 0,
    "total_processing_time": 0.0,
    "total_audio_seconds": 0.0,
    "server_start_time": time.time(),
    "gpu_available": False,
    "gpu_name": "None",
    "device": "cpu",
}


def load_models(device: str) -> None:
    compute_type = "float16" if device == "cuda" else "int8"
    models.clear()
    for size in configured_models():
        logger.info("Loading %s model (%s) on %s...", size, MODEL_DESCRIPTIONS.get(size, "custom"), device)
        models[size] = model_factory(size, device, compute_type)
    server_stats["device"] = device
    server_stats["gpu_available"] = device == "cuda"


def initialize_models() -> None:
    """Load models on GPU if available, falling back to CPU on any CUDA/cuDNN failure."""
    server_stats["gpu_name"] = gpu_name()
    device = "cuda" if cuda_device_count() > 0 else "cpu"
    try:
        load_models(device)
    except Exception as e:
        if device != "cuda":
            raise
        logger.warning("GPU model load failed (%s); falling back to CPU", e)
        load_models("cpu")
    logger.info("Server ready with %d models on %s", len(models), server_stats["device"].upper())


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_models()
    yield
    models.clear()


app = FastAPI(
    title="Jarvis GPU Transcription Server",
    description="High-speed GPU transcription service for Jarvis Assistant",
    version="1.1.0",
    lifespan=lifespan,
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Enforce TRANSCRIPTION_API_KEY when it is set; open access otherwise."""
    expected = os.getenv("TRANSCRIPTION_API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


def average_processing_time() -> float:
    n = server_stats["requests_processed"]
    return server_stats["total_processing_time"] / n if n else 0.0


@app.get("/", response_class=HTMLResponse)
async def root():
    """Server dashboard with status and information."""
    uptime = time.time() - server_stats["server_start_time"]
    model_items = "".join(
        f"<li><strong>{name}</strong> - {MODEL_DESCRIPTIONS.get(name, 'custom')}</li>" for name in models
    )
    return f"""
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
            <h1 class="header">Jarvis GPU Transcription Server</h1>
            <div class="status">
                <h2>Server Status: Running</h2>
                <p><strong>Device:</strong> {server_stats['device'].upper()} ({server_stats['gpu_name']})</p>
                <p><strong>Models Loaded:</strong> {', '.join(models.keys())}</p>
                <p><strong>Uptime:</strong> {uptime/3600:.1f} hours</p>
            </div>
            <div class="stats">
                <div class="stat"><h3>Requests</h3><p>{server_stats['requests_processed']}</p></div>
                <div class="stat"><h3>Avg Time</h3><p>{average_processing_time():.2f}s</p></div>
                <div class="stat"><h3>GPU</h3><p>{'In use' if server_stats['gpu_available'] else 'Not in use'}</p></div>
            </div>
            <div class="models"><h2>Available Models</h2><ul>{model_items}</ul></div>
            <h2>API Endpoints</h2>
            <div class="endpoint"><strong>Health Check:</strong> <code>GET /health</code></div>
            <div class="endpoint"><strong>Transcription:</strong> <code>POST /transcribe</code></div>
            <div class="endpoint"><strong>Fast Transcription:</strong> <code>POST /transcribe/fast</code></div>
            <div class="endpoint"><strong>Accurate Transcription:</strong> <code>POST /transcribe/accurate</code></div>
            <div class="endpoint"><strong>Statistics:</strong> <code>GET /stats</code>, <code>GET /metrics</code></div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint (never requires an API key)."""
    return {
        "status": "healthy" if models else "loading",
        "models_available": len(models),
        "gpu_available": server_stats["gpu_available"],
        "requests_processed": server_stats["requests_processed"],
    }


@app.get("/stats", dependencies=[Depends(require_api_key)])
async def get_stats():
    """Detailed server statistics."""
    return {
        "requests_processed": server_stats["requests_processed"],
        "requests_failed": server_stats["requests_failed"],
        "uptime_seconds": time.time() - server_stats["server_start_time"],
        "average_processing_time": average_processing_time(),
        "total_audio_seconds": server_stats["total_audio_seconds"],
        "device": server_stats["device"],
        "gpu_info": {"available": server_stats["gpu_available"], "name": server_stats["gpu_name"]},
        "models": {name: "loaded" for name in models},
    }


@app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(require_api_key)])
async def metrics():
    """Prometheus text-format metrics."""
    metric_defs = [
        (
            "transcription_requests_total",
            "counter",
            "Successful transcription requests.",
            server_stats["requests_processed"],
        ),
        (
            "transcription_failures_total",
            "counter",
            "Failed transcription requests.",
            server_stats["requests_failed"],
        ),
        (
            "transcription_processing_seconds_total",
            "counter",
            "Total time spent transcribing.",
            round(server_stats["total_processing_time"], 6),
        ),
        (
            "transcription_audio_seconds_total",
            "counter",
            "Total audio duration transcribed.",
            round(server_stats["total_audio_seconds"], 6),
        ),
        ("transcription_models_loaded", "gauge", "Number of loaded models.", len(models)),
        ("transcription_gpu_in_use", "gauge", "1 if models run on GPU.", int(server_stats["gpu_available"])),
    ]
    lines = []
    for name, kind, help_text, value in metric_defs:
        lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", f"{name} {value}"]
    return "\n".join(lines) + "\n"


def run_whisper(model_name: str, path: str, language: Optional[str], task: str):
    """Run a transcription; on a GPU runtime error, reload on CPU and retry once."""
    kwargs = dict(
        language=language,
        task=task,
        beam_size=5 if model_rank(model_name) >= model_rank("small") else 3,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
    )
    try:
        segments, info = models[model_name].transcribe(path, **kwargs)
        # faster-whisper returns a lazy generator; decoding (and any CUDA error) happens here.
        return list(segments), info
    except Exception as e:
        if server_stats["device"] != "cuda" or not any(
            k in str(e).lower() for k in ("cuda", "cudnn", "cublas")
        ):
            raise
        logger.warning("GPU error during transcription (%s); reloading models on CPU", e)
        load_models("cpu")
        segments, info = models[model_name].transcribe(path, **kwargs)
        return list(segments), info


async def _transcribe(audio_file: UploadFile, model: str, language: Optional[str], task: str):
    start_time = time.time()

    if model not in models:
        raise HTTPException(
            status_code=400, detail=f"Model '{model}' not available. Available: {list(models)}"
        )
    if task not in ("transcribe", "translate"):
        raise HTTPException(status_code=400, detail="task must be 'transcribe' or 'translate'")
    content_type = audio_file.content_type or ""
    if content_type and not content_type.startswith(("audio/", "video/", "application/octet-stream")):
        raise HTTPException(status_code=400, detail="File must be audio or video format")

    content = await audio_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    suffix = os.path.splitext(audio_file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        logger.info("Transcribing with %s model, language: %s", model, language or "auto")
        segments, info = await run_in_threadpool(run_whisper, model, tmp_path, language, task)
    except Exception as e:
        server_stats["requests_failed"] += 1
        logger.error("Transcription error: %s", e)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    text = " ".join(s.text.strip() for s in segments).strip()
    processing_time = time.time() - start_time
    server_stats["requests_processed"] += 1
    server_stats["total_processing_time"] += processing_time
    server_stats["total_audio_seconds"] += float(getattr(info, "duration", 0.0) or 0.0)

    logger.info("Transcription completed in %.2fs: '%s'", processing_time, text[:50])
    return {
        "text": text,
        "language": getattr(info, "language", language),
        "language_probability": getattr(info, "language_probability", 0.0),
        "duration": getattr(info, "duration", None),
        "processing_time": processing_time,
        "model_used": model,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    }


def _language(value: Optional[str]) -> Optional[str]:
    """Empty or 'auto' means let Whisper detect the language."""
    return None if not value or value.lower() == "auto" else value


@app.post("/transcribe", dependencies=[Depends(require_api_key)])
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    model: str = Form("base"),
    language: str = Form("en"),
    task: str = Form("transcribe"),
):
    """
    Transcribe an audio file.

    Form fields: audio_file (WAV, MP3, etc.), model (tiny/base/small/medium),
    language (ISO code or "auto"), task (transcribe or translate).
    """
    return await _transcribe(audio_file, model, _language(language), task)


@app.post("/transcribe/fast", dependencies=[Depends(require_api_key)])
async def transcribe_fast(audio_file: UploadFile = File(...), language: str = Form("en")):
    """Lowest-latency transcription with the smallest loaded model. Built for voice commands."""
    model = min(models, key=model_rank, default="tiny")
    return await _transcribe(audio_file, model, _language(language), "transcribe")


@app.post("/transcribe/accurate", dependencies=[Depends(require_api_key)])
async def transcribe_accurate(audio_file: UploadFile = File(...), language: str = Form("en")):
    """Highest-accuracy transcription with the largest loaded model."""
    model = max(models, key=model_rank, default="medium")
    return await _transcribe(audio_file, model, _language(language), "transcribe")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Jarvis GPU Transcription Server")
    parser.add_argument("--host", default=os.getenv("TRANSCRIPTION_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TRANSCRIPTION_PORT", "8000")))
    parser.add_argument("--ssl-keyfile", default=None)
    parser.add_argument("--ssl-certfile", default=None)
    args = parser.parse_args()

    if not os.getenv("TRANSCRIPTION_API_KEY") and args.host not in ("127.0.0.1", "localhost"):
        logger.warning(
            "TRANSCRIPTION_API_KEY is not set: anyone who can reach %s:%d can use this server",
            args.host,
            args.port,
        )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=os.getenv("TRANSCRIPTION_LOG_LEVEL", "info").lower(),
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
