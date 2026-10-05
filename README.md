# Jarvis GPU Transcription Server

[![CI](https://github.com/mikecavallo/jarvis-gpu-transcription/actions/workflows/ci.yml/badge.svg)](https://github.com/mikecavallo/jarvis-gpu-transcription/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A self-hosted speech-to-text server for the [Jarvis Voice Assistant](https://github.com/mikecavallo/jarvis-voice-assistant).
It runs on a machine with an NVIDIA GPU and gives the rest of the voice pipeline a
low-latency HTTP transcription API.

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) and FastAPI.

## How it fits

Jarvis listens for a wake word, records a command, and needs text back fast enough that
the reply feels conversational. This server moves transcription onto a separate GPU
machine (a gaming PC works) on the local network.

```
 mic ──► Jarvis (wake word, recording) ──HTTP──► this server (GPU Whisper) ──► text ──► Jarvis (intent, tools, TTS)
```

## Features

- **GPU first, CPU fallback.** Loads models on CUDA in float16. If CUDA or cuDNN fails at
  load time *or mid-request*, it reloads on CPU (int8) and retries, so a driver problem
  degrades speed instead of availability.
- **Several model sizes loaded at once.** `/transcribe/fast` uses the smallest loaded model
  for voice commands; `/transcribe/accurate` uses the largest for longer audio.
- **Non-blocking.** Inference runs in a worker thread so health checks stay responsive
  during long transcriptions.
- **Optional API-key auth** via the `X-API-Key` header.
- **Observability.** HTML dashboard at `/`, JSON stats at `/stats`, Prometheus metrics at `/metrics`.
- **Tested.** Unit tests with a fake model cover routing, validation, auth and both
  GPU-fallback paths; an integration test runs a real Whisper model in CI.

## Quick start

```bash
git clone https://github.com/mikecavallo/jarvis-gpu-transcription.git
cd jarvis-gpu-transcription
./install.sh          # creates venv, installs deps, checks for a GPU
./start_server.sh     # serves on 0.0.0.0:8000
```

Or manually:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python transcription_server.py --port 8000
```

Model weights download from Hugging Face on first start.

### Docker

```bash
docker build -t transcription-server .
docker run --gpus all -p 8000:8000 transcription-server          # GPU
docker run -p 8000:8000 transcription-server                     # CPU only
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness and loaded-model count. Never requires a key. |
| `POST` | `/transcribe` | Form fields: `audio_file`, `model`, `language` (ISO code or `auto`), `task` (`transcribe` or `translate`) |
| `POST` | `/transcribe/fast` | Smallest loaded model. Form fields: `audio_file`, `language` |
| `POST` | `/transcribe/accurate` | Largest loaded model. Form fields: `audio_file`, `language` |
| `GET`  | `/stats` | JSON request counts, timings, device info |
| `GET`  | `/metrics` | Prometheus text format |
| `GET`  | `/` | HTML status dashboard |

```bash
curl -X POST http://localhost:8000/transcribe/fast \
  -H "X-API-Key: $TRANSCRIPTION_API_KEY" \
  -F "audio_file=@command.wav" -F "language=en"
```

Example response:

```json
{
  "text": "turn off the office lights",
  "language": "en",
  "language_probability": 0.99,
  "duration": 1.8,
  "processing_time": 0.21,
  "model_used": "tiny",
  "segments": [{"start": 0.0, "end": 1.8, "text": " turn off the office lights"}]
}
```

From Python:

```python
import requests

with open("command.wav", "rb") as f:
    r = requests.post(
        "http://gpu-box.local:8000/transcribe/fast",
        files={"audio_file": f},
        data={"language": "en"},
        headers={"X-API-Key": "your-key"},
    )
print(r.json()["text"])
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRANSCRIPTION_MODELS` | `tiny,base,small,medium` | Comma-separated model sizes to load. Trim this on GPUs with little VRAM. |
| `TRANSCRIPTION_API_KEY` | unset | When set, every endpoint except `/health` requires a matching `X-API-Key` header. |
| `TRANSCRIPTION_HOST` | `0.0.0.0` | Bind address. |
| `TRANSCRIPTION_PORT` | `8000` | Port. |
| `TRANSCRIPTION_LOG_LEVEL` | `info` | Log level. |
| `CUDA_VISIBLE_DEVICES` | all | Set to `""` to force CPU. |

HTTPS: `python transcription_server.py --ssl-keyfile key.pem --ssl-certfile cert.pem`.

The server binds to all interfaces by default and logs a warning when no API key is set.
Set one before exposing it beyond your LAN.

## Benchmarks

Measured with `scripts/benchmark.py` on an 11-second speech clip (median of 3 runs after a
warm-up). RTF is latency divided by audio length; lower is better.

**4-core Intel Xeon @ 2.1 GHz, CPU only, int8:**

| Model | Median latency | RTF |
|-------|----------------|-----|
| tiny  | 0.84 s | 0.08 |
| base  | 1.28 s | 0.12 |
| small | 3.10 s | 0.28 |

Reproduce on your own hardware (GPU is picked automatically when available):

```bash
python scripts/benchmark.py your-audio.wav --models tiny,base,small,medium --runs 5
```

## Development

```bash
pip install -r requirements-dev.txt
pytest                    # fast unit tests, no model download
pytest -m integration     # real tiny model, ~75 MB download
ruff check . && ruff format --check .
```

## License

MIT. See [LICENSE](LICENSE).
