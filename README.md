# 🎮 Jarvis GPU Transcription Server

High-speed GPU-accelerated transcription server for the Jarvis Voice Assistant. Turn your gaming rig into a lightning-fast speech recognition powerhouse!

## ⚡ Performance

Transform your voice command experience:
- **RTX 3090**: 0.1-0.5 seconds (vs 2-5 seconds CPU)  
- **RTX 4090**: 0.05-0.3 seconds
- **Multiple models**: Speed/accuracy tradeoffs
- **Real-time processing**: Perfect for voice assistants

## 🚀 Quick Start

### Prerequisites
- **Ubuntu/Linux** with NVIDIA GPU
- **Python 3.8+**
- **CUDA-compatible GPU** (GTX 1060+ recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/mikecavallo/jarvis-gpu-transcription.git
cd jarvis-gpu-transcription

# Run the installation script
chmod +x install.sh
./install.sh

# Start the server
./start_server.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server
python transcription_server.py
```

## 🎯 Usage

### Server Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Fast Transcription (Real-time):**
```bash
curl -X POST "http://localhost:8000/transcribe/fast" \
  -F "audio_file=@audio.wav" \
  -F "language=en"
```

**Accurate Transcription:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio_file=@audio.wav" \
  -F "model=base" \
  -F "language=en"
```

**Server Statistics:**
```bash
curl http://localhost:8000/stats
```

### Available Models

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `tiny` | Fastest | Good | Real-time voice commands |
| `base` | Fast | Very Good | General transcription |
| `small` | Medium | Excellent | High-quality transcription |
| `medium` | Slower | Best | Maximum accuracy |

## 🔧 Configuration

### Environment Variables

```bash
export TRANSCRIPTION_HOST=0.0.0.0
export TRANSCRIPTION_PORT=8000
export TRANSCRIPTION_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=0
```

### GPU Memory Optimization

For systems with limited GPU memory:

```python
# Edit transcription_server.py
model_configs = {
    "tiny": {"size": "tiny"},
    "base": {"size": "base"},
    # Remove larger models to save memory
}
```

## 🐳 Docker Support

### GPU Mode (Recommended)
```bash
# Build Docker image
docker build -t transcription-server .

# Run with GPU support (if cuDNN compatible)
docker run --gpus all -p 8000:8000 transcription-server
```

### CPU Mode (Stable Fallback)
```bash
# Build Docker image
docker build -t transcription-server .

# Run in CPU-only mode (cuDNN compatibility issues)
docker run --rm -p 8000:8000 -e CUDA_VISIBLE_DEVICES="" transcription-server
```

**Note:** Due to cuDNN 9 compatibility requirements in faster-whisper, the Docker container defaults to CPU mode for stability. This provides reliable transcription across all systems, though slower than GPU mode.

## 🔗 Integration

### Jarvis Voice Assistant

This server is designed to work with the [Jarvis Voice Assistant](https://github.com/mikecavallo/jarvis-voice-assistant). 

1. Update your Jarvis `config.json`:
```json
{
  "transcription_server": {
    "enabled": true,
    "host": "192.168.1.100",
    "port": 8000,
    "timeout": 10
  }
}
```

2. Jarvis will automatically use GPU transcription when available!

### Custom Integration

```python
import requests
import io

# Send audio file for transcription
with open("audio.wav", "rb") as f:
    files = {"audio_file": f}
    response = requests.post(
        "http://localhost:8000/transcribe/fast",
        files=files,
        data={"language": "en"}
    )
    
result = response.json()
print(f"Transcription: {result['text']}")
print(f"Processing time: {result['processing_time']:.2f}s")
```

## 📊 Monitoring

### Web Dashboard
Visit `http://localhost:8000` for server status and statistics.

### Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

## 🛠️ Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### cuDNN Compatibility Issues
If you encounter `libcudnn_ops.so` errors, the server automatically falls back to CPU mode:

```bash
# Force CPU mode explicitly
export CUDA_VISIBLE_DEVICES=""
python transcription_server.py

# Or use Docker CPU mode
docker run --rm -p 8000:8000 -e CUDA_VISIBLE_DEVICES="" transcription-server
```

The server includes automatic CPU fallback handling for maximum compatibility.

### Memory Issues
```bash
# Monitor GPU memory
nvidia-smi

# Reduce model loading (edit transcription_server.py)
# Use only "tiny" and "base" models
```

### Network Issues
```bash
# Check firewall
sudo ufw allow 8000

# Test from another machine
curl http://GPU_SERVER_IP:8000/health
```

### Performance Optimization

**For Maximum Speed:**
- Use `tiny` model for real-time applications
- Set `beam_size=1` for fastest inference
- Use `temperature=0.0` for deterministic results

**For Maximum Accuracy:**
- Use `medium` or `small` models
- Increase `beam_size` to 5
- Use higher quality audio (16kHz+)

## 🔒 Security

### API Key Authentication (Optional)
```bash
export TRANSCRIPTION_API_KEY=your-secret-key
```

### HTTPS Support
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Start with HTTPS
python transcription_server.py --ssl-keyfile key.pem --ssl-certfile cert.pem
```

## 📈 Benchmarks

### RTX 3090 Performance
- **Tiny Model**: ~0.1s for 3-second audio
- **Base Model**: ~0.3s for 3-second audio  
- **Small Model**: ~0.5s for 3-second audio
- **Memory Usage**: 2-4GB VRAM

### RTX 4090 Performance
- **Tiny Model**: ~0.05s for 3-second audio
- **Base Model**: ~0.15s for 3-second audio
- **Small Model**: ~0.25s for 3-second audio
- **Memory Usage**: 2-4GB VRAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for the amazing speech recognition models
- **Faster-Whisper** for GPU optimization
- **FastAPI** for the excellent web framework
- **Jarvis Voice Assistant** community

---

**Transform your voice commands with GPU power!** 🎮⚡

[![Deploy to GPU Server](https://img.shields.io/badge/Deploy-GPU%20Server-blue)](https://github.com/mikecavallo/jarvis-gpu-transcription/blob/main/README.md#quick-start)
[![Docker Support](https://img.shields.io/badge/Docker-Supported-green)](https://github.com/mikecavallo/jarvis-gpu-transcription/blob/main/README.md#docker-support)
[![CUDA](https://img.shields.io/badge/CUDA-Required-red)](https://developer.nvidia.com/cuda-downloads)