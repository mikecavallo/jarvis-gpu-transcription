#!/bin/bash
# Jarvis GPU Transcription Server Installation Script
# For Ubuntu systems with NVIDIA GPUs

set -e  # Exit on any error

echo "🚀 Installing Jarvis GPU Transcription Server..."
echo "================================================"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ This script is designed for Linux systems"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python $python_version found"
else
    echo "❌ Python 3.8+ required. Found: $python_version"
    echo "Install with: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
else
    echo "⚠️  NVIDIA GPU not detected. Server will run on CPU (slower)"
    echo "Install NVIDIA drivers if you have a GPU"
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test GPU availability
echo "🧪 Testing GPU availability..."
python3 -c "
import ctranslate2
n = ctranslate2.get_cuda_device_count()
print(f'CUDA devices visible to CTranslate2: {n}')
print('GPU will be used' if n else 'GPU not available - will use CPU')
"

# Create startup script
echo "📝 Creating startup script..."
cat > start_server.sh << 'EOF'
#!/bin/bash
# Jarvis GPU Transcription Server Startup Script

echo "🚀 Starting Jarvis GPU Transcription Server..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start server (host/port/API key via TRANSCRIPTION_* env vars)
python transcription_server.py
EOF

chmod +x start_server.sh

# Create systemd service file (optional)
echo "📝 Creating systemd service template..."
cat > jarvis-transcription.service << EOF
[Unit]
Description=Jarvis GPU Transcription Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=$(pwd)/venv/bin/python transcription_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

echo "================================================"
echo "🎉 Installation Complete!"
echo "================================================"
echo ""
echo "🚀 To start the server:"
echo "   ./start_server.sh"
echo ""
echo "🔧 To install as system service:"
echo "   sudo cp jarvis-transcription.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable jarvis-transcription"
echo "   sudo systemctl start jarvis-transcription"
echo ""
echo "🌐 Server will be available at:"
echo "   http://localhost:8000"
echo "   http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "🧪 Test the server:"
echo "   curl http://localhost:8000/health"
echo ""
echo "📊 View dashboard:"
echo "   Open http://localhost:8000 in your browser"