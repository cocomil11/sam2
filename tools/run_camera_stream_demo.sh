#!/bin/bash
# Script to run camera_stream_demo.py with automatic port cleanup

set -e  # Exit on error

# Activate conda environment
echo "Activating conda environment: sam2env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam2env

# Port to kill (viewer port)
VIEWER_PORT=8099

# Kill any process using the viewer port
echo "Checking for processes using port $VIEWER_PORT..."
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:$VIEWER_PORT 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "Killing process $PID using port $VIEWER_PORT..."
        kill -9 $PID 2>/dev/null || true
        sleep 1
        echo "Port $VIEWER_PORT is now free."
    else
        echo "Port $VIEWER_PORT is already free."
    fi
elif command -v fuser &> /dev/null; then
    if fuser $VIEWER_PORT/tcp &> /dev/null; then
        echo "Killing process using port $VIEWER_PORT..."
        fuser -k $VIEWER_PORT/tcp 2>/dev/null || true
        sleep 1
        echo "Port $VIEWER_PORT is now free."
    else
        echo "Port $VIEWER_PORT is already free."
    fi
else
    echo "Warning: Neither 'lsof' nor 'fuser' is available. Cannot check/kill port $VIEWER_PORT."
    echo "You may need to manually kill processes using this port if you get 'Address already in use' errors."
fi

# Change to script directory (so relative paths work)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Run the camera stream demo
echo ""
echo "Starting camera_stream_demo.py..."
echo "=================================="
python tools/camera_stream_demo.py \
    --config sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
    --checkpoint checkpoints/sam2.1_hiera_small.pt \
    --source http://172.21.128.1:8080/video \
    --bbox 200 150 400 350 \
    --bbox 0 250 300 300 \
    --num-objects 2 \
    --viewer-port 8099 \
    --no-gui

