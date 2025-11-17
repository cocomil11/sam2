# Camera Streaming Setup Guide

This guide explains how to test real-time SAM2 inference with a camera stream, simulating a mobile device streaming to a server.

## Architecture

- **Windows Host** = Mobile device (streams webcam over HTTP)
- **WSL** = Server (receives stream and runs SAM2 inference)

## Quick Start

### Step 1: Start the Streaming Server (Windows)

You have three options:

#### Option A: Run directly from Windows PowerShell
```powershell
# Access WSL filesystem from Windows
python \\wsl$\Ubuntu\home\kenta\sam2\tools\webcam_stream_server.py --port 8080 --camera 0
```

#### Option B: Use the batch script (Windows)
Double-click `webcam_stream_server.bat` or run:
```cmd
tools\webcam_stream_server.bat
```

#### Option C: Copy script to Windows (if above don't work)
```powershell
# Copy the script to Windows
copy \\wsl$\Ubuntu\home\kenta\sam2\tools\webcam_stream_server.py C:\temp\
python C:\temp\webcam_stream_server.py --port 8080 --camera 0
```

### Step 2: Run Inference Client (WSL)

In your WSL terminal:
```bash
python tools/camera_stream_demo.py \
    --config sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
    --checkpoint checkpoints/sam2.1_hiera_small.pt \
    --source http://localhost:8080/video
```

## Troubleshooting

### Server can't open camera
- Try different camera IDs: `--camera 1`, `--camera 2`, etc.
- Make sure no other application is using the webcam
- Check Windows Camera app to verify webcam works

### Can't access WSL files from Windows
- Make sure WSL is running: `wsl --list`
- Try using `\\wsl.localhost\` instead of `\\wsl$\`
- Check your WSL distro name: `wsl --list -v`

### Connection refused in WSL
- Make sure the server is running on Windows
- Try `http://127.0.0.1:8080/video` instead of `localhost`
- Check Windows Firewall isn't blocking port 8080

### Alternative: Use a video file
If webcam access is problematic, test with a video file:
```bash
python tools/camera_stream_demo.py \
    --config sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
    --checkpoint checkpoints/sam2.1_hiera_small.pt \
    --source demo/data/gallery/05_default_juggle.mp4
```

## Next Steps

Once this works, you can:
1. Replace the Windows webcam server with a real mobile app (IP Webcam, DroidCam, etc.)
2. Change `localhost` to your phone's IP address
3. Deploy the inference server to a cloud instance
4. Add authentication and encryption for production use

