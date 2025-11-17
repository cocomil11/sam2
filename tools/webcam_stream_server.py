#!/usr/bin/env python3
"""
Simple HTTP MJPEG streaming server to simulate a mobile device streaming to the server.
This captures from a local webcam and serves it over HTTP, mimicking phone-to-server streaming.

Usage:
    # On Windows (or WSL with camera access):
    python tools/webcam_stream_server.py --port 8080 --camera 0
    
    # Then in WSL, connect to it:
    python tools/camera_stream_demo.py \
        --config sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --source http://localhost:8080/video
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import cv2
import numpy as np


class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves MJPEG video stream."""

    def __init__(self, cap: cv2.VideoCapture, *args, **kwargs):
        self.cap = cap
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/video" or self.path == "/":
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
            except (ConnectionResetError, BrokenPipeError, OSError, ConnectionAbortedError):
                # Client disconnected before we could send headers
                return
            
            try:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        # Camera read failed, wait a bit and try again
                        time.sleep(0.1)
                        continue
                    
                    # Encode frame as JPEG
                    try:
                        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if buffer is None:
                            continue
                        frame_bytes = buffer.tobytes()
                    except Exception:
                        # Encoding failed, skip this frame
                        continue
                    
                    # Send frame
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()  # Ensure frame is sent
                    except (ConnectionResetError, BrokenPipeError, OSError, ConnectionAbortedError):
                        # Client disconnected, which is normal
                        break
                    except Exception:
                        # Any other error, assume client disconnected
                        break
            except (ConnectionResetError, BrokenPipeError, OSError, ConnectionAbortedError):
                # Client disconnected, which is normal - suppress the error
                pass
            except Exception:
                # Any other error, log it but don't crash
                pass
        else:
            try:
                self.send_response(404)
                self.end_headers()
            except Exception:
                # Client disconnected, ignore
                pass

    def do_HEAD(self):
        if self.path == "/video" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging for normal requests
        # Only log errors, not normal disconnections
        message = format % args if args else format
        # Suppress common connection errors that are normal
        if any(term in message.lower() for term in ["error", "exception", "traceback"]):
            # But skip connection reset/aborted errors as they're normal
            if not any(term in message.lower() for term in ["connection reset", "connection aborted", "broken pipe"]):
                super().log_message(format, *args)
        # Otherwise suppress (normal connection/disconnection)


class StreamingServer:
    """HTTP server that streams video from a camera."""

    def __init__(self, camera_id: int, port: int = 8080):
        self.camera_id = camera_id
        self.port = port
        self.cap: Optional[cv2.VideoCapture] = None
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start the streaming server."""
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Set reasonable resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera opened: {width}x{height} @ {fps:.1f} FPS")

        def handler_factory(cap):
            def create_handler(*args, **kwargs):
                return StreamingHandler(cap, *args, **kwargs)

            return create_handler

        self.server = HTTPServer(("0.0.0.0", self.port), handler_factory(self.cap))
        # Allow socket reuse to avoid "Address already in use" errors
        self.server.allow_reuse_address = True
        # Get the actual IP addresses the server is listening on
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Starting HTTP MJPEG streaming server on:")
        print(f"  http://localhost:{self.port}/video")
        print(f"  http://127.0.0.1:{self.port}/video")
        print(f"  http://{local_ip}:{self.port}/video")
        print(f"  (Listening on 0.0.0.0:{self.port} - all interfaces)")
        print("Press Ctrl+C to stop")

        def run_server():
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                pass
            except Exception as e:
                # Log unexpected errors but don't crash
                print(f"Server error: {e}")
                print("Server will continue accepting new connections...")

        # Don't use daemon thread - we want the server to keep running
        self.thread = threading.Thread(target=run_server, daemon=False)
        self.thread.start()

    def stop(self):
        """Stop the streaming server."""
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()  # Explicitly close the socket
            except Exception:
                pass
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        print("Streaming server stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTTP MJPEG streaming server for webcam (simulates mobile device)."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP server port (default: 8080).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        with StreamingServer(camera_id=args.camera, port=args.port):
            # Keep the server running
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nNote: If you're in WSL, you may need to:")
        print("  1. Run this script on Windows (native Python)")
        print("  2. Or use a video file instead: --source path/to/video.mp4")


if __name__ == "__main__":
    main()

