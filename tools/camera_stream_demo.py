#!/usr/bin/env python3
"""
Real-time camera/video stream segmentation with the SAM 2 camera predictor.

Example:
    python tools/camera_stream_demo.py \
        --config sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --source 0
"""

import argparse
import contextlib
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_camera_predictor

# Fixed BGR color palette for up to eight objects.
PALETTE = (
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (255, 105, 180),
    (255, 255, 0),
    (138, 43, 226),
    (0, 255, 255),
    (255, 140, 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM 2 camera predictor directly on a live stream."
    )
    parser.add_argument(
        "--config",
        default="sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
        help="Path to SAM 2 config YAML (default: sam2/configs/sam2.1/sam2.1_hiera_s.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam2.1_hiera_small.pt",
        help="Path to model checkpoint (default: checkpoints/sam2.1_hiera_small.pt).",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source. Use an integer webcam ID (e.g. 0) or a path to an MP4.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for inference (default: cuda).",
    )
    parser.add_argument(
        "--obj-id",
        type=int,
        default=1,
        help="Internal object id for prompts (default: 1).",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        action="append",
        help=(
            "Bounding box prompt (repeat flag for multiple objects). "
            "Format per box: x0 y0 x1 y1 (top-left and bottom-right corners)."
        ),
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=1,
        help="Total number of objects to track (defaults to 1). "
        "If fewer --bbox flags are provided, the remaining boxes are selected interactively.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary masks (default: 0.5).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Overlay opacity in [0, 1] (default: 0.6).",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional path to save the visualized stream (e.g. outputs/camera_demo.mp4).",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable torch.autocast even when running on CUDA.",
    )
    parser.add_argument(
        "--vos-optimized",
        action="store_true",
        help="Switch to the VOS-optimized camera predictor variant.",
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=None,
        help="Port for HTTP visualization server (view in browser). If not set, no viewer server is started.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable OpenCV GUI windows (useful for headless environments).",
    )
    return parser.parse_args()


def get_windows_host_ip() -> Optional[str]:
    """Get the Windows host IP address from WSL2."""
    try:
        # Method 1: Read from /etc/resolv.conf (most reliable in WSL2)
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    ip = line.split()[1]
                    # WSL2 uses the nameserver as the Windows host IP
                    return ip
    except (FileNotFoundError, IndexError, PermissionError):
        pass
    
    try:
        # Method 2: Use ip route (fallback)
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if "via" in parts:
                idx = parts.index("via")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
        pass
    
    return None


def open_capture(source: str) -> cv2.VideoCapture:
    src_arg: Union[int, str]
    path_candidate = Path(source)
    
    # Check if it's a numeric camera ID
    if source.isdigit() and not path_candidate.exists():
        src_arg = int(source)
    else:
        src_arg = source
    
    # For HTTP/RTSP streams, handle WSL2 localhost issue
    if isinstance(src_arg, str) and (src_arg.startswith("http://") or src_arg.startswith("rtsp://")):
        # If using localhost in WSL2, try to resolve to Windows host IP
        if "localhost" in src_arg or "127.0.0.1" in src_arg:
            windows_ip = get_windows_host_ip()
            if windows_ip:
                # Replace localhost with Windows host IP
                original_url = src_arg
                src_arg = src_arg.replace("localhost", windows_ip).replace("127.0.0.1", windows_ip)
                print(f"WSL2 detected: Using Windows host IP {windows_ip}")
                print(f"Connecting to: {src_arg}")
            else:
                print("Warning: Could not detect Windows host IP. Trying localhost...")
                print("If connection fails, try using the Windows host IP directly.")
        
        # Use FFMPEG backend for network streams (more reliable)
        cap = cv2.VideoCapture(src_arg, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(src_arg)
    
    if not cap.isOpened():
        error_msg = f"Failed to open video source: {source}\n"
        error_msg += "For HTTP streams, make sure the server is running.\n"
        
        if isinstance(src_arg, str) and ("http://" in src_arg or "rtsp://" in src_arg):
            windows_ip = get_windows_host_ip()
            if windows_ip:
                suggested_url = src_arg.replace("localhost", windows_ip).replace("127.0.0.1", windows_ip)
                error_msg += f"\nIf you're in WSL2, try using the Windows host IP:\n"
                error_msg += f"  --source {suggested_url}\n"
            error_msg += "\nOr find your Windows IP with: ip route show default | grep -oP 'via \K\S+'"
        
        error_msg += "\nFor camera access in WSL, use webcam_stream_server.py on Windows."
        raise RuntimeError(error_msg)
    return cap


def select_or_use_bbox(frame: np.ndarray, cli_bbox: Optional[Sequence[float]]) -> Sequence[float]:
    if cli_bbox is not None:
        return cli_bbox
    print("Draw a bounding box around the target object, then press ENTER.")
    try:
        x, y, w, h = cv2.selectROI("Select Object", frame, showCrosshair=False)
        cv2.destroyWindow("Select Object")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "OpenCV could not open a GUI window for ROI selection. "
            "If you're running in a headless environment (e.g., WSL without DISPLAY), "
            "rerun the demo with --bbox x0 y0 x1 y1 to skip the GUI."
        ) from e
    if w == 0 or h == 0:
        raise RuntimeError("ROI selection cancelled; please provide --bbox instead.")
    return (x, y, x + w, y + h)


def format_masks(mask_logits: torch.Tensor) -> torch.Tensor:
    """Ensure mask tensor shape is (num_obj, H, W) and apply sigmoid."""
    if mask_logits.ndim == 4 and mask_logits.shape[1] == 1:
        mask_logits = mask_logits[:, 0]
    elif mask_logits.ndim == 4:
        mask_logits = mask_logits.squeeze(1)
    elif mask_logits.ndim == 3 and mask_logits.shape[0] == 1:
        mask_logits = mask_logits.squeeze(0)
    mask_probs = mask_logits.sigmoid()
    if mask_probs.dim() == 2:
        mask_probs = mask_probs.unsqueeze(0)
    return mask_probs


def overlay_masks(
    frame: np.ndarray,
    obj_ids,
    mask_logits: torch.Tensor,
    threshold: float,
    alpha: float,
    bboxes: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    output = frame.copy()
    
    # Draw masks if available
    if mask_logits is not None:
        mask_probs = format_masks(mask_logits).detach().cpu().numpy()
        for idx, obj_id in enumerate(obj_ids):
            if idx >= mask_probs.shape[0]:
                break
            mask = mask_probs[idx] > threshold
            if not np.any(mask):
                continue
            color = np.array(PALETTE[idx % len(PALETTE)], dtype=np.float32)
            overlay_region = output[mask]
            blended = (1.0 - alpha) * overlay_region + alpha * color
            output[mask] = blended
            
            # Find centroid of mask for label placement
            mask_uint8 = (mask * 255).astype(np.uint8)
            moments = cv2.moments(mask_uint8)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = 10, 30 + idx * 25
            
            # Draw object ID label at mask centroid
            label = f"Object ID: {obj_id}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            # Draw background rectangle for text
            cv2.rectangle(
                output,
                (cx - text_width // 2 - 5, cy - text_height - 5),
                (cx + text_width // 2 + 5, cy + baseline + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                output,
                label,
                (cx - text_width // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
    
    # Draw bounding boxes if provided
    if bboxes is not None:
        for idx, bbox in enumerate(bboxes):
            if len(bbox) == 4:
                x0, y0, x1, y1 = map(int, bbox)
                color = tuple(map(int, PALETTE[idx % len(PALETTE)]))
                # Draw bounding box with thicker line
                cv2.rectangle(output, (x0, y0), (x1, y1), color, 3)
                # Draw label at top-left corner of bbox
                if idx < len(obj_ids):
                    label = f"BBox ID: {obj_ids[idx]}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    # Background for bbox label
                    cv2.rectangle(
                        output,
                        (x0, y0 - text_height - 8),
                        (x0 + text_width + 4, y0),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        output,
                        label,
                        (x0 + 2, y0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
    
    return output


def initialize_predictor(
    predictor,
    frame: np.ndarray,
    obj_id: int,
    bbox: Sequence[float],
    is_first_object: bool = False,
):
    if is_first_object:
        predictor.load_first_frame(frame)
    
    _, obj_ids, mask_logits = predictor.add_new_prompt(
        frame_idx=0,
        obj_id=obj_id,
        bbox=bbox,
        clear_old_points=True,
        normalize_coords=True,
    )
    return obj_ids, mask_logits


class VisualizationHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves visualization frames."""

    def __init__(self, frame_queue: queue.Queue, *args, **kwargs):
        self.frame_queue = frame_queue
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                frame_count = 0
                while True:
                    try:
                        frame = self.frame_queue.get(timeout=2.0)
                        frame_count += 1
                        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if buffer is None:
                            continue
                        frame_bytes = buffer.tobytes()
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()  # Ensure frame is sent immediately
                    except queue.Empty:
                        # Send a keepalive message if no frames for a while
                        if frame_count == 0:
                            # First connection, send a placeholder
                            placeholder = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 0\r\n\r\n\r\n"
                            self.wfile.write(placeholder)
                            self.wfile.flush()
                        continue
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                # Client disconnected, which is normal
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def do_HEAD(self):
        if self.path == "/" or self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_visualization_server(port: int, frame_queue: queue.Queue) -> Tuple[HTTPServer, threading.Thread]:
    """Start HTTP server for visualization streaming."""
    def handler_factory(queue):
        def create_handler(*args, **kwargs):
            return VisualizationHandler(queue, *args, **kwargs)
        return create_handler

    server = HTTPServer(("0.0.0.0", port), handler_factory(frame_queue))
    # Allow socket reuse to avoid "Address already in use" errors
    server.allow_reuse_address = True
    
    def run_server():
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            # Ensure socket is closed
            try:
                server.server_close()
            except Exception:
                pass

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Get WSL IP for display
    windows_ip = get_windows_host_ip()
    print(f"\n{'='*60}")
    print(f"Visualization server started!")
    print(f"View the segmentation in your browser:")
    if windows_ip:
        print(f"  http://{windows_ip}:{port}/video")
    print(f"  http://localhost:{port}/video")
    print(f"{'='*60}\n")
    
    return server, thread


def main() -> None:
    args = parse_args()

    # Global variables for cleanup
    cleanup_vars = {"cap": None, "viewer_server": None, "viewer_thread": None, "gui_available": False}

    def cleanup_handler(signum, frame):
        """Handle termination signals gracefully."""
        print("\n\nShutting down gracefully...")
        if cleanup_vars["cap"] is not None:
            try:
                cleanup_vars["cap"].release()
            except Exception:
                pass
        if cleanup_vars["viewer_server"] is not None:
            # Shutdown server - this unblocks serve_forever()
            try:
                cleanup_vars["viewer_server"].shutdown()
                # Close the socket to release the port immediately
                cleanup_vars["viewer_server"].server_close()
            except Exception:
                pass
        if cleanup_vars["gui_available"]:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("Cleanup complete. Exiting.")
        os._exit(0)  # Force exit to avoid hanging

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_handler)  # kill command
    signal.signal(signal.SIGTSTP, cleanup_handler)   # Ctrl+Z (suspend)

    predictor = build_sam2_camera_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
        vos_optimized=args.vos_optimized,
    )

    cap = open_capture(args.source)
    cleanup_vars["cap"] = cap
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab the first frame from the stream.")

    # Prepare bounding boxes / objects
    provided_bboxes = args.bbox or []
    num_objects = max(args.num_objects, len(provided_bboxes) or 1)

    bbox_list: List[Sequence[float]] = []
    all_obj_ids: List[int] = []
    mask_logits_list: List[torch.Tensor] = []

    for obj_idx in range(num_objects):
        obj_id = args.obj_id + obj_idx
        cli_bbox = provided_bboxes[obj_idx] if obj_idx < len(provided_bboxes) else None
        bbox = select_or_use_bbox(first_frame, cli_bbox)
        bbox_list.append(bbox)

        print(f"Initializing object {obj_idx + 1} (ID={obj_id}) with bbox: {bbox}")
        obj_ids, mask_logits = initialize_predictor(
            predictor,
            first_frame,
            obj_id,
            bbox,
            is_first_object=(obj_idx == 0),
        )
        all_obj_ids.extend(list(obj_ids))
        if mask_logits is not None:
            if mask_logits.dim() == 2:
                mask_logits = mask_logits.unsqueeze(0)
            mask_logits_list.append(mask_logits)

    init_mask_logits = (
        torch.cat(mask_logits_list, dim=0) if mask_logits_list else None
    )

    # Setup visualization
    frame_queue = None
    if args.viewer_port:
        frame_queue = queue.Queue(maxsize=2)  # Keep only latest frame
        viewer_server, viewer_thread = start_visualization_server(args.viewer_port, frame_queue)
        cleanup_vars["viewer_server"] = viewer_server
        cleanup_vars["viewer_thread"] = viewer_thread
        # Put first frame
        vis_frame = overlay_masks(
            first_frame,
            all_obj_ids,
            init_mask_logits,
            threshold=args.mask_threshold,
            alpha=args.alpha,
            bboxes=bbox_list,
        )
        try:
            frame_queue.put_nowait(vis_frame.copy())
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put_nowait(vis_frame.copy())
        print("First frame queued for visualization server")
    
    # Try to open GUI window (may fail in WSL, that's OK)
    window_name = "SAM 2 Camera Predictor"
    gui_available = False
    cleanup_vars["gui_available"] = False
    should_try_gui = (not args.no_gui) and (os.environ.get("DISPLAY") or os.name == "nt")
    if should_try_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            vis_frame = overlay_masks(
                first_frame,
                all_obj_ids,
                init_mask_logits,
                threshold=args.mask_threshold,
                alpha=args.alpha,
                bboxes=bbox_list,
            )
            cv2.imshow(window_name, vis_frame)
            cv2.waitKey(1)
            gui_available = True
            cleanup_vars["gui_available"] = True
        except Exception as e:  # noqa: BLE001
            if args.viewer_port:
                print(f"GUI not available (this is OK): {e}")
                print(f"Using web viewer at http://localhost:{args.viewer_port}/video instead")
            else:
                print(f"Warning: GUI not available and no viewer port specified.")
                print(f"Add --viewer-port 8081 to view results in browser.")
    else:
        if args.viewer_port:
            print("GUI disabled (--no-gui or no DISPLAY). Using web viewer only.")
        else:
            print("GUI disabled (--no-gui or no DISPLAY) and no viewer-port set; "
                  "use --viewer-port to visualize results in a browser.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-2:
        fps = 30.0
    writer = None
    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height))
        writer.write(vis_frame)

    autocast_device = "cuda" if args.device.startswith("cuda") else "cpu"
    use_autocast = autocast_device == "cuda" and torch.cuda.is_available() and not args.no_autocast
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_autocast
        else contextlib.nullcontext()
    )

    start_time = time.time()
    frame_counter = 0

    with torch.inference_mode():
        with autocast_ctx:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                obj_ids, mask_logits = predictor.track(frame)
                vis_frame = overlay_masks(
                    frame,
                    obj_ids,
                    mask_logits,
                    threshold=args.mask_threshold,
                    alpha=args.alpha,
                    bboxes=bbox_list,  # Show initial bboxes for reference
                )
                frame_counter += 1
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps_text = f"FPS: {frame_counter / elapsed:.1f}"
                    cv2.putText(
                        vis_frame,
                        fps_text,
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )

                # Send to viewer server if enabled
                if frame_queue is not None:
                    try:
                        frame_queue.put_nowait(vis_frame.copy())  # Make a copy to avoid issues
                    except queue.Full:
                        # Remove old frame and add new one
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            frame_queue.put_nowait(vis_frame.copy())
                        except queue.Full:
                            pass  # Skip if still full

                # Display in GUI if available
                if gui_available:
                    cv2.imshow(window_name, vis_frame)
                if writer is not None:
                    writer.write(vis_frame)

                if gui_available:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        print("Re-initializing with the current frame...")
                        predictor.reset_state()
                        mask_logits_list = []
                        all_obj_ids = []
                        for idx, bbox in enumerate(bbox_list):
                            obj_id = args.obj_id + idx
                            obj_ids, mask_logits_obj = initialize_predictor(
                                predictor,
                                frame,
                                obj_id,
                                bbox,
                                is_first_object=(idx == 0),
                            )
                            all_obj_ids.extend(list(obj_ids))
                            if mask_logits_obj is not None:
                                if mask_logits_obj.dim() == 2:
                                    mask_logits_obj = mask_logits_obj.unsqueeze(0)
                                mask_logits_list.append(mask_logits_obj)
                        mask_logits = (
                            torch.cat(mask_logits_list, dim=0)
                            if mask_logits_list
                            else None
                        )
                        vis_frame = overlay_masks(
                            frame,
                            all_obj_ids,
                            mask_logits,
                            threshold=args.mask_threshold,
                            alpha=args.alpha,
                            bboxes=bbox_list,
                        )
                        if frame_queue is not None:
                            try:
                                frame_queue.put_nowait(vis_frame.copy())
                            except queue.Full:
                                try:
                                    frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                frame_queue.put_nowait(vis_frame.copy())
                        if gui_available:
                            cv2.imshow(window_name, vis_frame)
                        frame_counter = 0
                        start_time = time.time()
                else:
                    # In headless mode, check for interrupt
                    time.sleep(0.01)  # Small delay to prevent CPU spinning

    # Cleanup
    print("\nCleaning up...")
    if cleanup_vars["cap"] is not None:
        cleanup_vars["cap"].release()
    if writer is not None:
        writer.release()
    if cleanup_vars["viewer_server"] is not None:
        print("Shutting down visualization server...")
        try:
            cleanup_vars["viewer_server"].shutdown()
            cleanup_vars["viewer_server"].server_close()  # Close socket to release port
        except Exception:
            pass
        if cleanup_vars["viewer_thread"] is not None:
            cleanup_vars["viewer_thread"].join(timeout=1.0)
    if cleanup_vars["gui_available"]:
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

