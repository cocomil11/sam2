#!/usr/bin/env python3
"""
Test client for SAM2 Mobile Server.

This script demonstrates how to use the SAM2 Mobile Server API from a mobile client.
Supports both single frame testing and continuous streaming mode.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def encode_frame(frame: np.ndarray) -> str:
    """Encode OpenCV frame (numpy array) to base64 string."""
    # Encode frame as JPEG
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode frame as JPEG")
    image_bytes = buffer.tobytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def open_camera(camera_id: int = 0):
    """Open camera capture."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    
    # Set reasonable resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap


def annotate_frame_interactive(frame: np.ndarray, existing_bboxes: list = None) -> list:
    """
    Interactive annotation: Show frame and let user add bounding boxes.
    
    Returns:
        List of bounding boxes in format [[x0, y0, x1, y1], ...]
    """
    bboxes = existing_bboxes.copy() if existing_bboxes else []
    window_name = "Annotate Frame - Press 'a' to add box, 'd' to delete last, 'Enter' to confirm, 'q' to quit"
    
    # Create a copy for drawing
    display_frame = frame.copy()
    
    def draw_bboxes(img, boxes):
        """Draw all bounding boxes on image."""
        img_copy = img.copy()
        for idx, bbox in enumerate(boxes):
            if len(bbox) == 4:
                x0, y0, x1, y1 = map(int, bbox)
                color = (0, 255, 0) if idx == len(boxes) - 1 else (255, 0, 0)
                thickness = 3 if idx == len(boxes) - 1 else 2
                cv2.rectangle(img_copy, (x0, y0), (x1, y1), color, thickness)
                # Draw label
                label = f"Obj {idx + 1}"
                cv2.putText(img_copy, label, (x0, y0 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img_copy
    
    print("\n=== Interactive Annotation Mode ===")
    print("Instructions:")
    print("  - Press 'a' to add a new bounding box")
    print("  - Press 'd' to delete the last bounding box")
    print("  - Press 'Enter' or 'Space' to confirm and start tracking")
    print("  - Press 'q' to quit without starting")
    print()
    
    while True:
        # Draw current bounding boxes
        display_frame = draw_bboxes(frame, bboxes)
        
        # Show current status
        status_text = f"Bounding boxes: {len(bboxes)}"
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):
            # Add new bounding box
            print("Draw a bounding box with your mouse, then release...")
            try:
                x, y, w, h = cv2.selectROI(window_name, display_frame, showCrosshair=True)
                cv2.destroyWindow(window_name)
                if w > 0 and h > 0:
                    bbox = [float(x), float(y), float(x + w), float(y + h)]
                    bboxes.append(bbox)
                    print(f"  Added bounding box {len(bboxes)}: {bbox}")
                else:
                    print("  Bounding box selection cancelled")
            except Exception as e:
                print(f"  Error selecting bounding box: {e}")
        
        elif key == ord('d'):
            # Delete last bounding box
            if bboxes:
                removed = bboxes.pop()
                print(f"  Removed bounding box: {removed}")
                print(f"  Remaining: {len(bboxes)} bounding boxes")
            else:
                print("  No bounding boxes to remove")
        
        elif key == ord('\r') or key == ord('\n') or key == ord(' '):  # Enter or Space
            # Confirm and start
            if bboxes:
                print(f"\n✓ Confirmed {len(bboxes)} bounding box(es). Starting tracking...")
                cv2.destroyAllWindows()
                return bboxes
            else:
                print("  No bounding boxes added. Please add at least one bounding box.")
        
        elif key == ord('q'):
            # Quit
            print("\n✗ Annotation cancelled by user")
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return bboxes


def test_initialize_session(server_url: str, image_path: str = None, frame: np.ndarray = None, bboxes: list = None, session_id: str = None, is_interactive: bool = False):
    """Test the /initialize endpoint."""
    print(f"\n=== Testing /initialize ===")
    
    # Encode image or frame
    if frame is not None:
        image_data = encode_frame(frame)
    elif image_path:
        image_data = encode_image(image_path)
    else:
        raise ValueError("Either image_path or frame must be provided")
    
    # Use provided session_id or default
    if session_id is None:
        if is_interactive:
            session_id = "interactive_session_1"
        else:
            session_id = "test_session_1"
    
    # Prepare request
    payload = {
        "session_id": session_id,
        "image": image_data,
        "bounding_boxes": bboxes or [],
        "object_ids": [1, 2] if bboxes and len(bboxes) >= 2 else [1]
    }
    
    # Send request
    response = requests.post(f"{server_url}/initialize", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Session initialized successfully")
        print(f"  Session ID: {result['session_id']}")
        print(f"  Object IDs: {result['object_ids']}")
        print(f"  Bounding boxes: {result['bounding_boxes']}")
        print(f"  Frame shape: {result['frame_shape']}")
        return result["session_id"]
    else:
        print(f"✗ Failed to initialize session: {response.status_code}")
        print(f"  Error: {response.text}")
        return None


def test_track_frame(server_url: str, session_id: str, image_path: str = None, frame: np.ndarray = None, verbose: bool = True):
    """Test the /track endpoint."""
    if verbose:
        print(f"\n=== Testing /track ===")
    
    # Encode image or frame
    if frame is not None:
        image_data = encode_frame(frame)
    elif image_path:
        image_data = encode_image(image_path)
    else:
        raise ValueError("Either image_path or frame must be provided")
    
    # Prepare request
    payload = {
        "session_id": session_id,
        "image": image_data
    }
    
    # Send request
    response = requests.post(f"{server_url}/track", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        if verbose:
            print(f"✓ Frame tracked successfully")
            print(f"  Session ID: {result['session_id']}")
            print(f"  Object IDs: {result['object_ids']}")
            print(f"  Bounding boxes: {result['bounding_boxes']}")
            print(f"  Frame index: {result['frame_index']}")
        return True, result
    else:
        if verbose:
            print(f"✗ Failed to track frame: {response.status_code}")
            print(f"  Error: {response.text}")
        return False, None


def test_stream_frames(server_url: str, session_id: str, image_path: str = None, camera_id: int = None, fps: float = 2.0, duration: float = None, max_frames: int = None):
    """Test streaming frames at specified FPS from image file or camera."""
    print(f"\n=== Testing Stream Mode ===")
    print(f"  FPS: {fps}")
    if camera_id is not None:
        print(f"  Source: Camera {camera_id}")
    elif image_path:
        print(f"  Source: Image file")
    if duration:
        print(f"  Duration: {duration} seconds")
    if max_frames:
        print(f"  Max frames: {max_frames}")
    print()
    
    # Open camera if specified
    cap = None
    if camera_id is not None:
        try:
            cap = open_camera(camera_id)
            print(f"✓ Camera {camera_id} opened successfully")
        except Exception as e:
            print(f"✗ Failed to open camera {camera_id}: {e}")
            return False
    
    frame_interval = 1.0 / fps
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # Check duration limit
            if duration and (time.time() - start_time) >= duration:
                print(f"\n✓ Stream completed: Duration limit reached ({duration}s)")
                break
            
            # Check frame limit
            if max_frames and frame_count >= max_frames:
                print(f"\n✓ Stream completed: Frame limit reached ({max_frames} frames)")
                break
            
            frame_start = time.time()
            
            # Get frame from camera or use image file
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print(f"\n✗ Failed to read frame from camera")
                    break
                # Track frame from camera
                success, result = test_track_frame(server_url, session_id, frame=frame, verbose=False)
            else:
                # Track frame from image file
                success, result = test_track_frame(server_url, session_id, image_path=image_path, verbose=False)
            
            if not success:
                print(f"\n✗ Stream failed at frame {frame_count + 1}")
                if cap:
                    cap.release()
                return False
            
            frame_count += 1
            if result:
                print(f"Frame {frame_count:4d}: IDs={result['object_ids']}, "
                      f"BBoxes={len(result['bounding_boxes'])}, "
                      f"Index={result['frame_index']}", end='\r')
            
            # Calculate sleep time to maintain FPS
            elapsed = time.time() - frame_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print(f"\n\n✓ Stream interrupted by user after {frame_count} frames")
        if cap:
            cap.release()
        return True
    
    finally:
        if cap:
            cap.release()
    
    elapsed_time = time.time() - start_time
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n✓ Stream completed successfully")
    print(f"  Total frames: {frame_count}")
    print(f"  Elapsed time: {elapsed_time:.2f}s")
    print(f"  Actual FPS: {actual_fps:.2f}")
    return True


def test_interactive_annotation_and_stream(server_url: str, camera_id: int, fps: float = 2.0, duration: float = None, max_frames: int = None):
    """
    Interactive annotation workflow:
    1. Show first frame from camera
    2. User annotates with bounding boxes
    3. User confirms
    4. Send first frame with bounding boxes to /initialize
    5. Start streaming subsequent frames to /track for 10 seconds (skipping frames during annotation)
    
    Args:
        duration: Stream duration in seconds. Defaults to 10.0 if not specified.
    """
    # Default to 10 seconds if duration not specified
    if duration is None:
        duration = 10.0
    
    print(f"\n=== Interactive Annotation and Streaming Mode ===")
    print(f"  Camera: {camera_id}")
    print(f"  Streaming FPS: {fps}")
    print(f"  Duration: {duration} seconds")
    if max_frames:
        print(f"  Max frames: {max_frames}")
    print()
    
    # Open camera
    try:
        cap = open_camera(camera_id)
        print(f"✓ Camera {camera_id} opened successfully")
    except Exception as e:
        print(f"✗ Failed to open camera {camera_id}: {e}")
        return False
    
    try:
        # Step 1: Capture first frame for annotation
        print("\n--- Step 1: Capturing first frame for annotation ---")
        ret, first_frame = cap.read()
        if not ret:
            print("✗ Failed to read first frame from camera")
            return False
        
        print(f"✓ First frame captured: {first_frame.shape[1]}x{first_frame.shape[0]}")
        
        # Step 2: Interactive annotation
        print("\n--- Step 2: Interactive Annotation ---")
        print("The first frame will be displayed. Add bounding boxes to define what to track.")
        bboxes = annotate_frame_interactive(first_frame)
        
        if not bboxes:
            print("✗ No bounding boxes provided. Exiting.")
            return False
        
        print(f"✓ Annotation complete: {len(bboxes)} bounding box(es)")
        for idx, bbox in enumerate(bboxes):
            print(f"  Box {idx + 1}: {bbox}")
        
        # Step 3: Initialize session with first frame and bounding boxes
        print("\n--- Step 3: Initializing tracking session ---")
        session_id = test_initialize_session(server_url, frame=first_frame, bboxes=bboxes, is_interactive=True)
        
        if not session_id:
            print("✗ Failed to initialize session")
            return False
        
        # Step 4: Discard frames captured during annotation
        # (These frames were shown to user but not sent to server)
        print("\n--- Step 4: Starting frame stream (skipping frames during annotation) ---")
        print("Now streaming subsequent frames to server...")
        print("(Frames captured during annotation are skipped)")
        
        # Small delay to ensure we get a fresh frame
        time.sleep(0.1)
        
        # Step 5: Stream subsequent frames
        frame_interval = 1.0 / fps
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                # Check duration limit (duration is always set in interactive mode)
                if (time.time() - start_time) >= duration:
                    print(f"\n✓ Stream completed: Duration limit reached ({duration}s)")
                    break
                
                # Check frame limit
                if max_frames and frame_count >= max_frames:
                    print(f"\n✓ Stream completed: Frame limit reached ({max_frames} frames)")
                    break
                
                frame_start = time.time()
                
                # Get fresh frame from camera
                ret, frame = cap.read()
                if not ret:
                    print(f"\n✗ Failed to read frame from camera")
                    break
                
                # Track frame
                success, result = test_track_frame(server_url, session_id, frame=frame, verbose=False)
                
                if not success:
                    print(f"\n✗ Stream failed at frame {frame_count + 1}")
                    return False
                
                frame_count += 1
                if result:
                    print(f"Frame {frame_count:4d}: IDs={result['object_ids']}, "
                          f"BBoxes={len(result['bounding_boxes'])}, "
                          f"Index={result['frame_index']}", end='\r')
                
                # Calculate sleep time to maintain FPS
                elapsed = time.time() - frame_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print(f"\n\n✓ Stream interrupted by user after {frame_count} frames")
            return True
        
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n✓ Stream completed successfully")
        print(f"  Total frames sent: {frame_count}")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print(f"  Actual FPS: {actual_fps:.2f}")
        return True
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def test_health_check(server_url: str):
    """Test the /health endpoint."""
    print(f"\n=== Testing /health ===")
    
    response = requests.get(f"{server_url}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Server is healthy")
        print(f"  Predictor loaded: {result['predictor_loaded']}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SAM2 Mobile Server")
    parser.add_argument(
        "--server-url",
        default="http://localhost:8080",
        help="Server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to test image (required if --camera not specified)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device ID (e.g., 0 for default camera). If specified, uses camera instead of image file.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        action="append",
        help="Bounding box (repeat for multiple objects)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode (continuously send frames)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for streaming mode (default: 2.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stream duration in seconds (default: unlimited for non-interactive, 10.0 for interactive mode)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to send (default: unlimited)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive annotation mode: show first frame, let user annotate, then stream for 10 seconds",
    )
    
    args = parser.parse_args()
    
    # Interactive annotation mode (camera only)
    if args.interactive:
        if args.camera is None:
            print("Error: --interactive mode requires --camera to be specified")
            sys.exit(1)
        
        # Test health check
        if not test_health_check(args.server_url):
            print("\n✗ Server is not responding. Make sure the server is running.")
            sys.exit(1)
        
        # Run interactive annotation and streaming
        success = test_interactive_annotation_and_stream(
            args.server_url,
            args.camera,
            fps=args.fps,
            duration=args.duration,
            max_frames=args.max_frames
        )
        
        if not success:
            sys.exit(1)
        
        print("\n✓ All tests completed!")
        return
    
    # Non-interactive mode (original behavior)
    # Validate input source
    if args.camera is None and args.image is None:
        print("Error: Either --image or --camera must be specified")
        sys.exit(1)
    
    if not args.bbox:
        print("Error: At least one bounding box is required (--bbox x0 y0 x1 y1)")
        print("       Or use --interactive mode to annotate interactively")
        sys.exit(1)
    
    # Test health check
    if not test_health_check(args.server_url):
        print("\n✗ Server is not responding. Make sure the server is running.")
        sys.exit(1)
    
    # Initialize session
    cap = None
    try:
        if args.camera is not None:
            # Use camera for initialization
            cap = open_camera(args.camera)
            ret, frame = cap.read()
            if not ret:
                print(f"✗ Failed to read frame from camera {args.camera}")
                sys.exit(1)
            session_id = test_initialize_session(args.server_url, frame=frame, bboxes=args.bbox)
        else:
            # Use image file for initialization
            session_id = test_initialize_session(args.server_url, image_path=args.image, bboxes=args.bbox)
        
        if not session_id:
            sys.exit(1)
        
        # Test track or stream
        if args.stream:
            # Streaming mode
            success = test_stream_frames(
                args.server_url,
                session_id,
                image_path=args.image if args.camera is None else None,
                camera_id=args.camera,
                fps=args.fps,
                duration=args.duration,
                max_frames=args.max_frames
            )
            if not success:
                sys.exit(1)
        else:
            # Single frame mode
            if args.camera is not None:
                if cap is None:
                    cap = open_camera(args.camera)
                ret, frame = cap.read()
                if ret:
                    test_track_frame(args.server_url, session_id, frame=frame)
                else:
                    print("✗ Failed to read frame from camera")
                    sys.exit(1)
            else:
                test_track_frame(args.server_url, session_id, image_path=args.image)
    
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    main()

