#!/usr/bin/env python3
"""
Test client for SAM2 Mobile Server.

This script demonstrates how to use the SAM2 Mobile Server API from a mobile client.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import requests


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def test_initialize_session(server_url: str, image_path: str, bboxes: list):
    """Test the /initialize endpoint."""
    print(f"\n=== Testing /initialize ===")
    
    # Encode image
    image_data = encode_image(image_path)
    
    # Prepare request
    payload = {
        "session_id": "test_session_1",
        "image": image_data,
        "bounding_boxes": bboxes,
        "object_ids": [1, 2] if len(bboxes) >= 2 else [1]
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


def test_track_frame(server_url: str, session_id: str, image_path: str):
    """Test the /track endpoint."""
    print(f"\n=== Testing /track ===")
    
    # Encode image
    image_data = encode_image(image_path)
    
    # Prepare request
    payload = {
        "session_id": session_id,
        "image": image_data
    }
    
    # Send request
    response = requests.post(f"{server_url}/track", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Frame tracked successfully")
        print(f"  Session ID: {result['session_id']}")
        print(f"  Object IDs: {result['object_ids']}")
        print(f"  Bounding boxes: {result['bounding_boxes']}")
        print(f"  Frame index: {result['frame_index']}")
        return True
    else:
        print(f"✗ Failed to track frame: {response.status_code}")
        print(f"  Error: {response.text}")
        return False


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
        required=True,
        help="Path to test image",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        action="append",
        help="Bounding box (repeat for multiple objects)",
    )
    
    args = parser.parse_args()
    
    if not args.bbox:
        print("Error: At least one bounding box is required (--bbox x0 y0 x1 y1)")
        sys.exit(1)
    
    # Test health check
    if not test_health_check(args.server_url):
        print("\n✗ Server is not responding. Make sure the server is running.")
        sys.exit(1)
    
    # Test initialize
    session_id = test_initialize_session(args.server_url, args.image, args.bbox)
    if not session_id:
        sys.exit(1)
    
    # Test track (use same image for simplicity)
    test_track_frame(args.server_url, session_id, args.image)
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    main()

