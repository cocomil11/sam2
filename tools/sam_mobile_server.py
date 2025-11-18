#!/usr/bin/env python3
"""
SAM2 Mobile Server - Receives video streams and bounding boxes from mobile clients
and returns segmentation results (bounding boxes and object IDs).

Usage:
    python tools/sam_mobile_server.py \
        --config sam2/configs/sam2.1/sam2.1_hiera_s.yaml \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --port 8080
"""

import argparse
import base64
import io
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from sam2.build_sam import build_sam2_camera_predictor
from sam2.utils.amg import batched_mask_to_box

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile clients

# Global predictor instance
predictor = None
predictor_lock = threading.Lock()
session_states: Dict[str, Dict] = {}  # Store session states for tracking across frames
args = None  # Global args for access in route handlers

# Note: Currently using a single global predictor instance.
# Each session initialization resets the predictor state via load_first_frame().
# This means only one active session is supported at a time.
# For multiple concurrent sessions, we would need separate predictor instances per session.

# Fixed BGR color palette for up to eight objects (same as camera_stream_demo.py)
PALETTE = (
    (0, 255, 0),      # Green
    (0, 165, 255),    # Orange
    (255, 0, 0),      # Blue
    (255, 105, 180),  # Pink
    (255, 255, 0),    # Cyan
    (138, 43, 226),   # Purple
    (0, 255, 255),    # Yellow
    (255, 140, 0),    # Dark Orange
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 Mobile Server - Receives video streams and bounding boxes from mobile clients."
    )
    parser.add_argument(
        "--config",
        default="sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
        help="Path to SAM 2 config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam2.1_hiera_small.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for inference (default: cuda).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binary masks (default: 0.5).",
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
        "--save-images",
        action="store_true",
        help="Save analyzed images to disk for testing/debugging.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mobile_server_images",
        help="Directory to save analyzed images (default: outputs/mobile_server_images).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Overlay opacity for masks in [0, 1] (default: 0.6).",
    )
    return parser.parse_args()


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 encoded image or handle raw bytes."""
    try:
        # Try to decode as base64
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise


def extract_bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    """Extract bounding box from binary mask in format [x0, y0, x1, y1]."""
    if not np.any(mask):
        return None
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    
    return [float(x0), float(y0), float(x1 + 1), float(y1 + 1)]  # x1, y1 are exclusive


def extract_bboxes_from_masks(mask_probs: torch.Tensor, threshold: float = 0.5) -> List[Optional[List[float]]]:
    """Extract bounding boxes from mask probabilities."""
    try:
        # Ensure shape is (num_obj, H, W)
        if mask_probs.dim() == 2:
            mask_probs = mask_probs.unsqueeze(0)
        elif mask_probs.dim() == 4:
            # If shape is (B, C, H, W), squeeze channel dim
            if mask_probs.shape[1] == 1:
                mask_probs = mask_probs.squeeze(1)
            else:
                # Flatten batch and channel dims
                B, C, H, W = mask_probs.shape
                mask_probs = mask_probs.view(B * C, H, W)
        
        # Convert to numpy for easier processing
        mask_probs_np = mask_probs.detach().cpu().numpy()
        bboxes = []
        
        # Extract bounding box for each mask
        for i in range(mask_probs_np.shape[0]):
            mask = mask_probs_np[i] > threshold
            bbox = extract_bbox_from_mask(mask)
            bboxes.append(bbox)
        
        return bboxes
    except Exception as e:
        logger.error(f"Error extracting bounding boxes: {e}", exc_info=True)
        # Fallback: return None for all objects
        if mask_probs.dim() >= 2:
            num_objs = mask_probs.shape[0] if mask_probs.dim() >= 3 else 1
            return [None] * num_objs
        return []


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


def overlay_masks_and_bboxes(
    frame: np.ndarray,
    obj_ids: List[int],
    mask_logits: Optional[torch.Tensor],
    threshold: float,
    alpha: float,
    bboxes: Optional[List[List[float]]] = None,
) -> np.ndarray:
    """Overlay masks and bounding boxes on frame for visualization."""
    output = frame.copy()
    
    # Draw masks if available
    if mask_logits is not None:
        mask_probs = format_masks(mask_logits).detach().cpu().numpy()
        frame_h, frame_w = output.shape[:2]
        
        for idx, obj_id in enumerate(obj_ids):
            if idx >= mask_probs.shape[0]:
                break
            mask = mask_probs[idx] > threshold
            
            # Resize mask to match frame dimensions if needed
            mask_h, mask_w = mask.shape[:2]
            if mask_h != frame_h or mask_w != frame_w:
                mask = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            if not np.any(mask):
                continue
            color = np.array(PALETTE[idx % len(PALETTE)], dtype=np.float32)
            overlay_region = output[mask]
            blended = (1.0 - alpha) * overlay_region + alpha * color
            output[mask] = blended.astype(np.uint8)
    
    # Draw bounding boxes if provided
    if bboxes is not None:
        for idx, bbox in enumerate(bboxes):
            if len(bbox) == 4:
                x0, y0, x1, y1 = map(int, bbox)
                color = tuple(map(int, PALETTE[idx % len(PALETTE)]))
                # Draw bounding box
                cv2.rectangle(output, (x0, y0), (x1, y1), color, 3)
                # Draw label
                if idx < len(obj_ids):
                    label = f"ID: {obj_ids[idx]}"
                else:
                    label = f"Obj {idx + 1}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                # Background for label
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


def save_analyzed_image(
    frame: np.ndarray,
    session_id: str,
    frame_index: int,
    obj_ids: List[int],
    mask_logits: Optional[torch.Tensor],
    bboxes: Optional[List[List[float]]],
    output_dir: str,
    threshold: float = 0.5,
    alpha: float = 0.6,
    is_interactive: bool = False,
):
    """Save analyzed image with masks and bounding boxes overlaid.
    
    Args:
        is_interactive: If True, saves to a subfolder named after the session_id.
    """
    try:
        # Use subfolder for interactive mode - create a directory per session
        if is_interactive:
            output_dir = os.path.join(output_dir, session_id)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Overlay masks and bounding boxes
        vis_frame = overlay_masks_and_bboxes(
            frame, obj_ids, mask_logits, threshold, alpha, bboxes
        )
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_frame{frame_index:04d}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, vis_frame)
        logger.info(f"Saved analyzed image: {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving analyzed image: {e}", exc_info=True)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "predictor_loaded": predictor is not None})


@app.route("/initialize", methods=["POST"])
def initialize_session():
    global args
    """
    Initialize a new tracking session with initial frame and bounding boxes.
    
    This endpoint:
    1. Receives the first frame and bounding boxes from the mobile client
    2. Initializes SAM2 to track the objects defined by the bounding boxes
    3. Returns the initial segmentation results (masks converted to bounding boxes)
    
    IMPORTANT: The bounding boxes are only needed for the FIRST request.
    Subsequent frames sent via /track will automatically track these objects.
    
    Request body:
    {
        "session_id": "unique_session_id",
        "image": "base64_encoded_image_or_data_url",
        "bounding_boxes": [[x0, y0, x1, y1], ...],  # List of bounding boxes (ONLY in first request)
        "object_ids": [1, 2, ...]  # Optional: custom object IDs
    }
    
    Response:
    {
        "success": true,
        "session_id": "unique_session_id",
        "object_ids": [1, 2, ...],
        "bounding_boxes": [[x0, y0, x1, y1], ...],  # Bounding boxes extracted from segmentation masks
        "frame_shape": [height, width]
    }
    
    Note: Currently only one active session is supported at a time.
    Initializing a new session will reset the tracking state.
    """
    global predictor, session_states
    
    if predictor is None:
        return jsonify({"error": "Predictor not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        session_id = data.get("session_id", f"session_{int(time.time())}")
        image_data = data.get("image")
        bounding_boxes = data.get("bounding_boxes", [])
        object_ids = data.get("object_ids", None)
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        if not bounding_boxes:
            return jsonify({"error": "No bounding boxes provided"}), 400
        
        # Decode image
        frame = decode_image(image_data)
        height, width = frame.shape[:2]
        
        with predictor_lock:
            # Check if this session already exists - if so, reset it first
            if session_id in session_states:
                logger.warning(f"Session {session_id} already exists. Resetting predictor state for new initialization.")
            
            # Initialize predictor with first frame
            # This resets the predictor state completely (via _init_state())
            # This means only one active session is supported at a time
            predictor.load_first_frame(frame)
            
            # Initialize objects with bounding boxes
            # The bounding boxes define what to track in subsequent frames
            initialized_obj_ids = []
            mask_logits_list = []
            
            for idx, bbox in enumerate(bounding_boxes):
                if len(bbox) != 4:
                    continue
                
                # Use provided object ID or generate one
                obj_id = object_ids[idx] if object_ids and idx < len(object_ids) else (idx + 1)
                
                # Add prompt with bounding box on frame 0
                # This tells SAM2 what objects to track in subsequent frames
                # Each object has a unique obj_id, so clear_old_points=True is safe for all objects
                # (it only clears points for that specific object, not all objects)
                _, obj_ids, mask_logits = predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=obj_id,
                    bbox=bbox,
                    clear_old_points=True,  # Safe for all objects since each has unique obj_id
                    normalize_coords=True,
                )
                
                initialized_obj_ids.extend(list(obj_ids))
                if mask_logits is not None:
                    if mask_logits.dim() == 2:
                        mask_logits = mask_logits.unsqueeze(0)
                    mask_logits_list.append(mask_logits)
            
            # Store session state with original object IDs for consistent ID mapping
            session_states[session_id] = {
                "initialized": True,
                "frame_count": 0,
                "initial_frame_shape": [height, width],
                "original_object_ids": initialized_obj_ids.copy(),  # Store original IDs for consistent mapping
            }
        
        # Extract bounding boxes from masks for response
        result_bboxes = []
        combined_mask_logits = None
        if mask_logits_list:
            combined_mask_logits = torch.cat(mask_logits_list, dim=0)
            mask_probs = format_masks(combined_mask_logits)
            
            # Extract bounding boxes using efficient batched function
            extracted_bboxes = extract_bboxes_from_masks(mask_probs, threshold=0.5)
            
            for idx, obj_id in enumerate(initialized_obj_ids):
                if idx < len(extracted_bboxes):
                    bbox = extracted_bboxes[idx]
                    if bbox:
                        result_bboxes.append(bbox)
                    else:
                        # Fallback to input bbox if mask extraction fails
                        if idx < len(bounding_boxes):
                            result_bboxes.append(bounding_boxes[idx])
                else:
                    # Fallback to input bbox if index out of range
                    if idx < len(bounding_boxes):
                        result_bboxes.append(bounding_boxes[idx])
        
        # Save analyzed image if enabled
        if args.save_images:
            # Check if this is an interactive session (session_id contains "interactive")
            is_interactive = "interactive" in session_id.lower()
            save_analyzed_image(
                frame=frame,
                session_id=session_id,
                frame_index=0,
                obj_ids=initialized_obj_ids,
                mask_logits=combined_mask_logits,
                bboxes=result_bboxes if result_bboxes else bounding_boxes,
                output_dir=args.output_dir,
                threshold=args.mask_threshold,
                alpha=args.alpha,
                is_interactive=is_interactive,
            )
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "object_ids": initialized_obj_ids,
            "bounding_boxes": result_bboxes,
            "frame_shape": [height, width]
        })
    
    except Exception as e:
        logger.error(f"Error initializing session: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/track", methods=["POST"])
def track_frame():
    global args
    """
    Track objects in a new frame from the video stream.
    
    This endpoint:
    1. Receives a new frame from the mobile client (no bounding boxes needed!)
    2. Uses SAM2's internal memory to track the objects defined during /initialize
    3. Returns updated bounding boxes and object IDs for the tracked objects
    
    The predictor maintains temporal memory across frames, so it can track objects
    even if they move, change appearance, or are temporarily occluded.
    
    Request body:
    {
        "session_id": "unique_session_id",
        "image": "base64_encoded_image_or_data_url"  # No bounding boxes needed!
    }
    
    Response:
    {
        "success": true,
        "session_id": "unique_session_id",
        "object_ids": [1, 2, ...],  # Same object IDs from initialization
        "bounding_boxes": [[x0, y0, x1, y1], ...],  # Updated bounding boxes from segmentation
        "frame_index": 1  # Increments with each tracked frame
    }
    """
    global predictor, session_states
    
    if predictor is None:
        return jsonify({"error": "Predictor not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        session_id = data.get("session_id")
        image_data = data.get("image")
        
        if not session_id:
            return jsonify({"error": "No session_id provided"}), 400
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Check if session exists
        if session_id not in session_states:
            return jsonify({"error": f"Session {session_id} not found. Please initialize first."}), 404
        
        # Decode image
        frame = decode_image(image_data)
        
        # Get original object IDs from session state
        session_state = session_states[session_id]
        original_object_ids = session_state.get("original_object_ids", [])
        
        with predictor_lock:
            # Track objects in the new frame
            # The predictor maintains internal state (memory) from previous frames
            # and uses it to track the objects defined during initialization
            tracked_obj_ids, mask_logits = predictor.track(frame)
            
            # Update session state
            session_states[session_id]["frame_count"] += 1
            frame_index = session_states[session_id]["frame_count"]
        
        # Map tracked object IDs back to original IDs to maintain consistency
        # Create a mapping from tracked IDs to their indices in the mask_logits
        tracked_obj_ids_list = list(tracked_obj_ids)
        tracked_id_to_index = {obj_id: idx for idx, obj_id in enumerate(tracked_obj_ids_list)}
        
        # Extract bounding boxes from masks
        extracted_bboxes = []
        if mask_logits is not None:
            mask_probs = format_masks(mask_logits)
            # Extract bounding boxes using efficient batched function
            extracted_bboxes = extract_bboxes_from_masks(mask_probs, threshold=0.5)
        
        # Build result with consistent object IDs
        # For each original object ID, find its bounding box if it's still tracked
        result_bboxes = []
        result_object_ids = []
        
        for original_obj_id in original_object_ids:
            if original_obj_id in tracked_id_to_index:
                # Object is still tracked - get its bounding box
                idx = tracked_id_to_index[original_obj_id]
                if idx < len(extracted_bboxes) and extracted_bboxes[idx] is not None:
                    result_bboxes.append(extracted_bboxes[idx])
                    result_object_ids.append(original_obj_id)
                else:
                    # Object tracked but no valid bounding box - return None
                    result_bboxes.append(None)
                    result_object_ids.append(original_obj_id)
            else:
                # Object is lost - return None for bounding box but keep the ID
                result_bboxes.append(None)
                result_object_ids.append(original_obj_id)
        
        # For visualization, we need to create a mask_logits tensor that matches original IDs
        # This is tricky because mask_logits only has masks for tracked objects
        # We'll create a mapping for visualization purposes
        visualization_mask_logits = None
        visualization_obj_ids = []
        if mask_logits is not None:
            # For visualization, only include objects that are actually tracked
            visualization_obj_ids = tracked_obj_ids_list
            visualization_mask_logits = mask_logits
        else:
            visualization_obj_ids = []
            visualization_mask_logits = None
        
        # Save analyzed image if enabled
        if args.save_images:
            # Check if this is an interactive session (session_id contains "interactive")
            is_interactive = "interactive" in session_id.lower()
            save_analyzed_image(
                frame=frame,
                session_id=session_id,
                frame_index=frame_index,
                obj_ids=visualization_obj_ids,  # Use tracked IDs for visualization
                mask_logits=visualization_mask_logits,
                bboxes=[bbox for bbox in result_bboxes if bbox is not None],  # Only non-None bboxes for visualization
                output_dir=args.output_dir,
                threshold=args.mask_threshold,
                alpha=args.alpha,
                is_interactive=is_interactive,
            )
        
        # Convert None to null for JSON (Flask handles this automatically, but be explicit)
        # Filter out None values or keep them as null - we'll keep them as null to maintain ID consistency
        response_bboxes = [bbox if bbox is not None else None for bbox in result_bboxes]
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "object_ids": result_object_ids,  # Always return original IDs in order
            "bounding_boxes": response_bboxes,  # null for lost objects, bbox for tracked objects
            "frame_index": frame_index
        })
    
    except Exception as e:
        logger.error(f"Error tracking frame: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset_session():
    """
    Reset a tracking session.
    
    Request body:
    {
        "session_id": "unique_session_id"
    }
    """
    global predictor, session_states
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "No session_id provided"}), 400
        
        with predictor_lock:
            if predictor is not None:
                predictor.reset_state()
            
            if session_id in session_states:
                del session_states[session_id]
        
        return jsonify({
            "success": True,
            "message": f"Session {session_id} reset"
        })
    
    except Exception as e:
        logger.error(f"Error resetting session: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/sessions", methods=["GET"])
def list_sessions():
    """List all active sessions."""
    return jsonify({
        "sessions": list(session_states.keys()),
        "count": len(session_states)
    })


def initialize_predictor(args):
    """Initialize the SAM2 predictor."""
    global predictor
    
    logger.info("Initializing SAM2 predictor...")
    predictor = build_sam2_camera_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
        vos_optimized=args.vos_optimized,
    )
    logger.info("SAM2 predictor initialized successfully")


def main():
    global args
    args = parse_args()
    
    # Create output directory if saving images
    if args.save_images:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Image saving enabled. Output directory: {args.output_dir}")
    
    # Initialize predictor
    initialize_predictor(args)
    
    # Start Flask server
    logger.info(f"Starting SAM2 Mobile Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()

