#!/usr/bin/env python3
"""
Simple CLI utility to run SAM 2 video inference on a local video file or a
directory of JPEG frames. This is a lightweight alternative to the notebook
examples and is handy for quick smoke tests.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


# A small, fixed color palette in BGR format for overlay visualization.
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
        description="Test SAM 2 video segmentation on a single video."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a SAM 2 config (e.g. configs/sam2.1/sam2.1_hiera_t.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the corresponding checkpoint file (*.pt).",
    )
    parser.add_argument("--video-path", required=True, help="Input MP4 or JPEG folder.")
    parser.add_argument(
        "--output-dir",
        default="outputs/video_test",
        help="Directory where visualizations and masks will be written.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for running the predictor (default: cuda).",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index where prompts are provided (default: 0).",
    )
    parser.add_argument(
        "--obj-id",
        type=int,
        default=1,
        help="Object id used internally to keep track of prompts (default: 1).",
    )
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Bounding box prompt (pixel coordinates, same resolution as the video).",
    )
    parser.add_argument(
        "--point",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "LABEL"),
        help="Point prompt as (x, y, label). label=1 for positive, 0 for negative."
        " Repeat flag to add more points.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to propagate (defaults to full video).",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for converting logits into binary masks.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Opacity of the colored overlay (0-1).",
    )
    parser.add_argument(
        "--vos-optimized",
        action="store_true",
        help="Enable VOS-optimized predictor (uses torch.compile).",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Keep preprocessed frames on CPU to save GPU memory.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Keep inference state tensors on CPU to reduce GPU memory footprint.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="If set, saves binary masks for each object/frame alongside the video.",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable torch.autocast even when running on CUDA.",
    )
    return parser.parse_args()


def read_video_frames(video_path: str) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    """Load video frames as BGR uint8 numpy arrays via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-2:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, fps, (width, height)


def read_jpeg_folder(folder_path: str) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    frame_files = [
        p
        for p in os.listdir(folder_path)
        if os.path.splitext(p)[-1].lower() in {".jpg", ".jpeg"}
    ]
    if not frame_files:
        raise RuntimeError(f"No JPEG frames found in {folder_path}")
    try:
        frame_files.sort(key=lambda name: int(os.path.splitext(name)[0]))
    except ValueError:
        frame_files.sort()

    frames: List[np.ndarray] = []
    width = height = 0
    for name in frame_files:
        img = cv2.imread(os.path.join(folder_path, name), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read frame {name} from {folder_path}")
        if height == 0:
            height, width = img.shape[:2]
        frames.append(img)
    fps = 30.0  # best effort when FPS info is unavailable
    return frames, fps, (width, height)


def load_visual_frames(video_path: str) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    if os.path.isdir(video_path):
        return read_jpeg_folder(video_path)
    return read_video_frames(video_path)


def build_points_and_labels(points_arg: Sequence[Sequence[float]] | None) -> Tuple[
    np.ndarray | None, np.ndarray | None
]:
    if not points_arg:
        return None, None
    pts = []
    lbls = []
    for entry in points_arg:
        if len(entry) != 3:
            raise ValueError("--point expects three values: x y label")
        x, y, label = entry
        pts.append([x, y])
        lbls.append(int(label))
    return np.asarray(pts, dtype=np.float32), np.asarray(lbls, dtype=np.int32)


def pick_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    repeats = (num_colors + len(PALETTE) - 1) // len(PALETTE)
    palette = (PALETTE * repeats)[:num_colors]
    return list(palette)


def apply_mask_overlay(
    frame_bgr: np.ndarray,
    masks: Sequence[np.ndarray],
    colors: Sequence[Tuple[int, int, int]],
    alpha: float,
) -> np.ndarray:
    if not masks:
        return frame_bgr

    overlay = frame_bgr.copy()
    mask_union = np.zeros(frame_bgr.shape[:2], dtype=bool)

    for mask, color in zip(masks, colors):
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if not mask.any():
            continue
        color_arr = np.array(color, dtype=np.uint8)
        mask_union |= mask
        overlay[mask] = (overlay[mask] * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, thickness=2)

    output = frame_bgr.copy()
    output[mask_union] = overlay[mask_union]
    return output


def resolve_device(device_str: str) -> torch.device:
    try:
        device = torch.device(device_str)
    except RuntimeError as exc:
        raise ValueError(f"Invalid --device value: {device_str}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        device = torch.device("cpu")
    return device


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "sam2_overlay.mp4"
    masks_dir = output_dir / "masks"
    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=str(device),
        vos_optimized=args.vos_optimized,
    )

    frames_bgr, fps, (width, height) = load_visual_frames(args.video_path)
    if args.frame_idx < 0 or args.frame_idx >= len(frames_bgr):
        raise ValueError(f"--frame-idx {args.frame_idx} is out of range for video with {len(frames_bgr)} frames")

    points, labels = build_points_and_labels(args.point)
    if points is None and args.box is None:
        raise ValueError("Please provide at least one --point or a --box prompt.")

    box = np.asarray(args.box, dtype=np.float32) if args.box is not None else None

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (not args.no_autocast and device.type == "cuda")
        else contextlib.nullcontext()
    )

    def logits_to_masks(tensor: torch.Tensor) -> List[np.ndarray]:
        probs = torch.sigmoid(tensor.squeeze(1))
        bin_masks = (probs >= args.mask_threshold).cpu().numpy().astype(bool)
        return [mask for mask in bin_masks]

    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    with torch.inference_mode():
        state = predictor.init_state(
            args.video_path,
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
        )
        with autocast_ctx:
            frame_idx, obj_ids, _ = predictor.add_new_points_or_box(
                state,
                frame_idx=args.frame_idx,
                obj_id=args.obj_id,
                points=points,
                labels=labels,
                box=box,
                normalize_coords=True,
            )
            print(f"Added prompts on frame {frame_idx}, tracking objects: {obj_ids}")

            colors = pick_colors(len(obj_ids))
            propagation_iter = predictor.propagate_in_video(
                state,
                start_frame_idx=args.frame_idx,
                max_frame_num_to_track=args.max_frames,
                reverse=False,
            )

            for frame_idx, obj_ids, masks_logits in propagation_iter:
                masks_bool = logits_to_masks(masks_logits)
                overlay_frame = apply_mask_overlay(
                    frames_bgr[frame_idx], masks_bool, colors, args.alpha
                )
                writer.write(overlay_frame)
                if args.save_masks:
                    for obj_id, mask in zip(obj_ids, masks_bool):
                        cv2.imwrite(
                            str(masks_dir / f"frame_{frame_idx:05d}_obj_{obj_id}.png"),
                            (mask.astype(np.uint8) * 255),
                        )

    writer.release()

    print(f"Overlay video saved to {overlay_path}")
    if args.save_masks:
        print(f"Binary masks saved under {masks_dir}")


if __name__ == "__main__":
    main()

