#!/usr/bin/env bash
# Quick helper to run SAM2 video testing on the bundled default juggling video.
# Override CONFIG, CHECKPOINT, VIDEO, OUTPUT, FRAME_IDX, BOX, or POINTS via env vars.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-configs/sam2.1/sam2.1_hiera_t.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/sam2.1_hiera_tiny.pt}"
VIDEO="${VIDEO:-demo/data/gallery/05_default_juggle.mp4}"
OUTPUT="${OUTPUT:-outputs/juggle_demo}"
FRAME_IDX="${FRAME_IDX:-0}"
# Default bounding box roughly covering the juggler; adjust as needed.
BOX="${BOX:-600 250 1150 900}"
# Example positive click near the juggler's torso. Format: "x y label".
POINTS="${POINTS:-900 650 1}"

cmd=(
  python "${REPO_ROOT}/tools/video_test.py"
  --config "${CONFIG}"
  --checkpoint "${REPO_ROOT}/${CHECKPOINT}"
  --video-path "${REPO_ROOT}/${VIDEO}"
  --output-dir "${REPO_ROOT}/${OUTPUT}"
  --frame-idx "${FRAME_IDX}"
  --box ${BOX}
  --point ${POINTS}
  --device cuda
  --save-masks
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"

