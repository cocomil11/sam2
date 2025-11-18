#!/bin/bash
# Script to create a minimal conda environment for webcam_stream_server.py

set -e

ENV_NAME="webcam_stream_server"
ENV_FILE="$(dirname "$0")/webcam_stream_server_environment.yml"

echo "Creating conda environment '$ENV_NAME' from $ENV_FILE..."
conda env create -f "$ENV_FILE"

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To remove the environment later, run:"
echo "  conda env remove -n $ENV_NAME"


