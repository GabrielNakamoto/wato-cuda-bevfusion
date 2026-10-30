#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

docker build -t bevfusion -f "$REPO_DIR/CUDA-BEVFusion/bevfusion/docker/Dockerfile" "$REPO_DIR"
docker build -t cuda-bevfusion -f "$REPO_DIR/CUDA-BEVFusion/docker/Dockerfile" "$REPO_DIR"
docker run --gpus all --rm \
  -v "$REPO_DIR:/home/wato-cuda-bevfusion" \
  -w /home/wato-cuda-bevfusion/CUDA-BEVFusion \
  cuda-bevfusion:latest \
  bash -c "bash prepare_seg_model.sh && bash tool/run.sh"
