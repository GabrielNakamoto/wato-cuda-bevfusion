#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Support the rootless Docker daemon used in this dev session.
# The login environment may point DOCKER_HOST at the unavailable root socket.
if [ -S /tmp/run/docker.sock ]; then
  if ! docker info >/dev/null 2>&1; then
    export DOCKER_HOST=unix:///tmp/run/docker.sock
  fi
fi

docker build -t bevfusion -f "$REPO_DIR/CUDA-BEVFusion/bevfusion/docker/Dockerfile" "$REPO_DIR"
docker build -t cuda-bevfusion -f "$REPO_DIR/CUDA-BEVFusion/docker/Dockerfile" "$REPO_DIR"

docker run --gpus all --rm --shm-size=8g \
  -e DEBUG_MODEL=dualhead \
  -e DEBUG_PRECISION=fp16 \
  -e DEBUG_DATA=example-data \
  -e USE_Python=OFF \
  -v "$REPO_DIR:/home/wato-cuda-bevfusion" \
  -w /home/wato-cuda-bevfusion/CUDA-BEVFusion \
  cuda-bevfusion:latest \
  bash -c "bash prepare_seg_model.sh && bash tool/run.sh"
