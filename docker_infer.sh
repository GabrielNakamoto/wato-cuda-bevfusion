#!/bin/bash
set -euo pipefail

docker build -t bevfusion -f CUDA-BEVFusion/bevfusion/docker/Dockerfile .
docker build -t cuda-bevfusion -f CUDA-BEVFusion/docker/Dockerfile .
docker run --gpus all --rm \
  -v ~/wato-cuda-bevfusion:/home/wato-cuda-bevfusion \
  cuda-bevfusion:latest \
  bash -c "bash prepare_seg_model.sh && bash tool/run.sh"
