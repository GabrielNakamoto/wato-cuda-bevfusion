# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# CUDA-BEVFusion segmentation inference image.
#
# Build context MUST be the repository root (the directory that contains
# CUDA-BEVFusion/, dependencies/ and libraries/).  Initialise the submodules
# first:
#
#     git submodule update --init --recursive
#     docker build -t bevfusion-seg -f CUDA-BEVFusion/docker/Dockerfile .
#
# The resulting image bakes in:
#   * CUDA 11.3 + the bevfusion conda environment
#   * TensorRT 8.5.3 + cuDNN 8.6 (installed from NVIDIA's CUDA apt repository)
#   * the repository source, including submodules
#   * the bevfusion/mmdet3d CUDA extensions (compiled with FORCE_CUDA=1)
#   * the pretrained checkpoint, example-data and nuScenes-mini plus its
#     generated nuscenes_infos*.pkl files
#
# What cannot be baked: PTQ calibration, ONNX export and TensorRT engine
# building all call `.cuda()`.  `docker build` has no GPU access, so these run
# on the first GPU-enabled container start instead:
#
#     docker run --gpus all -it --rm bevfusion-seg
#
# If you already built the environment image from
# CUDA-BEVFusion/bevfusion/docker/Dockerfile, pass it as the base to skip the
# ~18 GB environment install (recommended):
#
#     docker build --build-arg BASE_IMAGE=bevfusion:latest \
#                  -t bevfusion-seg -f CUDA-BEVFusion/docker/Dockerfile .

ARG BASE_IMAGE=nvidia/cuda:11.3.1-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/envs/bevfusion/bin:/opt/conda/bin:${PATH}

# The image build has no GPU, but mmdet3d's setup.py falls back to CPU-only
# extensions when torch.cuda.is_available() is False.  Force the CUDA
# extensions so the runtime can use them.
ENV FORCE_CUDA=1

# 1) Python/CUDA environment.  No-op when the base image is bevfusion:latest.
COPY CUDA-BEVFusion/docker/install_env.sh /tmp/install_env.sh
RUN bash /tmp/install_env.sh && rm -f /tmp/install_env.sh

# 2) Repository source (the .dockerignore keeps generated artifacts out).
WORKDIR /home/wato-cuda-bevfusion
COPY . /home/wato-cuda-bevfusion

# 3) CPU-only half of the segmentation preparation: system + Python deps,
#    mmdet3d CUDA extensions, checkpoint/dataset downloads, info pkl files.
#    Cleanup runs in the same layer so deleted files never ship in the image.
WORKDIR /home/wato-cuda-bevfusion/CUDA-BEVFusion
RUN PREPARE_SKIP_GPU=1 bash prepare_seg_model.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache /tmp/* && \
    conda clean -afy

# 4) Finish PTQ/ONNX/engines on the first GPU start, then run the command.
RUN chmod +x docker/entrypoint.sh
ENTRYPOINT ["/home/wato-cuda-bevfusion/CUDA-BEVFusion/docker/entrypoint.sh"]
CMD ["/bin/bash"]
