#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Container entrypoint.
#
# The image build bakes the CPU-only half of prepare_seg_model.sh (env, source,
# checkpoint/dataset downloads).  PTQ calibration, ONNX export and TensorRT
# engine building need a CUDA device and are therefore finished here, on the
# first start of a GPU-enabled container.  Everything is idempotent, so
# subsequent starts are no-ops.
set -euo pipefail

cd /home/wato-cuda-bevfusion/CUDA-BEVFusion

if [ ! -f model/seg/build/head.seg.plan ]; then
    echo "[entrypoint] First run: completing PTQ + ONNX export + TensorRT engines..."
    bash prepare_seg_model.sh
fi

exec "$@"
