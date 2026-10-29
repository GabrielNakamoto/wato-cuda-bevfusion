#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

export TensorRT_Lib=/usr/lib/x86_64-linux-gnu
export TensorRT_Inc=/usr/include/x86_64-linux-gnu
export TensorRT_Bin=/usr/src/tensorrt/bin

export CUDA_HOME=/usr/local/cuda
export CUDA_Lib=$CUDA_HOME/lib64
export CUDA_Inc=$CUDA_HOME/include
export CUDA_Bin=$CUDA_HOME/bin

export CUDNN_Lib=/usr/lib/x86_64-linux-gnu

# For CUDA-11.x:    SPCONV_CUDA_VERSION=11.4
# For CUDA-12.x:    SPCONV_CUDA_VERSION=12.8
export SPCONV_CUDA_VERSION=11.4

# resnet50/resnet50int8/swint/seg
export DEBUG_MODEL=${DEBUG_MODEL:-seg}

# fp16/int8
export DEBUG_PRECISION=${DEBUG_PRECISION:-fp16}
export DEBUG_DATA=${DEBUG_DATA:-example-data}
export USE_Python=${USE_Python:-ON}

export ConfigurationStatus=Failed
if [ ! -f "${TensorRT_Bin}/trtexec" ]; then
    echo "Can not find ${TensorRT_Bin}/trtexec, there may be a mistake in the directory you configured."
    return
fi

if [ ! -f "${CUDA_Bin}/nvcc" ]; then
    echo "Can not find ${CUDA_Bin}/nvcc, there may be a mistake in the directory you configured."
    return
fi

echo "=========================================================="
echo "||  MODEL: $DEBUG_MODEL"
echo "||  PRECISION: $DEBUG_PRECISION"
echo "||  DATA: $DEBUG_DATA"
echo "||  USEPython: $USE_Python"
echo "||"
echo "||  TensorRT: $TensorRT_Lib"
echo "||  CUDA: $CUDA_HOME"
echo "||  CUDNN: $CUDNN_Lib"
echo "=========================================================="

BuildDirectory=`pwd`/build

if [ "$USE_Python" == "ON" ]; then
    export Python_Inc=`python3 -c "import sysconfig;print(sysconfig.get_path('include'))"`
    export Python_Lib=`python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"`
    export Python_Soname=`python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))"`
    echo Find Python_Inc: $Python_Inc
    echo Find Python_Lib: $Python_Lib
    echo Find Python_Soname: $Python_Soname
fi

export PATH=$TensorRT_Bin:$CUDA_Bin:$PATH
export LD_LIBRARY_PATH=$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BuildDirectory:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$BuildDirectory:${PYTHONPATH:-}
export ConfigurationStatus=Success

if [ -f "tool/cudasm.sh" ]; then
    echo "Try to get the current device SM"
    . "tool/cudasm.sh"
    echo "Current CUDA SM: $cudasm"
fi

export CUDASM=$cudasm

echo Configuration done!
