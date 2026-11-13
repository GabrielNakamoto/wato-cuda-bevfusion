#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# configure the environment
. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

# resnet50/resnet50-int8/swint-tiny/seg
base=model/$DEBUG_MODEL

# all engines are built fp16 only
trtexec_fp16_flags="--fp16"

function get_onnx_number_io(){
    model=$1
    if [ ! -f "$model" ]; then
        echo The model [$model] not exists.
        return
    fi
    number_of_input=`python3 -c "import onnx;m=onnx.load('$model');print(len(m.graph.input), end='')"`
    number_of_output=`python3 -c "import onnx;m=onnx.load('$model');print(len(m.graph.output), end='')"`
}

function compile_trt_model(){
    name=$1
    precision_flags=$2
    number_of_input=$3
    number_of_output=$4
    extra_flags=${5:-}
    output_format=${6:-fp16}
    result_save_directory=$base/build
    onnx=$base/$name.onnx

    if [ -f "${result_save_directory}/$name.plan" ] && \
       [ ! "$onnx" -nt "${result_save_directory}/$name.plan" ]; then
        echo Model ${result_save_directory}/$name.plan already build 🙋🙋🙋.
        return
    fi

    get_onnx_number_io $onnx

    input_flags="--inputIOFormats="
    output_flags=""
    for i in $(seq 1 $number_of_input); do
        input_flags+=fp16:chw,
    done

    if [ "$output_format" != "none" ]; then
        output_flags="--outputIOFormats="
        for i in $(seq 1 $number_of_output); do
            output_flags+="${output_format}:chw,"
        done
        output_flags=${output_flags%?}
    fi

    input_flags=${input_flags%?}

    cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} ${output_flags} ${extra_flags} \
        --saveEngine=${result_save_directory}/$name.plan \
        --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
        --dumpProfile --separateProfileRun \
        --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"

    mkdir -p $result_save_directory
    echo Building the model: ${result_save_directory}/$name.plan, this will take several minutes. Wait a moment 🤗🤗🤗~.
    trtexec $cmd > ${result_save_directory}/$name.log 2>&1
    if [ $? != 0 ]; then
        echo 😥 Failed to build model ${result_save_directory}/$name.plan.
        echo You can check the error message by ${result_save_directory}/$name.log 
        exit 1
    fi
}

# The bbox head uses the custom LayerNorm plugin.
bbox_extra_flags=""
if [ -f "$base/head.bbox.onnx" ]; then
    mkdir -p build
    echo "Building custom LayerNorm plugin for bbox head..."
    (cd build && cmake .. >/tmp/custom_layernorm_cmake.log 2>&1 && make -j2 custom_layernorm >/tmp/custom_layernorm_make.log 2>&1)
    bbox_extra_flags="--plugins=$(pwd)/build/libcustom_layernorm.so"
fi

# fp16 only (no int8 path)
compile_trt_model "camera.backbone" "$trtexec_fp16_flags" 2 2
compile_trt_model "fuser" "$trtexec_fp16_flags" 2 1
compile_trt_model "camera.vtransform" "$trtexec_fp16_flags" 1 1

if [ -f "$base/head.seg.onnx" ]; then
    compile_trt_model "head.seg" "$trtexec_fp16_flags" 1 1
fi
if [ -f "$base/head.bbox.onnx" ]; then
    compile_trt_model "head.bbox" "$trtexec_fp16_flags" 1 6 "$bbox_extra_flags"
fi
