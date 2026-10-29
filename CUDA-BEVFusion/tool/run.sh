#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# configure the environment
. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

set -e

# Regenerate protobuf headers to match system libprotobuf-dev (3.6.1)
bash src/onnx/make_pb.sh

mkdir -p build

cd build
cmake ..
make -j2

cd ..

./build/bevfusion $DEBUG_DATA $DEBUG_MODEL $DEBUG_PRECISION
