FROM bevfusion:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1

# Install system dependencies, protobuf-compiler, cuDNN 8.6, and TensorRT 8.5.3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libprotobuf-dev protobuf-compiler cmake unzip git-lfs \
        libcudnn8=8.6.0.163-1+cuda11.8 \
        libcudnn8-dev=8.6.0.163-1+cuda11.8 \
        libnvinfer8=8.5.3-1+cuda11.8 \
        libnvinfer-bin=8.5.3-1+cuda11.8 \
        libnvinfer-dev=8.5.3-1+cuda11.8 \
        libnvinfer-plugin8=8.5.3-1+cuda11.8 \
        libnvinfer-plugin-dev=8.5.3-1+cuda11.8 \
        libnvonnxparsers8=8.5.3-1+cuda11.8 \
        libnvonnxparsers-dev=8.5.3-1+cuda11.8 \
        libnvparsers8=8.5.3-1+cuda11.8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python ONNX/quantization requirements
COPY CUDA-BEVFusion/tool/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

WORKDIR /home/wato-cuda-bevfusion/CUDA-BEVFusion
