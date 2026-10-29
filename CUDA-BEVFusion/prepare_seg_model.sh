#!/bin/bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
export PYTHONPATH="$(pwd)/bevfusion:${PYTHONPATH:-}"

echo "=========================================="
echo " CUDA-BEVFusion Segmentation Model Setup "
echo "=========================================="

# 1) Build mmdet3d (bevfusion) package if not installed
if ! python3 -c "import mmdet3d.ops" >/dev/null 2>&1; then
    echo "[1/6] Building bevfusion Python package..."
    ( cd bevfusion && FORCE_CUDA=1 python3 setup.py develop )
    rm -rf bevfusion/build
else
    echo "[1/6] mmdet3d already built."
fi

# 2) Download pretrained segmentation checkpoint
echo "[2/6] Checking pretrained segmentation checkpoint..."
mkdir -p bevfusion/pretrained
if [ ! -f bevfusion/pretrained/bevfusion-seg.pth ]; then
    wget -q --show-progress -O bevfusion/pretrained/bevfusion-seg.pth \
        "https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9"
fi

# 3) Download example-data
echo "[3/6] Checking example-data..."
if [ ! -f example-data/example-data.pth ]; then
    wget -q --show-progress -c -O example-data.zip \
        "https://nvidia.box.com/shared/static/g8vxxes3xj1288teyo4og87rn99brdf8"
    unzip -q -o example-data.zip
    rm -f example-data.zip
fi

# 4) Fuse Conv+BN+ReLU into checkpoint (no dataset required for FP16)
echo "[4/6] Preparing fused checkpoint (qat/ckpt/bevfusion_ptq.pth)..."
if [ ! -f qat/ckpt/bevfusion_ptq.pth ]; then
    python3 qat/ptq_map.py \
        --config bevfusion/configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
        --ckpt   bevfusion/pretrained/bevfusion-seg.pth \
        --out    qat/ckpt/bevfusion_ptq.pth
fi

# 5) Export ONNX graphs (FP16)
echo "[5/6] Exporting ONNX graphs..."
if [ ! -f qat/onnx_fp16/camera.backbone.onnx ]; then
    python3 qat/export-camera.py --ckpt qat/ckpt/bevfusion_ptq.pth --fp16
fi
if [ ! -f qat/onnx_fp16/head.seg.onnx ]; then
    python3 qat/export-segmap.py --ckpt qat/ckpt/bevfusion_ptq.pth --fp16
fi
if [ ! -f qat/onnx_fp16/lidar.backbone.xyz.onnx ]; then
    python3 qat/export-scn.py --ckpt qat/ckpt/bevfusion_ptq.pth --save qat/onnx_fp16/lidar.backbone.onnx
fi

mkdir -p model/seg
cp -f qat/onnx_fp16/camera.backbone.onnx    model/seg/
cp -f qat/onnx_fp16/camera.vtransform.onnx  model/seg/
cp -f qat/onnx_fp16/fuser.onnx              model/seg/
cp -f qat/onnx_fp16/head.seg.onnx           model/seg/
cp -f qat/onnx_fp16/lidar.backbone.xyz.onnx model/seg/
cp -f bevfusion/configs/nuscenes/seg/fusion-bev256d2-lss.yaml model/seg/default.yaml

# 6) Build TensorRT engines
echo "[6/6] Building TensorRT engines..."
bash tool/build_trt_engine.sh

echo "Done! Run 'bash tool/run.sh' to build and execute C++ inference."
