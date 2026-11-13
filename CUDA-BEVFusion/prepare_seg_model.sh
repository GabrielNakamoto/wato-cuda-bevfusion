#!/bin/bash
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
export PYTHONPATH="$(pwd)/bevfusion:${PYTHONPATH:-}"

export DEBUG_MODEL="${DEBUG_MODEL:-dualhead}"
export DEBUG_PRECISION="${DEBUG_PRECISION:-fp16}"
MODEL_DIR="model/${DEBUG_MODEL}"
ASSET_DIR="${DUALHEAD_ASSET_DIR:-qat/ckpt/dualhead}"
CHECKPOINT="${DUALHEAD_CHECKPOINT:-${ASSET_DIR}/epoch_10.pth}"
MODEL_CONFIG="${DUALHEAD_MODEL_CONFIG:-${ASSET_DIR}/model_config.yaml}"
TRAIN_CONFIG="${DUALHEAD_TRAIN_CONFIG:-${ASSET_DIR}/train_config.yaml}"
FUSED_CHECKPOINT="qat/ckpt/dualhead_ptq.pth"

echo "=========================================="
echo " CUDA-BEVFusion Dual-Head Model Setup"
echo "=========================================="
echo "model:       $DEBUG_MODEL"
echo "checkpoint:  $CHECKPOINT"
echo "model cfg:   $MODEL_CONFIG"
echo "train cfg:   $TRAIN_CONFIG"
echo ""

# 1) Build mmdet3d (bevfusion) package if not installed
if ! python3 -c "import mmdet3d.ops" >/dev/null 2>&1; then
    echo "[1/6] Building bevfusion Python package..."
    ( cd bevfusion && MAX_JOBS=1 FORCE_CUDA=1 python3 setup.py develop )
    rm -rf bevfusion/build
else
    echo "[1/6] mmdet3d already built."
fi

# 2) Fetch and validate the custom checkpoint and configs
echo "[2/6] Checking custom dual-head checkpoint..."
bash scripts/fetch_dualhead_model.sh
if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing checkpoint: $CHECKPOINT"
    exit 1
fi
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Missing model config: $MODEL_CONFIG"
    exit 1
fi
if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "Missing train config: $TRAIN_CONFIG"
    exit 1
fi

# 3) Download example-data if needed
echo "[3/6] Checking example-data..."
if [ ! -f example-data/example-data.pth ]; then
    wget -q --show-progress -c -O example-data.zip \
        "https://nvidia.box.com/shared/static/g8vxxes3xj1288teyo4og87rn99brdf8"
    unzip -q -o example-data.zip
    rm -f example-data.zip
fi

# 4) Fuse Conv+BN+ReLU into the dual-head checkpoint
echo "[4/6] Preparing fused dual-head checkpoint ($FUSED_CHECKPOINT)..."
mkdir -p qat/ckpt
rm -f "$FUSED_CHECKPOINT"
python3 qat/ptq_map.py \
    --config "$MODEL_CONFIG" \
    --ckpt   "$CHECKPOINT" \
    --out    "$FUSED_CHECKPOINT"

# 5) Export both head graphs and the shared trunk
echo "[5/6] Exporting ONNX graphs (FP16)..."
rm -rf qat/onnx_fp16
python3 qat/export-camera.py --ckpt "$FUSED_CHECKPOINT" --fp16
python3 qat/export-scn.py --ckpt "$FUSED_CHECKPOINT" --save qat/onnx_fp16/lidar.backbone.onnx
python3 qat/export-transfuser.py --ckpt "$FUSED_CHECKPOINT" --fp16
python3 qat/export-segmap.py --ckpt "$FUSED_CHECKPOINT" --fp16

mkdir -p "$MODEL_DIR"
cp -p qat/onnx_fp16/camera.backbone.onnx    "$MODEL_DIR/"
cp -p qat/onnx_fp16/camera.vtransform.onnx  "$MODEL_DIR/"
cp -p qat/onnx_fp16/fuser.onnx              "$MODEL_DIR/"
cp -p qat/onnx_fp16/head.seg.onnx           "$MODEL_DIR/"
cp -p qat/onnx_fp16/head.bbox.onnx          "$MODEL_DIR/"
cp -p qat/onnx_fp16/lidar.backbone.xyz.onnx "$MODEL_DIR/"
cp -f "$MODEL_CONFIG" "$MODEL_DIR/model_config.yaml"
cp -f "$TRAIN_CONFIG" "$MODEL_DIR/train_config.yaml"
cp -f "$MODEL_CONFIG" "$MODEL_DIR/default.yaml"

# 6) Build TensorRT engines (shared trunk + both heads)
echo "[6/6] Building TensorRT engines..."
DEBUG_MODEL="$DEBUG_MODEL" bash tool/build_trt_engine.sh

echo "Done! Run bash tool/run.sh to build and execute C++ inference."
