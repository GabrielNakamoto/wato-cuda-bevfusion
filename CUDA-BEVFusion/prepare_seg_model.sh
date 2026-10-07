#!/bin/bash
set -euo pipefail

# Always operate from the CUDA-BEVFusion directory
cd "$(dirname "$(realpath "$0")")"

echo "=========================================="
echo " CUDA-BEVFusion Segmentation Model Prep "
echo "=========================================="

# ---------------------------------------------------------------------------
# 0) System dependencies
# ---------------------------------------------------------------------------
echo "[0/8] Installing system dependencies..."
if ! dpkg -s libprotobuf-dev &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq libprotobuf-dev wget unzip build-essential git
fi

# ---------------------------------------------------------------------------
# 1) Python dependencies
# ---------------------------------------------------------------------------
echo "[1/8] Installing Python dependencies..."
pip install -q -r tool/requirements.txt

# ---------------------------------------------------------------------------
# 2) Build / verify the bevfusion (mmdet3d) package
# ---------------------------------------------------------------------------
echo "[2/8] Setting up bevfusion Python package..."
if ! python -c "import mmdet3d" >/dev/null 2>&1; then
    ( cd bevfusion && python setup.py develop )
fi

# ---------------------------------------------------------------------------
# 3) Pretrained seg checkpoint (idempotent)
# ---------------------------------------------------------------------------
echo "[3/8] Downloading pretrained segmentation checkpoint..."
mkdir -p bevfusion/pretrained
if [ ! -f bevfusion/pretrained/bevfusion-seg.pth ]; then
    ( cd bevfusion && bash tools/download_pretrained.sh )
else
    echo "    bevfusion-seg.pth already exists, skipping."
fi

# ---------------------------------------------------------------------------
# 4) example-data for ONNX export
#    The NVBox link is a direct download; the archive's top-level dir is
#    example-data/.
# ---------------------------------------------------------------------------
echo "[4/8] Downloading example data..."
if [ ! -f example-data/example-data.pth ]; then
    wget -q --show-progress -c -O example-data.zip \
        "https://nvidia.box.com/shared/static/g8vxxes3xj1288teyo4og87rn99brdf8"
    unzip -q -o example-data.zip
    rm -f example-data.zip
else
    echo "    example-data already exists, skipping."
fi

# ---------------------------------------------------------------------------
# 5) nuScenes mini (free, no auth).
#    NOTE: the tarball only contains raw data; the info .pkl files are
#    generated in step 6.
# ---------------------------------------------------------------------------
echo "[5/8] Preparing nuScenes mini dataset..."
DATA_DIR="$(realpath -m data/nuscenes)"
if [ ! -d "$DATA_DIR/v1.0-mini" ]; then
    mkdir -p "$DATA_DIR"
    wget -q --show-progress -c -O "$DATA_DIR/v1.0-mini.tgz" \
        https://www.nuscenes.org/data/v1.0-mini.tgz
    tar -xzf "$DATA_DIR/v1.0-mini.tgz" -C "$DATA_DIR"
else
    echo "    nuScenes mini already exists, skipping."
fi

# ---------------------------------------------------------------------------
# 6) Generate nuscenes_infos_{train,val}.pkl + nuscenes_dbinfos_train.pkl.
#    Use absolute paths so ptq.py can be run from any CWD.
# ---------------------------------------------------------------------------
echo "[6/8] Generating nuScenes info files..."
if [ ! -f "$DATA_DIR/nuscenes_infos_train.pkl" ]; then
    ( cd bevfusion && python tools/create_data.py nuscenes \
        --root-path "$DATA_DIR" \
        --out-dir   "$DATA_DIR" \
        --extra-tag nuscenes \
        --version   v1.0-mini )
else
    echo "    nuscenes_infos already exist, skipping."
fi

# ---------------------------------------------------------------------------
# 7) PTQ calibration -> qat/ckpt/bevfusion_ptq.pth
# ---------------------------------------------------------------------------
echo "[7/8] Running PTQ calibration for segmentation..."
if [ ! -f "qat/ckpt/bevfusion_ptq.pth" ]; then
    python3 qat/ptq.py \
        --config bevfusion/configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
        --ckpt bevfusion/pretrained/bevfusion-seg.pth \
        --calibrate_batch 300
else
    echo "    PTQ checkpoint already exists, skipping."
fi

# ---------------------------------------------------------------------------
# 8) Export ONNX models (FP16)
# ---------------------------------------------------------------------------
echo "[8/8] Exporting ONNX models for TensorRT..."

# 8a) camera backbone + vtransform
if [ ! -f "qat/onnx_fp16/camera.backbone.onnx" ]; then
    python3 qat/export_camera.py --ckpt qat/ckpt/bevfusion_ptq.pth --fp16
else
    echo "    camera.backbone.onnx already exists, skipping."
fi

# 8b) fuser + segmentation head
if [ ! -f "qat/onnx_fp16/head.seg.onnx" ]; then
    python3 qat/export_segmap.py --ckpt qat/ckpt/bevfusion_ptq.pth --fp16
else
    echo "    head.seg.onnx already exists, skipping."
fi

# 8c) lidar backbone (SCN)
if [ ! -f "qat/onnx_fp16/lidar.backbone.xyz.onnx" ]; then
    python3 qat/export_scn.py --ckpt qat/ckpt/bevfusion_ptq.pth --save qat/onnx_fp16/lidar.backbone.onnx
else
    echo "    lidar.backbone.xyz.onnx already exists, skipping."
fi

# ---------------------------------------------------------------------------
# 9) Organize the model directory for TensorRT engine building
# ---------------------------------------------------------------------------
MODEL_NAME="seg"
MODEL_DIR="model/$MODEL_NAME"

echo ""
echo "Organizing model files into $MODEL_DIR/..."
mkdir -p "$MODEL_DIR"

cp -f qat/onnx_fp16/camera.backbone.onnx       "$MODEL_DIR/"
cp -f qat/onnx_fp16/camera.vtransform.onnx     "$MODEL_DIR/"
cp -f qat/onnx_fp16/fuser.onnx                 "$MODEL_DIR/"
cp -f qat/onnx_fp16/head.seg.onnx              "$MODEL_DIR/"
cp -f qat/onnx_fp16/lidar.backbone.xyz.onnx    "$MODEL_DIR/"
cp -f bevfusion/configs/nuscenes/seg/fusion-bev256d2-lss.yaml "$MODEL_DIR/default.yaml"

echo ""
echo "ONNX model files ready in $MODEL_DIR/:"
ls -lh "$MODEL_DIR/"*.onnx

# ---------------------------------------------------------------------------
# 10) Update tool/environment.sh for seg model
# ---------------------------------------------------------------------------
echo ""
echo "Configuring environment for seg model..."

# Auto-detect TensorRT if paths are still placeholders
if grep -q '/path/to/tensorrt' tool/environment.sh; then
    FOUND_TRT=""
    for trt_path in /usr/src/tensorrt /usr/local/tensorrt /opt/tensorrt; do
        if [ -f "$trt_path/bin/trtexec" ]; then
            FOUND_TRT="$trt_path"
            break
        fi
    done
    # Fallback: check if trtexec is already on PATH
    if [ -z "$FOUND_TRT" ] && command -v trtexec >/dev/null 2>&1; then
        TRTEXEC_PATH="$(command -v trtexec)"
        FOUND_TRT="$(realpath -m "$(dirname "$TRTEXEC_PATH")/../")"
    fi

    if [ -n "$FOUND_TRT" ]; then
        sed -i "s|/path/to/tensorrt|$FOUND_TRT|g" tool/environment.sh
        echo "    Auto-configured TensorRT path: $FOUND_TRT"
    else
        echo "    WARNING: Could not auto-detect TensorRT. Please set TensorRT_* in tool/environment.sh manually."
    fi
fi

# Update model / precision / data for seg mode
sed -i 's/^export DEBUG_MODEL=.*/export DEBUG_MODEL=seg/'   tool/environment.sh
sed -i 's/^export DEBUG_PRECISION=.*/export DEBUG_PRECISION=fp16/' tool/environment.sh
sed -i 's/^export DEBUG_DATA=.*/export DEBUG_DATA=example-data/'     tool/environment.sh

echo "    Updated tool/environment.sh:"
grep -E 'DEBUG_MODEL|DEBUG_PRECISION|DEBUG_DATA' tool/environment.sh || true

# ---------------------------------------------------------------------------
# 11) Build TensorRT engines
# ---------------------------------------------------------------------------
echo ""
echo "Building TensorRT engines..."

# Source the environment so that trtexec is available
set +u
# shellcheck source=/dev/null
. tool/environment.sh
set -u

if [ "${ConfigurationStatus:-Failed}" != "Success" ]; then
    echo "ERROR: tool/environment.sh failed to configure. Please set TensorRT/CUDA paths manually."
    exit 1
fi

bash tool/build_trt_engine.sh

# ---------------------------------------------------------------------------
# 12) Done
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Segmentation model preparation complete!"
echo "=========================================="
echo ""
echo "TensorRT plan files:"
ls -lh "$MODEL_DIR/build/"*.plan 2>/dev/null || true
echo ""
echo "Next steps (inside the bevfusion-dev container):"
echo "  1. source tool/environment.sh"
echo "  2. bash tool/run.sh"
echo ""
