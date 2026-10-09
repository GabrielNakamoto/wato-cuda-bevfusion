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
# 0b) TensorRT + cuDNN
#     `trtexec` (shipped by the `libnvinfer-bin` deb) is what actually builds
#     the .plan engines.  The base bevfusion image (CUDA 11.3) does not ship
#     TensorRT, so install it from NVIDIA's CUDA apt repository.
# ---------------------------------------------------------------------------
install_tensorrt() {
    # Covers both the deb install and a manual tarball that put trtexec on PATH.
    if command -v trtexec >/dev/null 2>&1 || [ -x /usr/src/tensorrt/bin/trtexec ]; then
        echo "    TensorRT already installed; skipping."
        return 0
    fi

    # NVIDIA's apt repo is keyed by distro (ubuntu2004) and arch (x86_64).
    local distro arch
    distro="ubuntu$(. /etc/os-release && echo "${VERSION_ID//./}")"
    case "$(dpkg --print-architecture)" in
        amd64) arch="x86_64" ;;
        arm64) arch="aarch64" ;;
        *)     arch="$(dpkg --print-architecture)" ;;
    esac

    # Add the CUDA apt repository only if it is not configured yet.  Checking
    # the source files (rather than apt-cache) avoids creating a conflicting
    # duplicate entry on images that already ship a cuda.list (nvidia/cuda:*).
    if ! grep -rqs 'developer.download.nvidia.com/compute/cuda/repos' \
            /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
        echo "    Adding NVIDIA CUDA apt repository (${distro}/${arch})..."
        local keyring
        keyring="$(mktemp --suffix=.deb)"
        wget -q -O "$keyring" \
            "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${arch}/cuda-keyring_1.1-1_all.deb"
        dpkg -i "$keyring"
        rm -f "$keyring"
    fi

    apt-get update -qq

    # TensorRT 8.5.3 / cuDNN 8.6.0 are the newest builds published for CUDA 11.x
    # and satisfy the README's TensorRT >= 8.5 requirement.  Override with
    # TENSORRT_APT_VERSION / CUDNN_APT_VERSION for a different CUDA version.
    local trt_ver="${TENSORRT_APT_VERSION:-8.5.3-1+cuda11.8}"
    local cudnn_ver="${CUDNN_APT_VERSION:-8.6.0.163-1+cuda11.8}"
    local pkgs=(libcudnn8 libcudnn8-dev
                libnvinfer8 libnvinfer-bin libnvinfer-dev
                libnvinfer-plugin8 libnvinfer-plugin-dev
                libnvonnxparsers8 libnvonnxparsers-dev libnvparsers8)
    local pinned=() p
    for p in "${pkgs[@]}"; do
        case "$p" in
            libcudnn8*) pinned+=("$p=$cudnn_ver") ;;
            *)          pinned+=("$p=$trt_ver")   ;;
        esac
    done

    echo "    Installing TensorRT ${trt_ver} and cuDNN ${cudnn_ver}..."
    if ! apt-get install -y --no-install-recommends "${pinned[@]}"; then
        echo "    Pinned versions unavailable; installing the latest from the repo..."
        apt-get install -y --no-install-recommends "${pkgs[@]}"
    fi
}
install_tensorrt

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
# Build-time / CPU-only stop.
# PTQ calibration, ONNX export and TensorRT engine building all require a
# CUDA device, which `docker build` cannot provide.  Set PREPARE_SKIP_GPU=1
# while building the image so these steps run on the first GPU-enabled
# container start (see docker/entrypoint.sh).
# ---------------------------------------------------------------------------
if [ "${PREPARE_SKIP_GPU:-0}" = "1" ]; then
    echo ""
    echo "PREPARE_SKIP_GPU=1 -> stopping before GPU-only steps (PTQ/ONNX/engines)."
    echo "Re-run prepare_seg_model.sh with a GPU (e.g. docker run --gpus all) to finish."
    exit 0
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

# Helper: detect TensorRT across common container / system layouts
auto_detect_tensorrt() {
    local trt_bin="" trt_lib="" trt_inc=""

    # 1) Look for a self-contained tree layout: <root>/{bin,lib,include}
    for root in /usr/src/tensorrt /usr/local/tensorrt /opt/tensorrt \
                /usr/local/TensorRT /opt/TensorRT; do
        if { [ -f "$root/bin/trtexec" ] || [ -f "$root/bin/trtexec.exe" ]; } \
           && [ -d "$root/lib" ] && [ -d "$root/include" ]; then
            trt_bin="$root/bin"
            trt_lib="$root/lib"
            trt_inc="$root/include"
            break
        fi
    done

    # 2) Locate the trtexec binary.  The libnvinfer-bin deb puts it in
    #    /usr/src/tensorrt/bin without a matching lib/include tree.
    if [ -z "$trt_bin" ]; then
        local trtexec_path=""
        if command -v trtexec >/dev/null 2>&1; then
            trtexec_path="$(command -v trtexec)"
        else
            for p in /usr/src/tensorrt/bin/trtexec \
                     /usr/bin/trtexec /usr/local/bin/trtexec; do
                [ -f "$p" ] && trtexec_path="$p" && break
            done
        fi

        if [ -n "$trtexec_path" ]; then
            trt_bin="$(dirname "$trtexec_path")"
            # Prefer a self-contained tree next to the binary (../lib, ../include)
            local inferred_root
            inferred_root="$(realpath -m "$trt_bin/..")"
            if [ -f "$inferred_root/lib/libnvinfer.so" ] \
               || [ -f "$inferred_root/lib/libnvinfer.so.8" ]; then
                trt_lib="$inferred_root/lib"
                trt_inc="$inferred_root/include"
            fi
        fi
    fi

    # 3) System-package (deb) layout: libs and headers live in multiarch dirs
    if [ -z "$trt_lib" ]; then
        for libdir in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib; do
            if [ -f "$libdir/libnvinfer.so" ] || [ -f "$libdir/libnvinfer.so.8" ]; then
                trt_lib="$libdir"
                break
            fi
        done
    fi
    if [ -z "$trt_inc" ]; then
        for incdir in /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include; do
            if [ -f "$incdir/NvInfer.h" ]; then
                trt_inc="$incdir"
                break
            fi
        done
    fi

    # 4) Last resort: the dynamic linker and existing environment variables
    if [ -z "$trt_lib" ]; then
        local ldconf_line
        ldconf_line="$(ldconfig -p 2>/dev/null | grep -m1 'libnvinfer.so ' | awk '{print $NF}')"
        if [ -n "$ldconf_line" ] && [ -f "$ldconf_line" ]; then
            trt_lib="$(dirname "$ldconf_line")"
        fi
    fi
    if [ -z "$trt_lib" ] && [ -n "${TENSORRT_LIB:-}" ] && [ -d "$TENSORRT_LIB" ]; then
        trt_lib="$TENSORRT_LIB"
    fi
    if [ -z "$trt_inc" ] && [ -n "${TENSORRT_INCLUDE:-}" ] && [ -d "$TENSORRT_INCLUDE" ]; then
        trt_inc="$TENSORRT_INCLUDE"
    fi
    if [ -z "$trt_bin" ] && [ -n "${TENSORRT_BIN:-}" ] && [ -d "$TENSORRT_BIN" ]; then
        trt_bin="$TENSORRT_BIN"
    fi

    # Return results
    printf "%s\n%s\n%s\n" "$trt_bin" "$trt_lib" "$trt_inc"
}

# --- run detection ---
mapfile -t TRT_VARS < <(auto_detect_tensorrt)
TRT_BIN="${TRT_VARS[0]}"
TRT_LIB="${TRT_VARS[1]}"
TRT_INC="${TRT_VARS[2]}"

if [ -n "$TRT_BIN" ] && [ -f "$TRT_BIN/trtexec" ]; then
    sed -i "s|^export TensorRT_Lib=.*|export TensorRT_Lib=$TRT_LIB|" tool/environment.sh
    sed -i "s|^export TensorRT_Inc=.*|export TensorRT_Inc=$TRT_INC|" tool/environment.sh
    sed -i "s|^export TensorRT_Bin=.*|export TensorRT_Bin=$TRT_BIN|" tool/environment.sh
    echo "    Auto-configured TensorRT:"
    echo "      TensorRT_Lib = $TRT_LIB"
    echo "      TensorRT_Inc = $TRT_INC"
    echo "      TensorRT_Bin = $TRT_BIN"
else
    echo "    ERROR: Could not auto-detect TensorRT installation."
    echo "    Please edit tool/environment.sh manually and set TensorRT_Lib, TensorRT_Inc, and TensorRT_Bin."
    echo "    Common container locations:"
    echo "      - /usr/src/tensorrt/{lib,include,bin}"
    echo "      - /usr/lib/x86_64-linux-gnu + /usr/include/x86_64-linux-gnu + /usr/bin"
    exit 1
fi

# Auto-detect CUDNN if still placeholder
if grep -q '/path/to/cudnn' tool/environment.sh; then
    CUDNN_LIB=""
    for d in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib; do
        if [ -f "$d/libcudnn.so" ] || [ -f "$d/libcudnn.so.8" ]; then
            CUDNN_LIB="$d"
            break
        fi
    done
    if [ -n "$CUDNN_LIB" ]; then
        sed -i "s|^export CUDNN_Lib=.*|export CUDNN_Lib=$CUDNN_LIB|" tool/environment.sh
        echo "    Auto-configured CUDNN: CUDNN_Lib = $CUDNN_LIB"
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
    echo "ERROR: tool/environment.sh failed to configure."
    echo "       TensorRT_Bin=$TensorRT_Bin  TensorRT_Lib=$TensorRT_Lib  TensorRT_Inc=$TensorRT_Inc"
    echo "       Please fix these paths in tool/environment.sh manually."
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
