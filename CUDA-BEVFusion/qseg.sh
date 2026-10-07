#!/bin/bash
set -euo pipefail

# Always operate from the CUDA-BEVFusion directory
cd "$(dirname "$(realpath "$0")")"

# 1) Python deps for quantization + ONNX export
pip install -r tool/requirements.txt

# 2) Build the bevfusion (mmdet3d) package once; compiles the CUDA ops
if ! python -c "import mmdet3d" >/dev/null 2>&1; then
    ( cd bevfusion && python setup.py develop )
fi

# 3) Pretrained seg checkpoint (idempotent)
mkdir -p bevfusion/pretrained
if [ ! -f bevfusion/pretrained/bevfusion-seg.pth ]; then
    ( cd bevfusion && bash tools/download_pretrained.sh )
fi

# 4) example-data for ONNX export (qat/export-*.py loads "example-data/example-data.pth").
#    The NVBox link is a direct download; the archive's top-level dir is example-data/.
if [ ! -f example-data/example-data.pth ]; then
    wget -c -O example-data.zip \
        "https://nvidia.box.com/shared/static/g8vxxes3xj1288teyo4og87rn99brdf8"
    python3 -c "import zipfile; zipfile.ZipFile('example-data.zip').extractall('.')"
fi

# 5) nuScenes mini (free, no auth).  NOTE: the tarball only contains raw data;
#    the info .pkl files are generated in step 6.
DATA_DIR="$(realpath -m data/nuscenes)"
if [ ! -d "$DATA_DIR/v1.0-mini" ]; then
    mkdir -p "$DATA_DIR"
    wget -c -O "$DATA_DIR/v1.0-mini.tgz" https://www.nuscenes.org/data/v1.0-mini.tgz
    tar -xzf "$DATA_DIR/v1.0-mini.tgz" -C "$DATA_DIR"
fi

# 6) Generate nuscenes_infos_{train,val}.pkl + nuscenes_dbinfos_train.pkl.
#    Use absolute paths so ptq.py can be run from any CWD.
if [ ! -f "$DATA_DIR/nuscenes_infos_train.pkl" ]; then
    ( cd bevfusion && python tools/create_data.py nuscenes \
        --root-path "$DATA_DIR" \
        --out-dir   "$DATA_DIR" \
        --extra-tag nuscenes \
        --version   v1.0-mini )
fi

# 7) PTQ calibration -> qat/ckpt/bevfusion_ptq.pth
python3 qat/ptq.py \
    --config bevfusion/configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
    --ckpt bevfusion/pretrained/bevfusion-seg.pth

python3 qat/export_camera.py --fp16
python3 qat/export_segmap.py --fp16
python3 qat/export_scn.py
