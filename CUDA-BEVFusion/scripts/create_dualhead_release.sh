#!/bin/bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."

ASSET_DIR="${DUALHEAD_ASSET_DIR:-qat/ckpt/dualhead}"
MODEL_REPO="${MODEL_REPO:-GabrielNakamoto/wato-cuda-bevfusion}"
MODEL_TAG="${MODEL_TAG:-dualhead-v1}"

if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI (gh) is required to upload release assets."
    exit 1
fi

TARGET="$(git rev-parse HEAD)"
ASSETS=(
    "${ASSET_DIR}/epoch_10.pth"
    "${ASSET_DIR}/model_config.yaml"
    "${ASSET_DIR}/train_config.yaml"
)

for f in "${ASSETS[@]}"; do
    [ -f "$f" ] || { echo "Missing asset: $f"; exit 1; }
done

if gh release view "$MODEL_TAG" --repo "$MODEL_REPO" >/dev/null 2>&1; then
    echo "Release $MODEL_TAG exists; uploading/replacing assets..."
    gh release upload "$MODEL_TAG" "${ASSETS[@]}" --repo "$MODEL_REPO" --clobber
else
    echo "Creating release $MODEL_TAG at $TARGET..."
    gh release create "$MODEL_TAG" "${ASSETS[@]}" \
        --repo "$MODEL_REPO" \
        --target "$TARGET" \
        --title "BEVFusion dual-head models v1" \
        --notes "Dual-head epoch-10 correction checkpoint and configs."
fi
