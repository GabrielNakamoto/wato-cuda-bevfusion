#!/bin/bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."

ASSET_DIR="${DUALHEAD_ASSET_DIR:-qat/ckpt/dualhead}"
CHECKPOINT="${DUALHEAD_CHECKPOINT:-${ASSET_DIR}/epoch_10.pth}"
MODEL_CONFIG="${DUALHEAD_MODEL_CONFIG:-${ASSET_DIR}/model_config.yaml}"
TRAIN_CONFIG="${DUALHEAD_TRAIN_CONFIG:-${ASSET_DIR}/train_config.yaml}"

MODEL_REPO="${MODEL_REPO:-GabrielNakamoto/wato-cuda-bevfusion}"
MODEL_TAG="${MODEL_TAG:-dualhead-v1}"

MODEL_ASSET="${MODEL_ASSET:-epoch_10.pth}"
MODEL_CONFIG_ASSET="${MODEL_CONFIG_ASSET:-model_config.yaml}"
TRAIN_CONFIG_ASSET="${TRAIN_CONFIG_ASSET:-train_config.yaml}"

MODEL_SHA256="${MODEL_SHA256:-4ffefa802b5ccab4af289c0d578269ad52ef924b74c007159e407870318f3f7e}"
MODEL_CONFIG_SHA256="${MODEL_CONFIG_SHA256:-5395e93b60893ab818cc6cd6504f51486b64dfa04ad778d5bfb1265be1cf5efc}"
TRAIN_CONFIG_SHA256="${TRAIN_CONFIG_SHA256:-05e62a5e285134aab702cf4b84b3813aeeb65dbe9c078c69133e83952682a741}"

BASE_URL="https://github.com/${MODEL_REPO}/releases/download/${MODEL_TAG}"

check_file() {
    local path="$1"
    local sha="$2"
    [ -f "$path" ] || return 1
    echo "${sha}  ${path}" | sha256sum -c - >/dev/null 2>&1
}

fetch_file() {
    local label="$1"
    local asset="$2"
    local path="$3"
    local sha="$4"
    local url="${BASE_URL}/${asset}"
    local part="${path}.part"

    if check_file "$path" "$sha"; then
        echo "${label} already present and verified: $path"
        return 0
    fi

    mkdir -p "$(dirname "$path")"
    rm -f "$part"
    echo "Downloading ${label}: $url"
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        wget -q --show-progress --tries=5 --timeout=30 --retry-connrefused \
            --header="Authorization: Bearer ${GITHUB_TOKEN}" \
            -O "$part" "$url"
    else
        wget -q --show-progress --tries=5 --timeout=30 --retry-connrefused \
            -O "$part" "$url"
    fi
    echo "${sha}  ${part}" | sha256sum -c -
    mv "$part" "$path"
}

cleanup() {
    rm -f "${CHECKPOINT}.part" "${MODEL_CONFIG}.part" "${TRAIN_CONFIG}.part"
}
trap cleanup EXIT

fetch_file "dual-head checkpoint" "$MODEL_ASSET" "$CHECKPOINT" "$MODEL_SHA256"
fetch_file "model config" "$MODEL_CONFIG_ASSET" "$MODEL_CONFIG" "$MODEL_CONFIG_SHA256"
fetch_file "train config" "$TRAIN_CONFIG_ASSET" "$TRAIN_CONFIG" "$TRAIN_CONFIG_SHA256"

trap - EXIT
echo "Dual-head assets ready:"
echo "  checkpoint: $CHECKPOINT"
echo "  model cfg:  $MODEL_CONFIG"
echo "  train cfg:  $TRAIN_CONFIG"
