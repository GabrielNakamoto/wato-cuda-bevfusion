#!/bin/bash

cd bevfusion
bash tool/download_pretrained.sh
cd ..

python3 qat/ptq.py \
  --config bevfusion/configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  --ckpt bevfusion/pretrained/bevfusion-seg.pth
