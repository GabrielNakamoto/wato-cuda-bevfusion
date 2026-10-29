import argparse
import os
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs

import lean.funcs as funcs
import lean.quantize as quantize
from ptq import load_model, fuse_conv_bn


def main():
    quantize.initialize()
    parser = argparse.ArgumentParser(description="Prepare BEVFusion segmentation checkpoint (fuse BN/ReLU)")
    parser.add_argument(
        "--config",
        metavar="FILE",
        default="bevfusion/configs/nuscenes/seg/fusion-bev256d2-lss.yaml",
    )
    parser.add_argument(
        "--ckpt",
        metavar="FILE",
        default="bevfusion/pretrained/bevfusion-seg.pth",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        default="qat/ckpt/bevfusion_ptq.pth",
    )
    args = parser.parse_args()

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    model = load_model(cfg, checkpoint_path=args.ckpt)
    model.encoders.lidar.backbone = funcs.layer_fusion_bn_relu(model.encoders.lidar.backbone)
    model = fuse_conv_bn(model)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model, args.out)
    print(f"Saved fused segmentation model to {args.out}")


if __name__ == "__main__":
    main()
