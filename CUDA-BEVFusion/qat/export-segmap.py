import argparse
import os

import torch
import torch.nn as nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

import lean.quantize as quantize
from export-transfuser import SubclassFuser


class SubclassHeadSeg(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def head_forward():
        pass

    def forward(self, x):
        for type_, head in self.parent.heads.items():
            if type_ == "map":
                return head(x)
        raise ValueError("Model does not have a map (segmentation) head")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export segmentation head to ONNX")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="qat/ckpt/bevfusion_ptq.pth",
        help="Pretrained model checkpoint",
    )
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    model = torch.load(args.ckpt).module

    suffix = "int8"
    if args.fp16:
        suffix = "fp16"
        quantize.disable_quantization(model).apply()

    save_root = f"qat/onnx_{suffix}"
    os.makedirs(save_root, exist_ok=True)

    model.eval()
    fuser = SubclassFuser(model).cuda()
    seghead = SubclassHeadSeg(model).cuda()

    TensorQuantizer.use_fb_fake_quant = True
    with torch.no_grad():
        camera_features = torch.randn(1, 80, 180, 180).cuda()
        lidar_features = torch.randn(1, 256, 180, 180).cuda()

        fuser_onnx_path = f"{save_root}/fuser.onnx"
        torch.onnx.export(
            fuser,
            [camera_features, lidar_features],
            fuser_onnx_path,
            opset_version=13,
            input_names=["camera", "lidar"],
            output_names=["middle"],
        )
        print(f"🚀 Fuser export completed. ONNX saved as {fuser_onnx_path}")

        seghead_onnx_path = f"{save_root}/head.seg.onnx"
        head_input = torch.randn(1, 512, 180, 180).cuda()
        torch.onnx.export(
            seghead,
            head_input,
            seghead_onnx_path,
            opset_version=13,
            input_names=["middle"],
            output_names=["segmap"],
        )
        print(
            f"🚀 Segmentation head export completed. ONNX saved as {seghead_onnx_path}"
        )
