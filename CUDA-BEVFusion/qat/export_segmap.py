import argparse, os, importlib, torch
import torch.nn as nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from torch.nn import functional as F

import lean.quantize as quantize
from export_transfuser import SubclassFuser

def grid_transform(transform, x:torch.Tensor):
    # --- optional pre-upscale ---
    if transform.prescale_factor != 1:
        x = F.interpolate(
            x,
            scale_factor=transform.prescale_factor,
            mode='bilinear',
            align_corners=False
        )

    # compute output grid dimensions from scopes
    # (opset-safe replacement for grid_sample since ONNX opset < 16
    #  doesn't expose GridSample)
    output_size = []
    for (_, _, _), (omin, omax, ostep) in zip(
        transform.input_scope, transform.output_scope
    ):
        n = int(torch.arange(omin + ostep / 2, omax, ostep).numel())
        output_size.append(n)

    # Bilinear resize to the target output size.
    # For regular grids this is equivalent to grid_sample with
    # align_corners=False while remaining fully ONNX-exportable.
    x = F.interpolate(
        x,
        size=tuple(output_size),
        mode='bilinear',
        align_corners=False
    )
    return x

class SubclassHeadSeg(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent


    @staticmethod
    def head_forward(self, x:torch.Tensor):
        x = grid_transform(self.transform, x)
        x = self.classifier(x)
        return torch.sigmoid(x)

    def forward(self, x):
        for type_, head in self.parent.heads.items():
            if type_ == "map":
                return self.head_forward(head, x)
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
