import argparse
import importlib.util
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

import lean.quantize as quantize

_spec = importlib.util.spec_from_file_location(
    "export_transfuser", os.path.join(os.path.dirname(__file__), "export-transfuser.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SubclassFuser = _mod.SubclassFuser


def _resample_matrix(in_scope, out_scope, device):
    """Bilinear resampling weights (n_out x n_in) matching grid_sample(align_corners=False)."""
    imin, imax, istep = in_scope
    omin, omax, ostep = out_scope
    n_in = int(round((imax - imin) / istep))
    n_out = int(round((omax - omin) / ostep))
    xo = torch.arange(n_out, dtype=torch.float64) * ostep + omin + ostep / 2
    g = (xo - imin) / (imax - imin) * 2 - 1            # normalized, as grid_sample wants
    pos = ((g + 1) * n_in - 1) / 2                     # -> input pixel coords
    i0 = torch.floor(pos)
    frac = pos - i0
    i1 = i0 + 1
    i0 = i0.clamp(0, n_in - 1).long()
    i1 = i1.clamp(0, n_in - 1).long()
    W = torch.zeros(n_out, n_in, dtype=torch.float32)
    rows = torch.arange(n_out)
    W[rows, i0] += (1 - frac).float()
    W[rows, i1] += frac.float()
    return W.to(device)


def grid_transform(transform, x: torch.Tensor):
    """Resample the BEV feature onto the output scope (crop, not stretch)."""
    if transform.prescale_factor != 1:
        x = F.interpolate(
            x,
            scale_factor=transform.prescale_factor,
            mode="bilinear",
            align_corners=False,
        )

    Wy = _resample_matrix(transform.input_scope[0], transform.output_scope[0], x.device)
    Wx = _resample_matrix(transform.input_scope[1], transform.output_scope[1], x.device)

    B, C, H, Wd = x.shape
    y = torch.matmul(Wy, x.reshape(B * C, H, Wd))       # (BC, n_out_y, W)
    z = torch.matmul(y, Wx.t())                         # (BC, n_out_y, n_out_x)
    return z.reshape(B, C, Wy.shape[0], Wx.shape[0])


class SubclassHeadSeg(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @staticmethod
    def head_forward(self, x: torch.Tensor):
        x = grid_transform(self.transform, x)
        # avoid sigmoid by thresholding raw logits in custom cuda kernel
        return self.classifier(x).transpose(2,3)

    def forward(self, x):
        for type_, head in self.parent.heads.items():
            if type_ == "map":
                return self.head_forward(head, x)
        raise ValueError("Model does not have a map (segmentation) head")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fuser and segmentation head to ONNX")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="qat/ckpt/dualhead_ptq.pth",
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
        print(f"🚀 Segmentation head export completed. ONNX saved as {seghead_onnx_path}")
