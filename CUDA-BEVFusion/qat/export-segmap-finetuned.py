#!/usr/bin/env python3
"""Export a fine-tuned segmentation head to ONNX for the CUDA-BEVFusion runtime.

Drop-in replacement for `head.seg.onnx` produced by `qat/export-segmap.py`.

Why not use that script directly:
  * it expects a PTQ checkpoint (`torch.load(...).module`) and pytorch_quantization,
    because it is built for the quantised NVIDIA models;
  * its hand-written `head_forward` reproduces the *stock* head, which silently
    drops our zero-initialised hi-res residual branch -- the exported engine
    would then produce different logits than the checkpoint we trained.

So we export the module itself: transform -> [hi-res residual] -> classifier ->
sigmoid -> threshold -> uint8, which is exactly the contract the C++ side wants
(`head-segmap.cu`: binding "middle" in, binding "segmap" out, UINT8, NCHW, square).

  export-segmap-finetuned.py --cfg configs/dualhead/finetune_eve_map_v3.yaml \
      --ckpt runs/run-.../epoch_40.pth --out qat/onnx_fp16/head.seg.onnx --validate
"""
import argparse
import pathlib
import sys

import torch


class SegExport(torch.nn.Module):
    """head -> sigmoid probabilities -> thresholded uint8 NCHW (transposed)."""

    def __init__(self, head, threshold):
        super().__init__()
        self.head = head
        # scalar or per-class vector, broadcast over the channel dim
        t = torch.as_tensor(threshold, dtype=torch.float32)
        if t.numel() > 1:
            t = t.view(1, -1, 1, 1)
        self.register_buffer("thr", t)

    def forward(self, x):
        p = self.head(x)                       # eval mode returns sigmoid probs
        out = torch.where(p > self.thr, torch.full_like(p, 100.0), torch.zeros_like(p))
        return out.transpose(2, 3).to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="qat/onnx_fp16/head.seg.onnx")
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: the head's own threshold")
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated per-class thresholds in head class order "
                         "(drivable,ped_crossing,walkway,stop_line,carpark_area,divider)")
    ap.add_argument("--opset", type=int, default=17, help="16+ required: GridSample")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--input-shape", default="1,512,128,128",
                    help="static input shape; must match the fuser engine's output "
                         "(stock CUDA-BEVFusion fuser: 128x128, our fork's: 180x180)")
    ap.add_argument("--input-npy", default=None,
                    help="real head input (1,C,128,128) to validate with")
    args = ap.parse_args()

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    cfg = Config.fromfile(args.cfg)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    head = model.heads["map"].to(args.device).eval()

    if args.thresholds:
        thr = [float(v) for v in args.thresholds.split(",")]
        assert len(thr) == len(head.classes), f"{len(thr)} thresholds for {len(head.classes)} classes"
    else:
        thr = args.threshold if args.threshold is not None else float(getattr(head, "threshold", 0.5))
    in_ch = int(head.in_channels)
    n_cls = len(head.classes)
    has_hi = getattr(head, "hi", None) is not None
    print(f"head: in_ch={in_ch} classes={n_cls} threshold={thr} hi_res={has_hi}")

    m = SegExport(head, thr).eval()
    shape = tuple(int(v) for v in args.input_shape.split(","))
    if args.input_npy:
        import numpy as np
        x = torch.from_numpy(np.load(args.input_npy)).float().to(args.device)
        if tuple(x.shape) != shape:
            import torch.nn.functional as Fn
            x = Fn.interpolate(x, size=shape[2:], mode="bilinear", align_corners=False)
        print(f"input from {args.input_npy} -> {tuple(x.shape)}")
    else:
        x = torch.randn(*shape, device=args.device)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        ref = m(x)
    try:
        # TorchScript exporter: opset 13, which is what the TRT pipeline expects
        torch.onnx.export(
            m, (x,), str(out_path),
            opset_version=args.opset,
            input_names=["middle"], output_names=["segmap"],
            do_constant_folding=True, dynamo=False,
        )
    except TypeError:
        # torch >= 2.9 dropped the dynamo flag path for some builds
        torch.onnx.export(
            m, (x,), str(out_path),
            opset_version=args.opset,
            input_names=["middle"], output_names=["segmap"],
            do_constant_folding=True,
        )
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)  "
          f"torch output {tuple(ref.shape)} {ref.dtype}")

    if args.validate:
        import numpy as np
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(onnx.load(str(out_path)))
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        got = sess.run(["segmap"], {"middle": x.detach().cpu().numpy()})[0]
        ref_np = ref.detach().cpu().numpy()
        flips = int((got != ref_np).sum())
        print(f"onnxruntime: {got.shape} {got.dtype}   threshold flips {flips}/{got.size} "
              f"({100.0 * flips / got.size:.4f}%)")

        # probabilities are the real check: uint8 ties at the threshold flip on
        # the last float bit, which is meaningless but would look like a failure
        class Probs(torch.nn.Module):
            def __init__(self, head):
                super().__init__()
                self.head = head

            def forward(self, t):
                return self.head(t)

        tmp = out_path.with_suffix(".probs.onnx")
        torch.onnx.export(Probs(head).eval(), (x,), str(tmp), opset_version=args.opset,
                          input_names=["middle"], output_names=["probs"],
                          do_constant_folding=True, dynamo=False)
        s2 = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
        p_ort = s2.run(["probs"], {"middle": x.detach().cpu().numpy()})[0]
        with torch.no_grad():
            p_ref = head(x).detach().cpu().numpy()
        d = np.abs(p_ort - p_ref)
        print(f"  probability parity: max |delta| {d.max():.3e}  mean {d.mean():.3e}")
        tmp.unlink(missing_ok=True)
        if d.max() > 1e-2:
            print("  FAIL: graph does not match the PyTorch head")
            sys.exit(1)
        print("  OK: graph matches the PyTorch head")


if __name__ == "__main__":
    main()
