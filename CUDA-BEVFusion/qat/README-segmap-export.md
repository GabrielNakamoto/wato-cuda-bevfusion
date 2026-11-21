# Exporting a fine-tuned segmentation head

`qat/export-segmap.py` is written for the stock NVIDIA PTQ models: it expects a
`torch.load(ckpt).module` wrapper, needs `pytorch_quantization`, and its
hand-written `head_forward` reproduces the *stock* head — which silently drops
our zero-initialised hi-res residual branch. Exporting our fine-tuned checkpoint
with it would produce an engine whose logits disagree with the checkpoint.

Use `qat/export-segmap-finetuned.py` instead. It exports the head module itself
(`transform -> [hi-res residual] -> classifier -> sigmoid -> threshold -> uint8`)
from a plain mmdet3d config + checkpoint, and validates probability parity
against onnxruntime.

## Usage

```bash
python qat/export-segmap-finetuned.py \
  --cfg  configs/dualhead/finetune_eve_map_v3.yaml \
  --ckpt /path/to/epoch_40.pth \
  --out  qat/onnx_fp16/head.seg.onnx \
  --input-shape 1,512,360,360 \
  --input-npy  qat/head_input.npy \
  --validate
```

`--input-shape` **must match the fuser engine's `middle` output**. The export is
static, so one engine per shape. Three are supported out of the box:

| shape | where it comes from |
|---|---|
| `1,512,128,128` | stock CUDA-BEVFusion fuser (nuScenes 0.6 m grid) |
| `1,512,180,180` | our training fork's fuser output (0.6 m, [-54, 54]) |
| `1,512,360,360` | the deployed node's geometry grid (`geometry_dim: [360, 360, 80]`, 0.3 m) |

Check the deployed engine before picking:

```bash
trtexec --loadEngine=model/seg/head.seg.fp16.engine --dumpLayerInfo | head -20
```

`qat/head_input.npy` (dumped once with `tools/dump_head_input.py` from the
training repo) is a real fuser output; validating on it gives ~3e-4 max
probability difference and **zero** threshold flips, where random noise gives a
handful of flips at the 0.5 boundary (pure interpolation rounding).

## Thresholds are baked into the graph

`export-segmap.py` (and this script by default) applies one threshold in the
ONNX graph, so changing it means re-exporting and rebuilding the engine. The
fine-tuned head's probabilities are compressed low, and a global 0.5 produces a
nearly empty map. Measured on a held-out frame (`oldbag__dataset_live__sample_0001`),
drivable IoU vs lanelet GT:

| export | drivable IoU | fill | divider IoU |
|---|---|---|---|
| `--threshold 0.5` (head default) | 0.003 | 0.1 % | 0.000 |
| `--threshold 0.30` | 0.180 | 3.5 % | 0.000 |
| `--threshold 0.20` | 0.371 | 9.2 % | 0.021 |
| `--threshold 0.10` | 0.397 | 12.2 % | 0.036 |
| `--thresholds 0.05,0.02,0.5,0.02,0.5,0.10` | 0.455 | 13.4 % | 0.036 |
| `--thresholds 0.03,0.02,0.5,0.02,0.5,0.08` | **0.465** | 13.6 % | **0.044** |

**The IoU numbers above are misleading on their own.** The valid mask is only
13.6 % of the grid, and GT drivable is 46.5 % of that, so a prediction that
paints *every* valid cell scores IoU 0.465. That is exactly what the low-threshold
exports do: at `--threshold 0.03` drivable fills 99.4 % of valid cells, i.e. the
model is flooding, not discriminating. Thresholds here only trade "empty map"
(0.5, fill 0.7 % of valid) against "everything is drivable" (0.03, fill 99.4 %).

Practical consequence: do not ship a low-threshold export. `0.30` (IoU 0.18,
fill ~26 % of valid) is the least-bad demo setting, and the underlying problem is
calibration/perception quality, not the threshold. The real fix is a per-frame
quantile threshold (needs a float probabilities output) or more training.

Long term these should become ROS parameters with a probabilities output, rather
than being compiled into the engine.

## Building the engine

```bash
trtexec --fp16 --stronglyTyped \
  --onnx=model/seg/head.seg.onnx \
  --saveEngine=model/seg/head.seg.fp16.engine \
  --minShapes=middle:1x512xHxW --optShapes=middle:1x512xHxW --maxShapes=middle:1x512xHxW
```

TRT 11 note: `--stronglyTyped` is what keeps the FP16 bindings HALF;
`bevfusion_core.cpp` also requests `kSTRONGLY_TYPED` when compiling at runtime.
Without it TRT 11 upcasts a weakly typed graph to FP32 and the CUDA kernels
abort on `dtype(BindingOutput) == HALF`.

## What still needs the TRT machine

The ONNX files are portable; the engines are not. On the vehicle (or any box
with TensorRT 11 + the CUDA-BEVFusion build):

1. drop `head.seg.onnx` into `model/seg/`
2. build the engine with the command above, using the shape printed by
   `--dumpLayerInfo` on the existing `fuser` engine
3. `./tool/run.sh` and check the seg output topic is non-empty
