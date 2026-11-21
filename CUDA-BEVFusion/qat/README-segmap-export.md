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
