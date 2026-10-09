# Containerised CUDA-BEVFusion segmentation model prep

This directory turns [`prepare_seg_model.sh`](../prepare_seg_model.sh) into a
buildable Docker image.

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile` | Builds the image (env, source, downloads, then defers GPU steps). |
| `install_env.sh` | Creates the CUDA 11.3 + conda `bevfusion` environment. No-op if the base image already has it. |
| `entrypoint.sh` | On the first GPU start, finishes PTQ / ONNX export / TensorRT engines, then runs the container command. |

## Why the script is split

`docker build` **cannot access the GPU**. The PTQ calibration, ONNX export and
TensorRT engine build steps all call `.cuda()`, so they are executed on the
first `docker run --gpus all` instead of at build time. Everything else --
system/Python dependencies, the mmdet3d CUDA extensions (compiled with
`FORCE_CUDA=1`), the pretrained checkpoint, `example-data` and the nuScenes
mini dataset + info files -- is baked into the image.

The script stays the single source of truth: it now supports
`PREPARE_SKIP_GPU=1` to stop before the GPU-only steps, and the entrypoint
re-runs it with a GPU to finish.

## Build

From the **repository root** (the directory containing `CUDA-BEVFusion/`,
`dependencies/` and `libraries/`):

```bash
git submodule update --init --recursive

# Fast path: reuse the environment image produced by
# CUDA-BEVFusion/bevfusion/docker/Dockerfile
docker build --build-arg BASE_IMAGE=bevfusion:latest \
             -t bevfusion-seg -f CUDA-BEVFusion/docker/Dockerfile .

# Fully self-contained (builds the conda environment from the CUDA base too)
docker build -t bevfusion-seg -f CUDA-BEVFusion/docker/Dockerfile .
```

The build downloads the checkpoint, example data and nuScenes mini, and
compiles the mmdet3d CUDA extensions, so expect it to take a while and to use
several GB of disk.

## Run

```bash
# First run finishes PTQ + ONNX export + TensorRT engine building, then opens
# a shell.
docker run --gpus all -it --rm bevfusion-seg

# Once the engines exist, run inference directly:
docker run --gpus all -it --rm bevfusion-seg bash tool/run.sh
```

Subsequent containers created from the same image re-run the (idempotent) prep
once, because the generated `.plan` files live in the container's writable
layer, not in the image.

## Image size and cleanup

The preparation removes its intermediates as it goes:

| Removed | Approx. size |
| --- | --- |
| nuScenes `v1.0-mini.tgz` after extraction | ~3.9 GB |
| Unused pretrained checkpoints (keeps only `bevfusion-seg.pth`) | ~0.7 GB |
| `bevfusion/build` object files | ~0.4 GB |
| pip cache (installs use `--no-cache-dir`) | ~0.35 GB |
| apt lists, `/tmp`, pip cache and conda pkg cache (Docker cleanup layer) | up to ~8 GB |

The big one is the conda package cache (`/opt/conda/pkgs`, ~8 GB). It lives in
the **base** image, and a later Docker layer cannot free space a lower layer
already occupies, so ``bevfusion-seg`` built `FROM bevfusion:latest` cannot
reclaim it. To get the smallest image, rebuild the base environment with the
cleaned `CUDA-BEVFusion/bevfusion/docker/Dockerfile` first:

```bash
docker build -t bevfusion:latest -f CUDA-BEVFusion/bevfusion/docker/Dockerfile .
docker build --build-arg BASE_IMAGE=bevfusion:latest \
             -t bevfusion-seg -f CUDA-BEVFusion/docker/Dockerfile .
```

Building fully self-contained (`FROM nvidia/cuda:...`) also avoids the base
bloat because `install_env.sh` cleans the conda cache inside its own layer.

## Known limitation

Running the C++ inference binary (`tool/run.sh`) also needs an **x86_64**
`libspconv.so`, but in this repository only the `aarch64` builds are real
binaries; the `x86_64_cuda*` files are Git-LFS pointer stubs. Building the
TensorRT engines is unaffected. See the repository issue tracker / build
`libspconv` for x86_64 before running inference on x86_64 hardware.
