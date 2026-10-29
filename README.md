# CUDA BEVFusion with Lane Segmentation




https://github.com/user-attachments/assets/c369d035-2d26-4a82-86ba-fd10e7aca5c5

> [!Note]
> If you have troubles use LLM lol

Requirements:
- modern-ish Nvidia card
- docker

How to run it:
```
git clone -j8 -b tensorrt-map --recursive https://github.com/GabrielNakamoto/wato-cuda-bevfusion/
cd wato-cuda-bevfusion && bash docker_infer.sh
```
What it does:
- extends the base bevfusion docker to include the tensorrt/onnx pipeline dependencies
- downloads the pretrained seg model and example data
- exports onnx weights for the seg model
- builds the engine and runs a sample forward pass using TensorRT
