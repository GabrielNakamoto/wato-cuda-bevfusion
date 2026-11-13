#include "head-segmap.hpp"

#include <cuda_fp16.h>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include "common/check.hpp"
#include "common/tensorrt.hpp"
#include "common/launch.cuh"

namespace bevfusion {
namespace head {
namespace segmap {

static __global__ void threshold_kernel(const unsigned int numel, const float log_threshold, const half *logits, uint8_t *output) {
  int i = cuda_linear_index;
  if (i >= numel) return;
  output[i] = __half2float(logits[i]) > log_threshold ? 100 : 0;
}

class SegMapImplement : public SegMap {
 public:
  const char* BindingInput  = "middle";
  const char* BindingOutput = "segmap";

  virtual ~SegMapImplement() {
    if (logits_) checkRuntime(cudaFree(logits_));
    if (output_) checkRuntime(cudaFree(output_));
    if (view_.data) free(view_.data);
  }

  bool init(const SegHeadParameters& param) {
    if (!param.enabled) return false;
    engine_ = TensorRT::load(param.model);
    confidence_threshold = param.confidence_threshold;
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported for map head.\n");
      return false;
    }

    auto shape = engine_->static_dims(BindingOutput);
    size_t volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    Asserts(engine_->dtype(BindingOutput) == TensorRT::DType::HALF, "Invalid map head output data type.");
    Asserts(shape.size() == 4 && shape[0] == 1, "Map head output must be NCHW with batch size 1.");
    Asserts(shape[2] == shape[3], "Map head output must be square.");
    view_.classes = shape[1];
    // transpose folded into onnx output, BEV convention
    view_.width  = shape[3];
    view_.height   = shape[2];
    view_.data = (uint8_t*)malloc(view_.numel());

    checkRuntime(cudaMalloc(&output_, volume*sizeof(uint8_t)));
    checkRuntime(cudaMalloc(&logits_, volume*sizeof(half)));
    return true;
  }

  virtual void print() override { engine_->print("Head SegMap"); }

  virtual MapView forward(const nvtype::half* fusion_feature, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    engine_->forward(
        std::unordered_map<std::string, const void*>{
            {BindingInput, fusion_feature},
            {BindingOutput, logits_},
        },
        _stream);

    cuda_linear_launch(threshold_kernel, _stream, view_.numel(),
      std::log(confidence_threshold / (1. - confidence_threshold)), logits_, output_);

    checkRuntime(cudaMemcpyAsync((void*)view_.data, output_, view_.numel(),
      cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream));

    return view_;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  float confidence_threshold;
  half* logits_ = nullptr;
  uint8_t* output_ = nullptr;
  MapView view_;
};

std::shared_ptr<SegMap> create_maphead(const SegHeadParameters& param) {
  std::shared_ptr<SegMapImplement> instance(new SegMapImplement());
  if (!instance->init(param)) instance.reset();
  return instance;
}

}  // namespace segmap
}  // namespace head
}  // namespace bevfusion
