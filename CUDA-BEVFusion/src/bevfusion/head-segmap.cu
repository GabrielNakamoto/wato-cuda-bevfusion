#include "head-segmap.hpp"

#include <cuda_fp16.h>
#include <numeric>
#include <unordered_map>

#include "common/check.hpp"
#include "common/tensorrt.hpp"

namespace bevfusion {
namespace head {
namespace segmap {

class SegMapImplement : public SegMap {
 public:
  const char* BindingInput  = "middle";
  const char* BindingOutput = "segmap";

  virtual ~SegMapImplement() {
    if (output_) checkRuntime(cudaFree(output_));
  }

  bool init(const SegHeadParameters& param) {
    if (!param.enabled) return false;
    engine_ = TensorRT::load(param.model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported for map head.\n");
      return false;
    }

    auto shape = engine_->static_dims(BindingOutput);
    size_t volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    Asserts(engine_->dtype(BindingOutput) == TensorRT::DType::HALF, "Invalid map head output data type.");
    Asserts(shape.size() == 4 && shape[0] == 1, "Map head output must be NCHW with batch size 1.");
    view_.classes = shape[1];
    view_.height  = shape[2];
    view_.width   = shape[3];
    view_.data    = (uint8_t*)malloc(volume);
    host_buffer = (half*)malloc(volume * sizeof(half));

    checkRuntime(cudaMalloc(&output_, volume * sizeof(half)));
    return true;
  }

  virtual void print() override { engine_->print("Head SegMap"); }

  virtual MapView forward(const nvtype::half* fusion_feature, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    engine_->forward(
        std::unordered_map<std::string, const void*>{
            {BindingInput, fusion_feature},
            {BindingOutput, output_},
        },
        _stream);

    checkRuntime(cudaMemcpyAsync((void*)host_buffer, output_, view_.numel() * sizeof(half),
      cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream));

    for (std::size_t i=0; i<view_.numel(); ++i)
      view_.data[i] = __half2float(host_buffer[i]) > 0;

    return view_;
  }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  half* output_ = nullptr;
  half *host_buffer = nullptr;
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
