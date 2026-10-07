#include "head-segmap.hpp"

#include <numeric>
#include <vector>

#include "common/check.hpp"

#include "common/tensorrt.hpp"

namespace bevfusion {
namespace head {
namespace segmap {

class SegMapImplement : public SegMap {
public:

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

    auto shape = engine_->static_dims("map_logits");
    Asserts(engine_->dtype("map_logits") == TensorRT::DType::HALF, "Invalid map head output data type.");
    Asserts(shape.size() == 4 && shape[0] == 1, "Map head output must be NCHW with batch size 1.");
    view_.classes = shape[1];
    view_.height = shape[2];
    view_.width = shape[3];

    size_t volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    checkRuntime(cudaMalloc(&output_, volume * sizeof(half)));
    view_.data = reinterpret_cast<nvtype::half*>(output_);
    return true;
  }

	virtual void print() { engine_->print("Map head"); }

	virtual MapView forward(const nvtype::half *fusion_feature, void* stream) {
		std::vector<const void*> bindings(engine_->num_bindings(), nullptr):
		bindings[engine_->index("middle")] = fusion_features;
		bindings[engine_->index("map_logits")] = output_;
		Asserts(engine_->forward(bindings, stream), "Failed to execute map head on TensorRT engine.");
		return view_;
	}

private:
	std::shared_ptr<TensorRT::Engine> engine_;
	half* output_ = nullptr;
	MapView view_;
};

std::shared_ptr<SegMap> create_maphead(const SegHeadParameters& param) {
	std::shared_ptr<SegMapImplement> instance(new SegMapImplement());
	if (!instance->init(param)) instance.reset();
	return instance;
}
	
}
}
}
