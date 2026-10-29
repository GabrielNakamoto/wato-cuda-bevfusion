/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include <cuda_fp16.h>
#include <numeric>

#include "camera-backbone.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace bevfusion {
namespace camera {

class BackboneImplement : public Backbone {
 public:
  const char* BindingImages        = "img";
  const char* BindingDepth         = "depth";
  const char* BindingCameraDepths  = "camera_depth_weights";
  const char* BindingCameraFeature = "camera_feature";

  virtual ~BackboneImplement() {
    if (feature_) checkRuntime(cudaFree(feature_));
    if (depth_weights_) checkRuntime(cudaFree(depth_weights_));
  }

  bool init(const std::string& model) {
    engine_ = TensorRT::load(model);
    if (engine_ == nullptr) return false;

    has_depth_input_ = (engine_->num_bindings() >= 4);

    depth_dims_   = engine_->static_dims(BindingCameraDepths);
    feature_dims_ = engine_->static_dims(BindingCameraFeature);
    int32_t volumn = std::accumulate(depth_dims_.begin(), depth_dims_.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMalloc(&depth_weights_, volumn * sizeof(nvtype::half)));

    volumn = std::accumulate(feature_dims_.begin(), feature_dims_.end(), 1, std::multiplies<int32_t>());
    checkRuntime(cudaMalloc(&feature_, volumn * sizeof(nvtype::half)));

    // N C D H W
    camera_shape_ = {feature_dims_[0], feature_dims_[3], depth_dims_[1], feature_dims_[1], feature_dims_[2]};
    return true;
  }

  virtual void print() override { engine_->print("Camera Backbone"); }

  virtual void forward(const nvtype::half* images, const nvtype::half* depth, void* stream = nullptr) override {
    if (has_depth_input_) {
      engine_->forward({
        {BindingImages, images},
        {BindingDepth, depth},
        {BindingCameraDepths, depth_weights_},
        {BindingCameraFeature, feature_}
      }, static_cast<cudaStream_t>(stream));
    } else {
      engine_->forward({
        {BindingImages, images},
        {BindingCameraDepths, depth_weights_},
        {BindingCameraFeature, feature_}
      }, static_cast<cudaStream_t>(stream));
    }
  }

  virtual nvtype::half* depth() override { return depth_weights_; }
  virtual nvtype::half* feature() override { return feature_; }
  virtual std::vector<int> depth_shape() override { return depth_dims_; }
  virtual std::vector<int> feature_shape() override { return feature_dims_; }
  virtual std::vector<int> camera_shape() override { return camera_shape_; }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  nvtype::half* feature_ = nullptr;
  nvtype::half* depth_weights_ = nullptr;
  std::vector<int> feature_dims_, depth_dims_, camera_shape_;
  bool has_depth_input_ = true;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model) {
  std::shared_ptr<BackboneImplement> instance(new BackboneImplement());
  if (!instance->init(model)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion
