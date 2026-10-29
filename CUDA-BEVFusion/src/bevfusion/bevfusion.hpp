/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __BEVFUSION_HPP__
#define __BEVFUSION_HPP__

#include <utility>
#include <vector>

#include "camera-backbone.hpp"
#include "camera-bevpool.hpp"
#include "camera-depth.hpp"
#include "camera-geometry.hpp"
#include "camera-normalization.hpp"
#include "camera-vtransform.hpp"
#include "head-segmap.hpp"
#include "head-transbbox.hpp"
#include "lidar-scn.hpp"
#include "transfusion.hpp"

namespace bevfusion {

using HeadOutputs = std::pair<std::vector<head::transbbox::BoundingBox>, head::segmap::MapView>;

struct CoreParameter {
  std::string camera_model;
  std::string camera_vtransform;
  camera::GeometryParameter geometry;
  camera::NormalizationParameter normalize;
  lidar::SCNParameter lidar_scn;
  std::string transfusion;
  head::segmap::SegHeadParameters segmap;
  head::transbbox::TransBBoxParameter transbbox;
};

class Core {
 public:
  virtual ~Core() = default;
  virtual HeadOutputs forward(const unsigned char** camera_images, const nvtype::half* lidar_points,
                              int num_points, void* stream) = 0;

  virtual HeadOutputs forward_no_normalize(const nvtype::half* camera_normed_images_device,
                                           const nvtype::half* lidar_points, int num_points,
                                           void* stream) = 0;

  virtual void print() = 0;
  virtual void set_timer(bool enable) = 0;

  virtual void update(const float* camera2lidar, const float* camera_intrinsics, const float* lidar2image,
                      const float* img_aug_matrix, void* stream = nullptr) = 0;

  virtual void free_excess_memory() = 0;
};

std::shared_ptr<Core> create_core(const CoreParameter& param);

};  // namespace bevfusion

#endif  // __BEVFUSION_HPP__
