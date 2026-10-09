#ifndef __HEAD_SEGMAP_HPP___
#define __HEAD_SEGMAP_HPP__

#include "common/dtype.hpp"
#include <memory>
#include <string>

#include "common/dtype.hpp"

// mostly copied from:
// https://github.com/hualixueyuan/BEVFusion-ROS-TensorRT-GDUT/blob/11bc73889c6e401514665385ad74cca5ce9a3ff2/src/bevfusion/head-map.hpp

namespace bevfusion {
namespace head {
namespace segmap {

struct SegHeadParameters {
  std::string model;
};

struct MapView {
  const nvtype::half* data = nullptr;
  int classes = 0;
  int height = 0;
  int width = 0;

  bool valid() const { return data != nullptr && classes > 0 && height > 0 && width > 0; }
  size_t numel() const { return static_cast<int>(classes) * height * width; }
}:

class SegMap {
public:
  virtual MapView forward(const nvtype::half *fusion_feature, void* stream) = 0;
  virtual void print() = 0;
};

std::shared_ptr<SegMap> create_maphead(const SegHeadParameters& param);

} // namespace map
} // namespace head
} // namespace segmap

#endif
