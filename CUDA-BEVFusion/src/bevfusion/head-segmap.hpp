#ifndef __HEAD_SEGMAP_HPP__
#define __HEAD_SEGMAP_HPP__

#include <memory>
#include <string>

#include "common/dtype.hpp"

namespace bevfusion {
namespace head {
namespace segmap {

struct SegHeadParameters {
  std::string model;
  bool enabled = false;
};

struct MapView {
  const nvtype::half* data = nullptr;
  int classes = 0;
  int height = 0;
  int width = 0;

  bool valid() const { return data != nullptr && classes > 0 && height > 0 && width > 0; }
  size_t numel() const { return static_cast<size_t>(classes) * height * width; }
};

class SegMap {
 public:
  virtual ~SegMap() = default;
  virtual MapView forward(const nvtype::half* fusion_feature, void* stream) = 0;
  virtual void print() = 0;
};

std::shared_ptr<SegMap> create_maphead(const SegHeadParameters& param);

}  // namespace segmap
}  // namespace head
}  // namespace bevfusion

#endif  // __HEAD_SEGMAP_HPP__
