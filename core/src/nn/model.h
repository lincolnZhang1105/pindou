// ONNX 模型推理的薄封装（T0.11）。识别主路径只用自训的小模型，不用平台 OCR（03 §1）。
//
// 隐私（AGENTS 规则 5）：ONNX Runtime 官方构建默认开启微软的 1DS 遥测（会联网、生成设备标识）。
// 这里在第一次初始化 ORT 之前设置 ORT_DISABLE_TELEMETRY=1，并调用 DisableTelemetryEvents；
// App 里用的定制构建还必须加 --no_telemetry 编译（03 §8）。
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bc::nn {

struct Tensor {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

// 编译时是否带了 ONNX Runtime
bool available();

class Model {
 public:
  // 加载失败抛出 bc::Error(invalid_input)
  explicit Model(const std::string& path);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // 单输入单输出（float32）。input 长度必须等于 shape 各维之积。
  Tensor run(const std::vector<float>& input, const std::vector<int64_t>& shape) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bc::nn
