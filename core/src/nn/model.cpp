#include "nn/model.h"

#include <cstdlib>
#include <filesystem>
#include <mutex>

#include "common/common.h"

#ifdef BC_HAVE_ORT
#include <onnxruntime_cxx_api.h>
#endif

namespace bc::nn {

#ifdef BC_HAVE_ORT

namespace {

// 全进程共用一个 Ort::Env，故意不在退出时析构：进程退出时 ORT 自己的静态对象（日志管理器的互斥锁等）
// 可能已经先析构，再析构 Env 会锁一把已销毁的互斥锁，Android 的 bionic 会因此直接中止进程
Ort::Env& env() {
  static std::once_flag once;
  static Ort::Env* e = nullptr;
  std::call_once(once, [] {
    // 必须在 ORT 初始化之前设置，否则 1DS 上传器和设备标识已经创建
#ifdef _WIN32
    _putenv_s("ORT_DISABLE_TELEMETRY", "1");
#else
    setenv("ORT_DISABLE_TELEMETRY", "1", 1);
#endif
    e = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "beadcore");
    e->DisableTelemetryEvents();
  });
  return *e;
}

}  // namespace

struct Model::Impl {
  std::unique_ptr<Ort::Session> session;
  std::string input_name, output_name;
};

bool available() { return true; }

Model::Model(const std::string& path) : impl_(std::make_unique<Impl>()) {
  try {
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(1);  // 结果确定；并行交给调用方（每个 Isolate 一个 ctx）
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
    const std::wstring p = std::filesystem::u8path(path).wstring();
    impl_->session = std::make_unique<Ort::Session>(env(), p.c_str(), opt);
#else
    impl_->session = std::make_unique<Ort::Session>(env(), path.c_str(), opt);
#endif
    Ort::AllocatorWithDefaultOptions alloc;
    impl_->input_name = impl_->session->GetInputNameAllocated(0, alloc).get();
    impl_->output_name = impl_->session->GetOutputNameAllocated(0, alloc).get();
  } catch (const Ort::Exception& e) {
    fail(BC_ERR_INVALID_INPUT, "无法加载模型 " + path + "：" + e.what());
  }
}

Model::~Model() = default;

Tensor Model::run(const std::vector<float>& input, const std::vector<int64_t>& shape) const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  require(n == static_cast<int64_t>(input.size()), BC_ERR_INTERNAL, "模型输入长度和形状不符");
  try {
    const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(input.data()), input.size(), shape.data(),
                                                    shape.size());
    const char* in_names[] = {impl_->input_name.c_str()};
    const char* out_names[] = {impl_->output_name.c_str()};
    auto outs = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    const auto info = outs[0].GetTensorTypeAndShapeInfo();
    Tensor t;
    t.shape = info.GetShape();
    const float* p = outs[0].GetTensorData<float>();
    t.data.assign(p, p + info.GetElementCount());
    return t;
  } catch (const Ort::Exception& e) {
    fail(BC_ERR_INTERNAL, std::string("模型推理失败：") + e.what());
  }
}

#else  // 不带 ORT 的构建：只走颜色路径

struct Model::Impl {};
bool available() { return false; }
Model::Model(const std::string& path) { fail(BC_ERR_INVALID_INPUT, "这个构建没有带 ONNX Runtime，无法加载 " + path); }
Model::~Model() = default;
Tensor Model::run(const std::vector<float>&, const std::vector<int64_t>&) const {
  fail(BC_ERR_INTERNAL, "这个构建没有带 ONNX Runtime");
}

#endif

}  // namespace bc::nn
