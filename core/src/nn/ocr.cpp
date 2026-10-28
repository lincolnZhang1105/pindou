#include "nn/ocr.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

#include "common/common.h"

namespace bc::nn {
namespace {

// Python 的 round() 是「四舍六入五成双」；std::nearbyint 在默认舍入模式下行为相同
int py_round(double x) { return static_cast<int>(std::nearbyint(x)); }

double logsumexp(double a, double b) {
  if (a == -std::numeric_limits<double>::infinity()) return b;
  if (b == -std::numeric_limits<double>::infinity()) return a;
  const double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

}  // namespace

cv::Mat ink_crop(const cv::Mat& gray, const cv::Rect& box) {
  CV_Assert(gray.type() == CV_8UC1);
  const int pad = std::max(2, py_round(0.15 * box.height));
  const int x0 = std::max(0, box.x - pad), y0 = std::max(0, box.y - pad);
  const int x1 = std::min(gray.cols, box.x + box.width + pad), y1 = std::min(gray.rows, box.y + box.height + pad);
  require(x1 > x0 && y1 > y0, BC_ERR_INTERNAL, "文字框在图外");
  const cv::Mat c = gray(cv::Rect(x0, y0, x1 - x0, y1 - y0));
  std::vector<uint8_t> flat;
  flat.reserve(c.total());
  for (int y = 0; y < c.rows; ++y) flat.insert(flat.end(), c.ptr<uint8_t>(y), c.ptr<uint8_t>(y) + c.cols);
  std::vector<uint8_t> sorted = flat;
  std::sort(sorted.begin(), sorted.end());
  const int bg = sorted[sorted.size() / 2];
  cv::Mat ink(c.size(), CV_32F);
  float mx = 0;
  for (int y = 0; y < c.rows; ++y) {
    const uint8_t* s = c.ptr<uint8_t>(y);
    float* d = ink.ptr<float>(y);
    for (int x = 0; x < c.cols; ++x) {
      d[x] = static_cast<float>(std::abs(int(s[x]) - bg));
      mx = std::max(mx, d[x]);
    }
  }
  ink /= std::max(mx, 32.0f);
  int nw = py_round(double(c.cols) * kOcrInputH / c.rows);
  nw = std::clamp(nw, 16, 320);
  cv::Mat out;
  cv::resize(ink, out, cv::Size(nw, kOcrInputH), 0, 0, c.rows > kOcrInputH ? cv::INTER_AREA : cv::INTER_LINEAR);
  return out;
}

std::unique_ptr<OcrModel> OcrModel::load(const std::string& models_dir, const std::string& name) {
  if (models_dir.empty() || !available()) return nullptr;
  const std::string onnx = models_dir + "/" + name + ".onnx", meta = models_dir + "/" + name + ".json";
  std::ifstream fm(meta, std::ios::binary);
  if (!std::ifstream(onnx).good() || !fm.good()) return nullptr;
  std::stringstream ss;
  ss << fm.rdbuf();
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(ss.str());
  } catch (const nlohmann::json::exception&) {
    fail(BC_ERR_INVALID_INPUT, "模型元数据不是合法 JSON：" + meta);
  }
  auto m = std::make_unique<OcrModel>();
  m->charset = j.at("charset").get<std::string>();
  require(j.value("blank", 0) == 0 && j.value("input_h", 0) == kOcrInputH, BC_ERR_INVALID_INPUT,
          "模型元数据和 core 的约定不符：" + meta);
  m->name = name;
  m->model = std::make_unique<Model>(onnx);
  return m;
}

OcrText read_text(const OcrModel& m, const cv::Mat& gray, const cv::Rect& box) {
  const cv::Mat x = ink_crop(gray, box);
  std::vector<float> input(x.total());
  for (int y = 0; y < x.rows; ++y) std::copy(x.ptr<float>(y), x.ptr<float>(y) + x.cols, input.begin() + y * x.cols);
  const Tensor out = m.model->run(input, {1, 1, kOcrInputH, x.cols});  // [T, 1, C]
  require(out.shape.size() == 3 && out.shape[1] == 1, BC_ERR_INTERNAL, "OCR 模型输出形状不对");
  OcrText t;
  t.T = static_cast<int>(out.shape[0]);
  t.C = static_cast<int>(out.shape[2]);
  t.lp = out.data;
  int prev = 0;
  for (int i = 0; i < t.T; ++i) {
    const float* row = &t.lp[static_cast<size_t>(i) * t.C];
    const int k = static_cast<int>(std::max_element(row, row + t.C) - row);
    if (k != prev && k != 0 && k - 1 < static_cast<int>(m.charset.size())) t.text.push_back(m.charset[k - 1]);
    prev = k;
  }
  t.logp = ctc_logprob(t, m.charset, t.text);
  return t;
}

double ctc_logprob(const OcrText& t, const std::string& charset, const std::string& label) {
  const double NEG = -std::numeric_limits<double>::infinity();
  std::vector<int> l;
  for (char ch : label) {
    const auto pos = charset.find(ch);
    if (pos == std::string::npos) return NEG;
    l.push_back(static_cast<int>(pos) + 1);
  }
  // 扩展序列：blank, l1, blank, l2, ..., blank
  const int S = 2 * static_cast<int>(l.size()) + 1;
  if (t.T == 0) return NEG;
  auto ext = [&](int s) { return (s % 2 == 0) ? 0 : l[s / 2]; };
  auto lp = [&](int i, int k) { return static_cast<double>(t.lp[static_cast<size_t>(i) * t.C + k]); };
  std::vector<double> a(S, NEG), b(S, NEG);
  a[0] = lp(0, 0);
  if (S > 1) a[1] = lp(0, ext(1));
  for (int i = 1; i < t.T; ++i) {
    std::fill(b.begin(), b.end(), NEG);
    for (int s = 0; s < S; ++s) {
      double v = a[s];
      if (s >= 1) v = logsumexp(v, a[s - 1]);
      if (s >= 2 && ext(s) != 0 && ext(s) != ext(s - 2)) v = logsumexp(v, a[s - 2]);
      b[s] = v == NEG ? NEG : v + lp(i, ext(s));
    }
    a.swap(b);
  }
  return S > 1 ? logsumexp(a[S - 1], a[S - 2]) : a[S - 1];
}

}  // namespace bc::nn
