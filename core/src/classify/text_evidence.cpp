#include "classify/text_evidence.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace bc {
namespace {

// 格子里文字像素的外接框（和 sample_cell 用同样的阈值）；找不到时返回空框
cv::Rect text_box(const cv::Mat& gray, const CellRect& r, const CellSample& s, const SampleParams& sp) {
  const double w = r.x1 - r.x0, h = r.y1 - r.y0;
  const int x0 = std::max(0, static_cast<int>(std::ceil(r.x0 + w * sp.margin_frac)));
  const int y0 = std::max(0, static_cast<int>(std::ceil(r.y0 + h * sp.margin_frac)));
  const int x1 = std::min(gray.cols, static_cast<int>(std::floor(r.x1 - w * sp.margin_frac)));
  const int y1 = std::min(gray.rows, static_cast<int>(std::floor(r.y1 - h * sp.margin_frac)));
  const int bg = (299 * s.bg[0] + 587 * s.bg[1] + 114 * s.bg[2] + 500) / 1000;
  int bx0 = x1, by0 = y1, bx1 = x0 - 1, by1 = y0 - 1;
  for (int y = y0; y < y1; ++y) {
    const uint8_t* g = gray.ptr<uint8_t>(y);
    for (int x = x0; x < x1; ++x) {
      if (std::abs(int(g[x]) - bg) >= sp.text_delta) {
        bx0 = std::min(bx0, x);
        by0 = std::min(by0, y);
        bx1 = std::max(bx1, x);
        by1 = std::max(by1, y);
      }
    }
  }
  if (bx1 < bx0 || by1 < by0) return {};
  return cv::Rect(cv::Point(bx0, by0), cv::Point(bx1 + 1, by1 + 1));
}

double logsumexp(const std::vector<double>& v) {
  double m = -std::numeric_limits<double>::infinity();
  for (double x : v) m = std::max(m, x);
  if (!std::isfinite(m)) return m;
  double s = 0;
  for (double x : v) s += std::exp(x - m);
  return m + std::log(s);
}

}  // namespace

int apply_text_evidence(Assignment& a, const std::vector<CellSample>& samples, const std::vector<CellRef>& refs,
                        const std::vector<std::string>& class_codes, const nn::OcrModel& ocr, const AssignParams& p,
                        const TextEvidenceParams& tp) {
  const int K = static_cast<int>(a.classes.size());
  std::vector<int> known;
  for (int k = 0; k < K; ++k) {
    if (!class_codes[k].empty()) known.push_back(k);
  }
  if (known.size() < 2) return 0;
  const double two_s2 = 2.0 * p.sigma * p.sigma;
  const SampleParams sp;

  // 找难格子，按颜色置信度从低到高排，最多读 max_cells 个
  struct Cand {
    float conf;
    int cell;
  };
  std::vector<Cand> cands;
  for (int i = 0; i < static_cast<int>(a.cls.size()); ++i) {
    const int c = a.cls[i];
    if (c < 0 || class_codes[c].empty() || !samples[i].has_text || !refs[i].gray) continue;
    const auto& r = refs[i].rect;
    if (std::min(r.x1 - r.x0, r.y1 - r.y0) < tp.min_pitch) continue;
    double d1 = 1e9, d2 = 1e9;
    for (int k : known) {
      const double d = color::delta_e_2000(samples[i].bg_lab, a.classes[k].lab);
      if (d < d1) {
        d2 = d1;
        d1 = d;
      } else if (d < d2) {
        d2 = d;
      }
    }
    if (a.conf[i] < tp.conf_thr || d2 - d1 < tp.gap_de) cands.push_back({a.conf[i], i});
  }
  std::stable_sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.conf < y.conf; });
  if (static_cast<int>(cands.size()) > tp.max_cells) cands.resize(tp.max_cells);

  int changed = 0;
  for (const auto& cd : cands) {
    const int i = cd.cell;
    const cv::Rect tb = text_box(*refs[i].gray, refs[i].rect, samples[i], sp);
    if (tb.area() < 6 || tb.height < 3) continue;
    const nn::OcrText t = nn::read_text(ocr, *refs[i].gray, tb);
    // 颜色和文字的对数概率，各自在候选类里归一化后相加
    std::vector<double> lc, lt;
    for (int k : known) {
      const double d = color::delta_e_2000(samples[i].bg_lab, a.classes[k].lab);
      lc.push_back(-d * d / two_s2);
      lt.push_back(nn::ctc_logprob(t, ocr.charset, class_codes[k]));
    }
    const double zc = logsumexp(lc), zt = logsumexp(lt);
    if (!std::isfinite(zt)) continue;
    std::vector<double> post(known.size());
    for (size_t j = 0; j < known.size(); ++j) post[j] = (lc[j] - zc) + tp.text_weight * (lt[j] - zt);
    const double zp = logsumexp(post);
    const size_t best = static_cast<size_t>(std::max_element(post.begin(), post.end()) - post.begin());
    const int k = known[best];
    const float conf = static_cast<float>(std::exp(post[best] - zp));
    if (k != a.cls[i]) {
      --a.classes[a.cls[i]].cells;
      ++a.classes[k].cells;
      a.cls[i] = k;
      ++changed;
    }
    a.conf[i] = conf;
  }
  return changed;
}

}  // namespace bc
