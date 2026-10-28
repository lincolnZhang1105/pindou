"""θ（拉普拉斯方差阈值）的标定：只用 dev 集。

依据：dev 集真值里人工标注的 `blurry` 标签（标注者目测「糊」）。只看格边长 ≥ min_cell_px 的主统计池图纸
（格边长 < min_cell_px 的已经按第一条规则判为 blurry，θ 对它们不起作用）。
候选阈值取相邻两个拉普拉斯方差的中点（外加两端），判定规则 lap_var < θ 即 blurry，
选 Youden J = TPR − FPR 最大的阈值；并列时取最小的 θ（Hard 集更小、Clean 集更大，第 2 条更严格，偏保守）。
"""
from __future__ import annotations

from bench.eval.config import Config
from bench.eval.dataset import Unit
from bench.eval.report import md_table


def suggest_theta(units: list[Unit], cfg: Config) -> dict:
    cand = [u for u in units if u.in_main and u.min_cell_px >= cfg.min_cell_px]
    pos = [u for u in cand if "blurry" in u.human_tags]
    neg = [u for u in cand if "blurry" not in u.human_tags]
    out = {"n": len(cand), "n_pos": len(pos), "n_neg": len(neg), "theta": None, "j": None, "tpr": None, "fpr": None,
           "table": sorted(cand, key=lambda u: (u.lap_var, u.image_id))}
    if not pos or not neg:
        out["reason"] = "dev 集（格边长 ≥ 阈值的主统计池图纸）里人工 blurry 标签只有一类，无法有监督地标定"
        return out
    vals = sorted({u.lap_var for u in cand})
    thetas = [vals[0]] + [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [vals[-1] + 1.0]
    best = None
    for th in thetas:
        tp = sum(u.lap_var < th for u in pos)
        fp = sum(u.lap_var < th for u in neg)
        tpr, fpr = tp / len(pos), fp / len(neg)
        j = tpr - fpr
        if best is None or j > best[0] + 1e-12:
            best = (j, th, tpr, fpr)
    out["j"], out["theta"], out["tpr"], out["fpr"] = best
    return out


def render(res: dict, cfg: Config) -> str:
    L = ["# θ 标定（只用 dev 集）", "",
         f"当前 config：θ = {cfg.theta:g}（{'已冻结' if cfg.theta_frozen else '占位'}），最小格边长阈值 {cfg.min_cell_px:g} px。", "",
         f"参与标定：主统计池中格边长 ≥ {cfg.min_cell_px:g} px 的 dev 图纸 {res['n']} 张，"
         f"其中人工标为 blurry 的 {res['n_pos']} 张、未标的 {res['n_neg']} 张。", ""]
    if res["theta"] is None:
        L.append(f"**无法给出建议值**：{res.get('reason', '')}。请人工查看下表的分布后决定，并在 dev 集上补充 blurry 标注。")
    else:
        L.append(f"**建议 θ = {res['theta']:.4g}**（Youden J = {res['j']:.3f}，TPR = {res['tpr'] * 100:.1f}%，"
                 f"FPR = {res['fpr'] * 100:.1f}%；并列时取最小值）。")
        L.append("")
        L.append("确认后把 `bench/eval/config.yaml` 的 `blurry.theta` 改为该值、`theta_frozen` 改为 true，之后不再改动"
                 "（改动会让 gate 基线失效，且 test 集评测要求 θ 已冻结）。")
    L.append("")
    L.append("注意：拉普拉斯方差同时受内容（网格线、格内文字密度）和分辨率影响，只在与 dev 集同类的图纸上有意义。")
    L.append("")
    rows = [[u.image_id, u.type, u.size_bucket, f"{u.min_cell_px:.2f}", f"{u.lap_var:.1f}",
             "是" if "blurry" in u.human_tags else "", "是" if (res["theta"] is not None and u.lap_var < res["theta"]) else ""]
            for u in res["table"]]
    L.append(md_table(["image_id", "类型", "尺寸", "格边长 px", "拉普拉斯方差", "人工 blurry", "按建议 θ 判为 blurry"], rows))
    L.append("")
    return "\n".join(L)
