"""各子命令共用的流程：路径解析、评测一个预测结果目录、results.json 读写、数据集指纹。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from bench.common import formats as F
from bench.eval.config import Config
from bench.eval.dataset import Unit, load_units
from bench.eval.discipline import TestDisciplineError, find_usage, git_rev, now_iso
from bench.eval.scoring import ImageResult, score_units

RESULTS_SCHEMA = 1


class UsageError(RuntimeError):
    pass


@dataclass
class Paths:
    labels: Path
    images: Path
    manifest: dict[str, dict]
    data: Path | None


def resolve_paths(labels: str | None, images: str | None, manifest: str | None) -> Paths:
    data = F.data_dir(required=False)
    if labels is None and data is None:
        raise UsageError("没有 --labels，也没有设置 PINDOU_DATA_DIR")
    if images is None and data is None:
        raise UsageError("没有 --images，也没有设置 PINDOU_DATA_DIR（blurry 判定需要原图）")
    lab = Path(labels) if labels else data / "labels"  # type: ignore[operator]
    img = Path(images) if images else data / "images"  # type: ignore[operator]
    if not lab.is_dir():
        raise UsageError(f"真值目录不存在：{lab}")
    if not img.is_dir():
        raise UsageError(f"图片目录不存在：{img}")
    man = F.load_manifest(Path(manifest) if manifest else F.DATASETS / "manifest.jsonl")
    return Paths(lab, img, man, data)


def resolve_preds_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    data = F.data_dir(required=False)
    if data is not None and (data / "preds" / arg).is_dir():
        return data / "preds" / arg
    raise UsageError(f"预测结果目录不存在：{arg}（也不在 $PINDOU_DATA_DIR/preds/ 下）")


def default_report_dir(name: str, split: str) -> Path:
    data = F.data_dir(required=False)
    if data is None:
        raise UsageError("没有 --out，也没有设置 PINDOU_DATA_DIR")
    return data / "reports" / name / split


def resolve_results_file(arg: str, split: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        for c in (p / "results.json", p / split / "results.json"):
            if c.is_file():
                return c
    data = F.data_dir(required=False)
    if data is not None:
        c = data / "reports" / arg / split / "results.json"
        if c.is_file():
            return c
    raise UsageError(f"找不到 {arg} 在 {split} 集上的 results.json（先用 run 生成）")


def units_fingerprint(units: list[Unit], cfg: Config) -> str:
    """数据集指纹：图纸集合、真值内容、Hard/口径划分、评测参数。任何一项变化，gate 基线都要重建。"""
    rows = [[u.image_id, u.label_sha, u.type, u.hard, u.in_main, u.in_flag_scope] for u in units]
    blob = json.dumps({"units": rows, "config": cfg.fingerprint()}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def evaluate(split: str, preds_dir: Path, cfg: Config, paths: Paths,
             allow_missing: bool = False) -> tuple[list[Unit], list[ImageResult], list[str], str]:
    units, warns = load_units(split, cfg, paths.labels, paths.images, paths.manifest, allow_missing)
    results, w2, preds_sha = score_units(units, preds_dir, cfg, paths.manifest)
    return units, results, warns + w2, preds_sha


def build_results_json(*, run: str, split: str, preds_dir: Path, preds_sha: str, labels_dir: Path,
                       cfg: Config, units: list[Unit], results: list[ImageResult], warnings: list[str],
                       test_reason: str | None) -> tuple[dict, str]:
    meta = {
        "tool": "bench.eval", "schema": RESULTS_SCHEMA, "run": run, "split": split, "created_at": now_iso(),
        "git_rev": git_rev(), "preds_dir": str(preds_dir), "preds_sha256": preds_sha, "labels_dir": str(labels_dir),
        "config_sha256": cfg.fingerprint(), "units_sha256": units_fingerprint(units, cfg), "config": cfg.raw,
        "test_reason": test_reason, "warnings": warnings,
    }
    doc = dict(meta, images=[r.to_dict() for r in results])
    text = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=False) + "\n"
    return meta, text


def load_results(path: Path) -> tuple[dict, list[ImageResult], str]:
    """读取 results.json，返回 (元数据, 逐图结果, 文件 sha256)。"""
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("tool") != "bench.eval" or doc.get("schema") != RESULTS_SCHEMA:
        raise UsageError(f"{path} 不是本评测器生成的 results.json（或版本不符）")
    results = [ImageResult.from_dict(d) for d in doc["images"]]
    meta = {k: v for k, v in doc.items() if k != "images"}
    return meta, results, hashlib.sha256(raw).hexdigest()


def verify_test_results(meta: dict, sha: str, reason: str | None = None) -> dict:
    """test 集的 results.json 必须登记在 test_usage.jsonl 里，且文件内容没被改过。返回对应的使用记录。"""
    if meta.get("split") != "test":
        raise UsageError(f"{meta.get('run')} 不是 test 集的结果")
    if reason is not None and meta.get("test_reason") != reason:
        raise UsageError(f"{meta.get('run')} 的 test_reason 是 {meta.get('test_reason')}，这里需要 {reason}")
    rec = find_usage(meta.get("test_reason"), meta.get("run"))
    if rec is None:
        raise TestDisciplineError(f"test_usage.jsonl 里没有 run={meta.get('run')}、reason={meta.get('test_reason')} 的记录")
    if rec.get("results_sha256") != sha:
        raise TestDisciplineError(f"{meta.get('run')} 的 results.json 与 test_usage.jsonl 登记的 sha256 不一致（文件被改动过）")
    return rec
