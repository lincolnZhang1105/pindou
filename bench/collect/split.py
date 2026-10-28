"""dev / test 划分（04 §2.1）：按（size_bucket, type）分层，dev : test = 1 : 2。

- 排除 T7（多图合集）和只用于分辨率对比的 Web 版本；
- 分页图（T4）同一个 group 作为一个单元，不拆开；
- **增量划分**：已经在 dev.txt / test.txt 里的单元保持原样，只给新单元分配。否则补充样本后重新划分，
  会把已经用来调参的 dev 图换进 test，污染 test 集；
- 固定种子，同样的输入得到同样的结果；
- test.lock 存在时拒绝覆盖。
"""
from __future__ import annotations

import random

from bench.common import dataset as ds
from bench.common import formats

DEFAULT_SEED = 20261027


def _target_test(n: int) -> int:
    """一个分层里 n 个单元，test 应有几个：round(2n/3)，四舍五入偏向 test（test 集有覆盖要求）。"""
    return (2 * n + 1) // 3


def plan_split(manifest: dict[str, dict], existing_dev: list[str], existing_test: list[str], *,
               seed: int = DEFAULT_SEED, reshuffle: bool = False) -> tuple[list[str], list[str], list[str]]:
    """返回 (dev_ids, test_ids, 警告)。ids 是逐页的 image_id，按单元 key 排序。"""
    warnings: list[str] = []
    units = {k: v for k, v in ds.build_units(manifest).items() if ds.unit_attrs(v)["type"] != "T7"}
    old_split: dict[str, str] = {}
    if not reshuffle:
        for name, ids in (("dev", existing_dev), ("test", existing_test)):
            for iid in ids:
                rec = manifest.get(iid)
                if rec is None:
                    warnings.append(f"{name}.txt 里的 {iid} 已不在 manifest 里，丢弃")
                    continue
                key = ds.unit_key(rec)
                if key in old_split and old_split[key] != name:
                    warnings.append(f"{key} 同时出现在 dev 和 test 里，按 test 处理")
                    old_split[key] = "test"
                else:
                    old_split.setdefault(key, name)
        for key in list(old_split):
            if key not in units:
                warnings.append(f"{key} 现在是 T7 或已删除，从划分里移除")
                del old_split[key]
    strata: dict[tuple, list[str]] = {}
    for key, pages in units.items():
        a = ds.unit_attrs(pages)
        if len(a["types"]) > 1:
            warnings.append(f"{key}：同一组的页面类型不一致 {a['types']}，按 {a['type']} 分层")
        strata.setdefault((a["size_bucket"] or "?", a["type"]), []).append(key)
    assign: dict[str, str] = dict(old_split)
    for skey in sorted(strata):
        keys = sorted(strata[skey])
        new = [k for k in keys if k not in old_split]
        if not new:
            continue
        n_test_old = sum(1 for k in keys if old_split.get(k) == "test")
        need_test = min(len(new), max(0, _target_test(len(keys)) - n_test_old))
        rng = random.Random(f"{seed}:{skey[0]}:{skey[1]}")
        rng.shuffle(new)
        for i, k in enumerate(new):
            assign[k] = "test" if i < need_test else "dev"
    dev_ids: list[str] = []
    test_ids: list[str] = []
    for key in sorted(assign):
        ids = [p["image_id"] for p in units[key]]
        (test_ids if assign[key] == "test" else dev_ids).extend(ids)
    return dev_ids, test_ids, warnings


def write_split(name: str, ids: list[str], header: str) -> None:
    p = formats.DATASETS / f"{name}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".txt.tmp")
    tmp.write_text(header + "".join(f"{i}\n" for i in ids), encoding="utf-8")
    tmp.replace(p)


def run_split(*, seed: int = DEFAULT_SEED, reshuffle: bool = False, dry_run: bool = False):
    from bench.collect.lock import lock_path
    from bench.collect.manifest import CollectError

    if lock_path().exists():
        raise CollectError([f"{lock_path()} 已存在：test 集已锁定，不能重新划分"])
    manifest = formats.load_manifest(formats.MANIFEST)
    if not manifest:
        raise CollectError(["manifest 为空：先用 add 登记图片"])
    dev, test, warnings = plan_split(manifest, formats.read_split("dev"), formats.read_split("test"),
                                     seed=seed, reshuffle=reshuffle)
    if not dry_run:
        units_dev = len(ds.build_units(manifest, dev))
        units_test = len(ds.build_units(manifest, test))
        common = (f"# 由 python -m bench.collect split 生成（seed={seed}；按 size_bucket×type 分层，dev:test=1:2；"
                  "分页图同组不拆开；已有的划分保持不变）\n")
        write_split("dev", dev, common + f"# dev：{units_dev} 个单元，{len(dev)} 张图\n")
        write_split("test", test, common + f"# test：{units_test} 个单元，{len(test)} 张图\n")
    return dev, test, warnings
