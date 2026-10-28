"""test 集锁定（T0.5b）。格式见 bench/README.md「划分与 test 集锁定」。

评测器在使用 test 集之前调用 `check_lock()`：

    from bench.collect import check_lock
    errs = check_lock()
    if errs: raise SystemExit("\\n".join(errs))
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from bench.common import dataset as ds
from bench.common import formats


def lock_path() -> Path:
    return formats.DATASETS / "test.lock"


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def check_lock(check_files: bool = True, root: Path | None = None) -> list[str]:
    """校验 test 集没有被改动。返回错误列表，空列表表示通过。

    - test.lock 必须存在；
    - test.txt 的 image_id 集合与 test.lock 一致（ids_sha256 和 entries 都核对）；
    - manifest 里每张 test 图的 sha256 与 test.lock 一致；
    - check_files=True 且设置了 PINDOU_DATA_DIR（或传了 root）时，再核对图片文件本身的 sha256。
    """
    p = lock_path()
    if not p.exists():
        return [f"{p} 不存在：test 集还没有锁定（T0.5b）"]
    try:
        lock = formats.load_json(p)
    except ValueError as e:
        return [f"test.lock 不是合法 JSON：{e}"]
    errs: list[str] = []
    test_ids = formats.read_split("test")
    if not test_ids:
        errs.append("test.txt 为空或不存在")
    if len(set(test_ids)) != len(test_ids):
        errs.append("test.txt 里有重复的 image_id")
    if ids_sha256(set(test_ids)) != lock.get("ids_sha256"):
        errs.append("test.txt 与 test.lock 的 ids_sha256 不一致：test 集的成员被改动过")
    entries = {e["image_id"]: e["sha256"] for e in lock.get("entries", [])}
    if ids_sha256(entries) != lock.get("ids_sha256"):
        errs.append("test.lock 自身不一致：entries 与 ids_sha256 对不上")
    missing = sorted(set(entries) - set(test_ids))
    extra = sorted(set(test_ids) - set(entries))
    if missing:
        errs.append(f"test.txt 缺少已锁定的图片：{missing[:10]}")
    if extra:
        errs.append(f"test.txt 多出未锁定的图片：{extra[:10]}")
    manifest = formats.load_manifest(formats.MANIFEST)
    for iid, sha in sorted(entries.items()):
        rec = manifest.get(iid)
        if rec is None:
            errs.append(f"{iid}：不在 manifest 里")
        elif rec.get("sha256") != sha:
            errs.append(f"{iid}：manifest 的 sha256 与 test.lock 不一致")
    if check_files:
        if root is None:
            root = formats.data_dir(required=False)
        if root is not None:
            for iid, sha in sorted(entries.items()):
                rec = manifest.get(iid)
                if rec is None:
                    continue
                f = ds.image_path(rec, root)
                if not f.is_file():
                    errs.append(f"{iid}：图片不存在 {f}")
                elif formats.sha256_file(f) != sha:
                    errs.append(f"{iid}：图片文件的 sha256 与 test.lock 不一致")
    return errs


def lock_test(root: Path | None = None, *, allow_unlabeled: bool = False) -> tuple[dict, list[str]]:
    """生成 test.lock。test.lock 已存在时拒绝（锁定后不得修改）。返回 (lock, 警告)。"""
    from bench.collect.manifest import CollectError

    p = lock_path()
    if p.exists():
        raise CollectError([f"{p} 已存在：test 集锁定后不得修改"])
    root = Path(root) if root is not None else formats.data_dir()
    test_ids = formats.read_split("test")
    if not test_ids:
        raise CollectError(["test.txt 为空或不存在：先运行 split"])
    if len(set(test_ids)) != len(test_ids):
        raise CollectError(["test.txt 里有重复的 image_id"])
    manifest = formats.load_manifest(formats.MANIFEST)
    errs: list[str] = []
    warnings: list[str] = []
    entries = []
    for iid in sorted(test_ids):
        rec = manifest.get(iid)
        if rec is None:
            errs.append(f"{iid}：不在 manifest 里")
            continue
        f = ds.image_path(rec, root)
        if not f.is_file():
            errs.append(f"{iid}：图片不存在 {f}")
        elif formats.sha256_file(f) != rec["sha256"]:
            errs.append(f"{iid}：图片文件与 manifest 的 sha256 不一致")
        entries.append({"image_id": iid, "sha256": rec["sha256"]})
    # 真值检查：T0.5b 要求 test 集全部标完、通过图例校验后再锁定
    units = ds.build_units(manifest, test_ids)
    label_problems = []
    for key in units:
        lp = ds.label_path(key, root)
        if not lp.is_file():
            label_problems.append(f"{key}：还没有真值 {lp.name}")
            continue
        try:
            lab = formats.load_json(lp)
        except ValueError as e:
            label_problems.append(f"{key}：真值不是合法 JSON：{e}")
            continue
        verrs = formats.validate_label(lab)
        if verrs:
            label_problems.append(f"{key}：真值校验不通过：{verrs[0]}")
    if label_problems:
        if allow_unlabeled:
            warnings.extend(label_problems)
        else:
            errs.extend(label_problems)
            errs.append("（确实要在标完之前锁定，加 --allow-unlabeled）")
    if errs:
        raise CollectError(errs)
    lock = {"created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "ids_sha256": ids_sha256(test_ids), "entries": entries}
    formats.save_json(p, lock)
    return lock, warnings
