"""test 集纪律（AGENTS 规则 3；04 §6）：test.lock 校验 + test_usage.jsonl 使用记录。

格式见 bench/README.md「划分与 test 集锁定」。bench/collect 也会提供 check_lock()；
这里是评测器自己的最小实现，按同一格式校验，避免两个模块互相依赖。
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from bench.common import formats as F

REASONS = ("baseline_w2", "final_w6")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TestDisciplineError(RuntimeError):
    pass


def lock_path() -> Path:
    return F.DATASETS / "test.lock"


def usage_path() -> Path:
    return F.DATASETS / "test_usage.jsonl"


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def check_test_lock(test_ids: list[str], images_dir: Path | None = None,
                    manifest: dict[str, dict] | None = None) -> dict:
    """校验 test.lock。images_dir 给出时，逐张核对图片 sha256。任何不一致都抛 TestDisciplineError。"""
    p = lock_path()
    if not p.exists():
        raise TestDisciplineError(f"没有 {p}：test 集尚未锁定（T0.5b），拒绝在 test 集上运行")
    try:
        lock = F.load_json(p)
    except Exception as e:  # noqa: BLE001
        raise TestDisciplineError(f"test.lock 不是合法 JSON：{e}") from e
    if not isinstance(lock, dict) or not isinstance(lock.get("created_at"), str) \
            or not isinstance(lock.get("ids_sha256"), str) or not isinstance(lock.get("entries"), list):
        raise TestDisciplineError("test.lock 缺少 created_at / ids_sha256 / entries")
    entries: dict[str, str] = {}
    for e in lock["entries"]:
        if not isinstance(e, dict) or not isinstance(e.get("image_id"), str) or not isinstance(e.get("sha256"), str):
            raise TestDisciplineError(f"test.lock 的 entries 里有不合法的项：{e!r}")
        if e["image_id"] in entries:
            raise TestDisciplineError(f"test.lock 里 {e['image_id']} 出现了两次")
        if not _HEX64.match(e["sha256"].lower()):
            raise TestDisciplineError(f"test.lock 里 {e['image_id']} 的 sha256 不合法")
        entries[e["image_id"]] = e["sha256"].lower()
    if not test_ids:
        raise TestDisciplineError("test.txt 为空")
    if len(set(test_ids)) != len(test_ids):
        raise TestDisciplineError("test.txt 里有重复的 image_id")
    if ids_sha256(test_ids) != lock["ids_sha256"].lower():
        raise TestDisciplineError("test.txt 的 image_id 列表与 test.lock 的 ids_sha256 不一致：test 集在锁定后被改动过")
    if set(entries) != set(test_ids):
        diff = sorted(set(entries) ^ set(test_ids))
        raise TestDisciplineError(f"test.lock 的 entries 与 test.txt 不一致：{diff[:10]}")
    bad_m = [iid for iid, sha in sorted(entries.items())
             if ((manifest or {}).get(iid) or {}).get("sha256") and manifest[iid]["sha256"].lower() != sha]
    if bad_m:
        raise TestDisciplineError(f"{len(bad_m)} 张 test 图片在 manifest 里的 sha256 与 test.lock 不一致：{bad_m[:10]}")
    if images_dir is not None:
        from bench.eval.dataset import DatasetError, find_image_by_id
        bad: list[str] = []
        for iid, sha in sorted(entries.items()):
            try:
                path = find_image_by_id(images_dir, iid, manifest or {})
            except DatasetError as e:
                raise TestDisciplineError(f"test 图片缺失：{e}") from e
            if F.sha256_file(path) != sha:
                bad.append(iid)
        if bad:
            raise TestDisciplineError(f"{len(bad)} 张 test 图片的 sha256 与 test.lock 不一致（锁定后被改动）：{bad[:10]}")
    return {"created_at": lock["created_at"], "ids_sha256": lock["ids_sha256"], "entries": entries}


def check_label_hashes(units, lock: dict) -> None:
    """真值里登记的图片 sha256（非空时）必须与 test.lock 一致：防止真值对应的是另一个版本的图片。"""
    bad: list[str] = []
    for u in units:
        for im in u.label["images"]:
            sha = (im.get("sha256") or "").lower()
            iid = Path(im["file"]).stem
            if sha and iid in lock["entries"] and lock["entries"][iid] != sha:
                bad.append(f"{u.image_id}/{iid}")
    if bad:
        raise TestDisciplineError(f"真值里登记的图片 sha256 与 test.lock 不一致：{bad[:10]}")


def usage_records() -> list[dict]:
    return F.read_jsonl(usage_path())


def check_reason_available(reason: str | None) -> None:
    if reason not in REASONS:
        raise TestDisciplineError(f"--split test 必须带 --test-reason，取值只能是 {'/'.join(REASONS)}（实际：{reason!r}）")
    used = [r for r in usage_records() if r.get("reason") == reason]
    if used:
        r = used[0]
        raise TestDisciplineError(f"test 集的 {reason} 已经在 {r.get('at')} 用过（run={r.get('run')}），"
                                  "每个 reason 只能用一次（04 §6 纪律），拒绝运行")


def git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=F.REPO, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            rev = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=F.REPO, capture_output=True,
                                   text=True, timeout=30)
            return rev + ("-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_usage(reason: str, run: str, extra: dict | None = None) -> dict:
    """追加一条使用记录（README 规定的 at/reason/run/git_rev 四个字段，另加评测器自己的字段）。"""
    check_reason_available(reason)  # 追加前再查一次
    rec = {"at": now_iso(), "reason": reason, "run": run, "git_rev": git_rev()}
    rec.update(extra or {})
    p = usage_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def find_usage(reason: str, run: str) -> dict | None:
    for r in usage_records():
        if r.get("reason") == reason and r.get("run") == run:
            return r
    return None
