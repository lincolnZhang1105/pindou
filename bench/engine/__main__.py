"""用 beadcore-cli 批量跑识别，产出评测器要的预测结果目录（T0.8 / T0.10 / T0.12 共用）。

    python -m bench.engine run --split dev --run beadcore-v0 [--cli PATH] [--palettes FILE] [--jobs N]
    python -m bench.engine run --ids xhs_6aa72c3e_0 ... --run debug

约定（见 bench/eval/README.md）：
- 输出 `$PINDOU_DATA_DIR/preds/<run>/<单元 key>.json`：普通图纸 key = image_id；分页图（T4）key = group id，
  各页按页码顺序一起传给 CLI（多页合成由引擎负责，B9）。
- CLI 报错时照样写 `{"error": ...}`，评测器按失败计分；CLI 崩溃或超时也写成 error JSON（code = internal）。
- 另写 `_run.json`：引擎版本、git 版本、CLI 路径、色卡文件 sha256、逐单元耗时。
- 跑 test 集必须加 `--allow-test`。这里只产出预测结果；test 集的使用登记由 `bench.eval run --test-reason` 负责。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.common import dataset as D
from bench.common import formats as F

REPO = F.REPO
DEFAULT_PALETTES = REPO / "bench" / "palettes_unverified" / "build" / "palettes_bench.json"
DEFAULT_MODELS = REPO / "core" / "models"


def default_cli() -> Path | None:
    env = os.environ.get("BC_CLI")
    if env:
        return Path(env)
    for p in (REPO / "build" / "win" / "beadcore-cli.exe", REPO / "build" / "linux" / "beadcore-cli",
              REPO / "build" / "macos" / "beadcore-cli"):
        if p.exists():
            return p
    return None


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_unit(cli: Path, palettes: Path, key: str, pages: list[dict], root: Path, timeout: float,
             models: Path | None = None) -> tuple[dict, float]:
    attrs = D.unit_attrs(pages)
    args = [str(cli), "recognize", "--palettes", str(palettes)]
    if models is not None:
        args += ["--models", str(models)]
    if attrs.get("palette_hint"):
        args += ["--hint", attrs["palette_hint"]]
    for rec in pages:
        p = D.readable_image_path(rec, root)
        if not p.exists():
            return {"error": {"code": "invalid_input", "message": f"缺少图片 {p}"}}, 0.0
        args.append(str(p))
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(args, capture_output=True, timeout=timeout)
        dt = (time.perf_counter() - t0) * 1000
        out = cp.stdout.decode("utf-8", errors="replace").strip()
        try:
            return json.loads(out), dt
        except json.JSONDecodeError:
            msg = f"CLI 退出码 {cp.returncode}，输出不是 JSON：{out[:200]} {cp.stderr.decode('utf-8', 'replace')[:200]}"
            return {"error": {"code": "internal", "message": msg}}, dt
    except subprocess.TimeoutExpired:
        return {"error": {"code": "internal", "message": f"超时（>{timeout}s）"}}, timeout * 1000


def cmd_run(a: argparse.Namespace) -> int:
    root = F.data_dir()
    cli = Path(a.cli) if a.cli else default_cli()
    if cli is None or not cli.exists():
        print("找不到 beadcore-cli，用 --cli 指定或设置 BC_CLI（先编译 core）", file=sys.stderr)
        return 2
    palettes = Path(a.palettes)
    if not palettes.exists():
        print(f"色卡文件不存在：{palettes}（先运行 python -m bench.palettes_unverified.import_unverified）",
              file=sys.stderr)
        return 2
    if not D.valid_id(a.run):
        print(f"run 名不合法：{a.run}", file=sys.stderr)
        return 2
    manifest = F.load_manifest(Path(a.manifest)) if a.manifest else F.load_manifest()
    if a.ids:
        ids = a.ids
    else:
        if a.split == "test" and not a.allow_test:
            print("跑 test 集要加 --allow-test（test 集只在 W2 baseline 和 W6 最终评测时使用）", file=sys.stderr)
            return 2
        ids = F.read_split(a.split)
        if not ids:
            print(f"bench/datasets/{a.split}.txt 为空或不存在", file=sys.stderr)
            return 2
    units = D.build_units(manifest, ids)
    missing = [i for i in ids if i not in manifest]
    if missing:
        print(f"manifest 里没有这些 image_id：{missing[:10]}", file=sys.stderr)
        return 2
    models = None if a.no_models else Path(a.models)
    if models is not None and not (models / "ocr_v1.onnx").exists():
        print(f"提示：{models} 下没有 OCR 模型，只走颜色路径", file=sys.stderr)
    out_dir = D.preds_dir(root) / a.run
    if out_dir.exists() and any(out_dir.glob("*.json")) and not a.overwrite:
        print(f"{out_dir} 已有结果，加 --overwrite 覆盖", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    def work(item):
        key, pages = item
        res, dt = run_unit(cli, palettes, key, pages, root, a.timeout, models)
        F.save_json(out_dir / f"{key}.json", res, indent=None)
        return key, dt, res.get("error", {}).get("code")

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        results = list(ex.map(work, units.items()))
    failed = {k: c for k, _, c in results if c}
    meta = {
        "run": a.run,
        "split": a.split if not a.ids else None,
        "git_rev": git_rev(),
        "cli": str(cli),
        "engine_version": json.loads(subprocess.run([str(cli), "version"], capture_output=True, text=True).stdout)
        .get("engine_version"),
        "palettes": str(palettes),
        "models": str(models) if models else None,
        "palettes_sha256": F.sha256_file(palettes),
        "units": len(results),
        "failed": failed,
        "wall_ms": {k: round(dt, 1) for k, dt, _ in results},
    }
    F.save_json(out_dir / "_run.json", meta)
    print(f"{len(results)} 个单元 → {out_dir}；失败 {len(failed)} 个")
    for k, c in sorted(failed.items()):
        print(f"  {k}: {c}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.engine", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="跑一个划分，写预测结果")
    r.add_argument("--split", choices=("dev", "test"), default="dev")
    r.add_argument("--ids", nargs="+", help="只跑这些 image_id（调试用，不受划分限制）")
    r.add_argument("--run", required=True, help="run 名，即 preds/ 下的目录名")
    r.add_argument("--cli", help="beadcore-cli 路径（默认 $BC_CLI 或 build/ 下的产物）")
    r.add_argument("--palettes", default=str(DEFAULT_PALETTES))
    r.add_argument("--models", default=str(DEFAULT_MODELS), help="模型目录（默认 core/models）")
    r.add_argument("--no-models", action="store_true", help="不加载模型，只走颜色路径")
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    r.add_argument("--timeout", type=float, default=60.0)
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--allow-test", action="store_true")
    r.add_argument("--manifest", help="manifest 路径（默认 bench/datasets/manifest.jsonl）")
    a = ap.parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
