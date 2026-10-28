"""python -m bench.collect <子命令>：add / verify / stats / split / lock-test / resolution / convert。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.common import formats


def _print_list(title: str, items: list[str], stream=None) -> None:
    if items:
        stream = stream or sys.stdout
        print(title, file=stream)
        for x in items:
            print(f"  - {x}", file=stream)


def cmd_add(a) -> int:
    from bench.collect.imageinfo import IMAGE_SUFFIXES
    from bench.collect.manifest import CollectError, add_images, list_inbox, read_sidecar

    root = formats.data_dir()
    defaults = {k: v for k, v in {
        "platform": a.platform, "note_id": a.note_id, "author": a.author, "url": a.url, "saved_at": a.saved_at,
        "source_path": a.source_path, "type": a.type, "size_bucket": a.size_bucket, "grid": a.grid,
        "palette_hint": a.palette_hint, "tags": a.tags, "group": a.group, "notes": a.notes, "pair_of": a.pair_of,
    }.items() if v is not None}
    try:
        if a.csv:
            if a.paths or a.id:
                raise CollectError(["--csv 不能和图片路径 / --id 同时使用"])
            entries = [(p, {**defaults, **m}) for p, m in read_sidecar(Path(a.csv))]
        else:
            paths: list[Path] = []
            for s in a.paths:
                p = Path(s)
                if p.is_dir():
                    paths.extend(sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES))
                else:
                    paths.append(p)
            if not a.paths:
                paths = list_inbox(root)
                if not paths:
                    raise CollectError([f"{root / 'inbox'} 里没有图片（或给出图片路径 / --csv）"])
            if a.id and len(paths) != 1:
                raise CollectError(["--id 只能用于登记单张图片"])
            entries = [(p, {**defaults, **({"image_id": a.id} if a.id else {})}) for p in paths]
        if not entries:
            raise CollectError(["没有要登记的图片"])
        new, warnings = add_images(entries, root, move=a.move, dry_run=a.dry_run)
    except CollectError as e:
        _print_list("登记失败（没有做任何改动）：", e.errors, sys.stderr)
        return 1
    _print_list("警告：", warnings)
    verb = "将登记" if a.dry_run else "已登记"
    print(f"{verb} {len(new)} 张：")
    for r in new:
        print(f"  {r['image_id']}  {r['width']}×{r['height']} {r['format']}  {r['type']}/{r['size_bucket']}"
              f"{'  group=' + r['group'] if r.get('group') else ''}")
    return 0


def cmd_verify(a) -> int:
    from bench.collect.manifest import verify

    errors, warnings = verify(check_header=not a.no_header)
    _print_list("警告：", warnings)
    if errors:
        _print_list(f"发现 {len(errors)} 个问题：", errors, sys.stderr)
        return 1
    n = len(formats.read_jsonl(formats.MANIFEST))
    print(f"OK：manifest {n} 条，图片和 sha256 全部一致")
    return 0


def cmd_stats(a) -> int:
    from bench.collect import stats

    manifest = formats.load_manifest(formats.MANIFEST)
    text, _ = stats.render(stats.compute(manifest, formats.read_split("dev"), formats.read_split("test")))
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"已写入 {a.out}")
    else:
        print(text)
    return 0


def cmd_split(a) -> int:
    from bench.collect.manifest import CollectError
    from bench.collect.split import run_split

    if a.reshuffle:
        print("注意：--reshuffle 会丢弃已有划分。如果已经在 dev 集上调过参，重新划分会污染 test 集。", file=sys.stderr)
    try:
        dev, test, warnings = run_split(seed=a.seed, reshuffle=a.reshuffle, dry_run=a.dry_run)
    except CollectError as e:
        _print_list("划分失败：", e.errors, sys.stderr)
        return 1
    _print_list("警告：", warnings)
    print(f"{'（试运行，未写文件）' if a.dry_run else ''}dev {len(dev)} 张，test {len(test)} 张")
    return 0


def cmd_lock(a) -> int:
    from bench.collect.lock import lock_path, lock_test
    from bench.collect.manifest import CollectError

    try:
        lock, warnings = lock_test(allow_unlabeled=a.allow_unlabeled)
    except CollectError as e:
        _print_list("锁定失败：", e.errors, sys.stderr)
        return 1
    _print_list("警告：", warnings)
    print(f"已写入 {lock_path()}：{len(lock['entries'])} 张，ids_sha256={lock['ids_sha256']}")
    print("提交 test.lock 之后不得再修改 test.txt（04 §6 纪律）")
    return 0


def cmd_resolution(a) -> int:
    from bench.collect.resolution import report

    text, warnings = report(formats.load_manifest(formats.MANIFEST), formats.data_dir(required=False))
    _print_list("警告：", warnings, sys.stderr)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"已写入 {a.out}")
    else:
        print(text)
    return 0


def cmd_convert(a) -> int:
    from bench.collect.convert import convert

    done, skipped, errors = convert(a.ids or None, force=a.force)
    for x in done:
        print(x)
    _print_list("跳过：", skipped)
    if errors:
        _print_list("失败：", errors, sys.stderr)
        return 1
    if not done and not skipped:
        print("没有需要转换的 HEIC/HEIF/AVIF 图片")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bench.collect",
                                 description="benchmark 样本的 manifest 工具（T0.3）。只处理人工保存的图片，不抓取。")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="登记图片：原字节复制到 images/，追加到 manifest")
    p.add_argument("paths", nargs="*", help="图片文件或目录；省略时登记 $PINDOU_DATA_DIR/inbox/ 下的全部图片")
    p.add_argument("--csv", help="sidecar CSV（批量登记，每行一张图，必须有 file 列）")
    p.add_argument("--id", help="手动指定 image_id（只能登记单张）")
    p.add_argument("--platform", help="xhs / douyin / own / photo / other")
    p.add_argument("--note-id", dest="note_id")
    p.add_argument("--author")
    p.add_argument("--url")
    p.add_argument("--saved-at", dest="saved_at", help="ISO 8601 时间；写 mtime 表示取文件修改时间")
    p.add_argument("--source-path", dest="source_path", help="app_save / own_export / photo / web（Web 端版本）")
    p.add_argument("--type", help="T1–T7")
    p.add_argument("--size-bucket", dest="size_bucket", help="S / M / L / XL")
    p.add_argument("--grid", help="行列数（列x行，如 58x53），自动推算 size_bucket")
    p.add_argument("--palette-hint", dest="palette_hint", help="如 MARD-221")
    p.add_argument("--tags", help="难点标签，逗号分隔")
    p.add_argument("--group", help="分页图（T4）的组 id；T4 不填时默认 <platform>_<note_id 前 8 位>")
    p.add_argument("--notes")
    p.add_argument("--pair-of", dest="pair_of", help="Web 端版本对应的 App 保存版本 image_id（分辨率对比用）")
    p.add_argument("--move", action="store_true", help="登记成功后把源文件移到同目录的 _registered/")
    p.add_argument("--dry-run", action="store_true", help="只校验、不复制不写入")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("verify", help="核对 manifest、images/、划分和 test.lock")
    p.add_argument("--no-header", action="store_true", help="不重新读文件头核对尺寸")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("stats", help="对照 04 §2.1 的覆盖要求输出统计和警告")
    p.add_argument("--out", help="写到 Markdown 文件")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("split", help="分层划分 dev/test（1:2，增量、确定性）")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reshuffle", action="store_true", help="丢弃已有划分重新分（慎用）")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("lock-test", help="写 bench/datasets/test.lock（T0.5b，之后不得修改）")
    p.add_argument("--allow-unlabeled", action="store_true", help="允许在 test 集标完之前锁定")
    p.set_defaults(func=cmd_lock)

    p = sub.add_parser("resolution", help="App 保存 vs Web 端分辨率对比表")
    p.add_argument("--out", help="写到 Markdown 文件")
    p.set_defaults(func=cmd_resolution)

    p = sub.add_parser("convert", help="HEIC/HEIF/AVIF 无损转 PNG 到 cache/decoded/")
    p.add_argument("ids", nargs="*", help="只转换这些 image_id（默认全部）")
    p.add_argument("--force", action="store_true", help="已存在也重新转换")
    p.set_defaults(func=cmd_convert)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "split" and a.seed is None:
        from bench.collect.split import DEFAULT_SEED
        a.seed = DEFAULT_SEED
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
