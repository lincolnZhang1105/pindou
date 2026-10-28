"""合成渲染用的字体：下载、校验、加载。

- 字体二进制不进 git（`.gitignore` 里的 `/bench/synth/fonts/`），用本模块按锁文件下载：
  `python -m bench.synth fonts`（或 `python -m bench.synth.fonts`）。
- 锁文件 `bench/synth/fonts.lock.json` 记录每个文件的 URL（钉在 google/fonts 的某个 commit）、sha256 和许可证。
  **只允许 OFL-1.1 / Apache-2.0 这类允许商用和嵌入的字体**（05 §7），加载时再检查一遍。
- 新增字体：在锁文件里加一项（sha256 先留空），跑 `python -m bench.synth fonts --update-lock` 写入 sha256，
  再到 05 §7 登记。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
LOCK = HERE / "fonts.lock.json"
FONT_DIR = HERE / "fonts"

ALLOWED_LICENSES = ("OFL-1.1", "Apache-2.0")
WEIGHTS = {"light": 300, "regular": 400, "medium": 500, "semibold": 600, "bold": 700}


@dataclass(frozen=True)
class FontFace:
    """一个可用的字体文件。style：sans / mono / rounded / condensed / cjk / cjk_display。"""
    id: str
    family: str
    path: str
    style: str
    roles: tuple[str, ...]
    variable: bool
    weights: tuple[str, ...]          # 可用的字重名（静态字体每个文件一个字重）
    license: str
    cjk: bool = False
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def _download(url: str, dst: Path, quiet: bool) -> None:
    if not quiet:
        print(f"下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "pindou-bench-synth"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)


def fetch(update_lock: bool = False, quiet: bool = False) -> int:
    """按锁文件下载字体和许可证文本。sha256 不符时报错（--update-lock 时改为写入锁文件）。返回错误数。"""
    lock = load_lock()
    errors = 0
    for e in lock["fonts"]:
        if e["license"] not in ALLOWED_LICENSES:
            print(f"[拒绝] {e['id']}：许可证 {e['license']} 不在白名单 {ALLOWED_LICENSES}", file=sys.stderr)
            errors += 1
            continue
        for item in [*e["files"], e["license_file"]]:
            dst = FONT_DIR / e["id"] / item["name"]
            if not dst.exists() or (item.get("sha256") and not update_lock and _sha256(dst) != item["sha256"]):
                _download(item["url"], dst, quiet)
            got = _sha256(dst)
            if not item.get("sha256") or update_lock:
                item["sha256"] = got
            elif got != item["sha256"]:
                print(f"[sha256 不符] {dst}：锁文件 {item['sha256']}，实际 {got}", file=sys.stderr)
                errors += 1
    if update_lock:
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if not quiet:
            print(f"已更新 {LOCK}")
    if not quiet:
        print(f"完成：{len(lock['fonts'])} 个字体，错误 {errors} 个")
    return errors


@lru_cache(maxsize=1)
def available_faces() -> tuple[FontFace, ...]:
    """锁文件里已下载、且所有文件 sha256 校验通过的字体（按 id 排序，保证确定性）。"""
    if not LOCK.exists():
        return ()
    faces = []
    for e in sorted(load_lock()["fonts"], key=lambda x: x["id"]):
        if e["license"] not in ALLOWED_LICENSES:
            continue
        ok = True
        for item in e["files"]:
            p = FONT_DIR / e["id"] / item["name"]
            if not p.exists() or not item.get("sha256") or _sha256(p) != item["sha256"]:
                ok = False
        if not ok:
            continue
        faces.append(FontFace(id=e["id"], family=e["family"], path=str(FONT_DIR / e["id"] / e["files"][0]["name"]),
                              style=e["style"], roles=tuple(e["roles"]), variable=bool(e.get("variable")),
                              weights=tuple(e.get("weights", ["regular"])), license=e["license"],
                              cjk=bool(e.get("cjk")), meta=e))
    return tuple(faces)


class FontBook:
    """按角色挑字体、按（字体, 字重, 字号）缓存 FreeTypeFont。

    角色：cell（格内色号）、coord（坐标）、legend（图例）、title（标题，需要 CJK）、watermark（水印，需要 CJK）。
    没有下载字体且 allow_fallback=True 时，退回 Pillow 内置的 Aileron（CC0），只用于测试。
    """

    def __init__(self, allow_fallback: bool = False):
        self.faces = available_faces()
        self.fallback = not self.faces
        if self.fallback and not allow_fallback:
            raise SystemExit("没有可用字体：先运行 `python -m bench.synth fonts` 下载（见 bench/synth/README.md）")
        self._cache: dict[tuple, ImageFont.FreeTypeFont] = {}

    def candidates(self, role: str) -> list[FontFace]:
        return [f for f in self.faces if role in f.roles]

    def pick(self, rng, role: str) -> FontFace | None:
        c = self.candidates(role)
        if not c:
            return None
        return c[int(rng.integers(len(c)))]

    def pick_weight(self, rng, face: FontFace | None, prefer: tuple[str, ...] = ("regular", "medium", "bold")) -> str:
        if face is None:
            return "regular"
        ok = [w for w in prefer if w in face.weights] or list(face.weights)
        return ok[int(rng.integers(len(ok)))]

    def get(self, face: FontFace | None, size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
        size = max(4, int(size))
        key = (face.id if face else None, weight, size)
        f = self._cache.get(key)
        if f is not None:
            return f
        if face is None:
            f = ImageFont.load_default(size)
        else:
            path = face.path
            if not face.variable:
                # 静态字体：每个字重一个文件
                for item in face.meta["files"]:
                    if item.get("weight") == weight:
                        path = str(FONT_DIR / face.id / item["name"])
            f = ImageFont.truetype(path, size)
            if face.variable:
                _set_weight(f, WEIGHTS.get(weight, 400))
        self._cache[key] = f
        return f

    def describe(self, face: FontFace | None, weight: str) -> str:
        return f"{face.id}:{weight}" if face else f"pillow-default:{weight}"


def _set_weight(font: ImageFont.FreeTypeFont, wght: int) -> None:
    axes = font.get_variation_axes()
    vals = []
    for ax in axes:
        name = ax.get("name")
        name = name.decode() if isinstance(name, bytes) else str(name)
        if name.lower().startswith("weight") or name.lower() == "wght":
            vals.append(float(min(max(wght, ax["minimum"]), ax["maximum"])))
        else:
            vals.append(float(ax["default"]))
    font.set_variation_by_axes(vals)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.synth fonts", description="按锁文件下载合成渲染用的字体")
    ap.add_argument("--update-lock", action="store_true", help="把下载文件的 sha256 写回锁文件（新增字体时用）")
    a = ap.parse_args(argv)
    return 1 if fetch(update_lock=a.update_lock) else 0


if __name__ == "__main__":
    sys.exit(main())
