#!/usr/bin/env python3
"""下载并校验 beadcore 的第三方依赖，再按需编译精简版 OpenCV。

用法：
    python core/scripts/fetch_deps.py                    # 下载全部依赖源码，校验 sha256
    python core/scripts/fetch_deps.py --build-opencv     # 另外编译 OpenCV（只含 core、imgproc）
    python core/scripts/fetch_deps.py --update-lock      # 首次引入依赖时，把 sha256 写回 deps.json

目录约定（都在仓库根目录的 .deps/ 下，已被 .gitignore 忽略）：
    .deps/dl/<file>                          下载缓存
    .deps/src/<name>/                        解压后的源码
    .deps/<platform>/opencv/                 编译安装好的 OpenCV（platform 见 --platform）

OpenCV 只编 core 和 imgproc，关闭 videoio / highgui / imgcodecs（03 §1）：
不引入 FFmpeg（LGPL），图片解码交给平台层；桌面 CLI 另用 stb_image + libwebp 解码。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEPS_JSON = REPO / "core" / "deps.json"
DEPS_DIR = Path(os.environ.get("BC_DEPS_DIR", REPO / ".deps"))

OPENCV_FLAGS = [
    "-DBUILD_LIST=core,imgproc",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_DOCS=OFF",
    "-DBUILD_opencv_apps=OFF",
    "-DBUILD_JAVA=OFF",
    "-DBUILD_opencv_python2=OFF",
    "-DBUILD_opencv_python3=OFF",
    "-DBUILD_ZLIB=ON",
    "-DWITH_IPP=OFF",          # ippicv 是 Intel 的单独许可，且只有 x86，关掉以保证各平台结果一致
    "-DWITH_ITT=OFF",
    "-DWITH_OPENCL=OFF",
    "-DWITH_CUDA=OFF",
    "-DWITH_EIGEN=OFF",
    "-DWITH_LAPACK=OFF",
    "-DWITH_PROTOBUF=OFF",
    "-DWITH_FFMPEG=OFF",
    "-DWITH_GSTREAMER=OFF",
    "-DWITH_ADE=OFF",
    "-DWITH_OPENMP=OFF",
    "-DWITH_TBB=OFF",
    "-DWITH_JPEG=OFF",
    "-DWITH_PNG=OFF",
    "-DWITH_WEBP=OFF",
    "-DWITH_TIFF=OFF",
    "-DWITH_OPENJPEG=OFF",
    "-DWITH_JASPER=OFF",
    "-DWITH_OPENEXR=OFF",
    "-DWITH_IMGCODEC_HDR=OFF",
    "-DWITH_IMGCODEC_SUNRASTER=OFF",
    "-DWITH_IMGCODEC_PXM=OFF",
    "-DWITH_IMGCODEC_PFM=OFF",
    "-DWITH_QT=OFF",
    "-DWITH_GTK=OFF",
    "-DWITH_WIN32UI=OFF",
    "-DWITH_VTK=OFF",
    "-DOPENCV_ENABLE_NONFREE=OFF",
    "-DCV_TRACE=OFF",
    "-DOPENCV_GENERATE_PKGCONFIG=OFF",
    "-DBUILD_WITH_STATIC_CRT=OFF",  # MSVC：和 beadcore 一样用 /MD，否则链接时运行库不匹配
]


def default_platform() -> str:
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    if sysname == "windows":
        return f"windows-{arch}-msvc"
    if sysname == "darwin":
        return f"macos-{arch}"
    return f"linux-{arch}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[deps] 下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "beadcore-fetch-deps"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def extract(archive: Path, dest: Path) -> None:
    """解压到 dest，并去掉压缩包里唯一的顶层目录。"""
    if dest.exists():
        return
    staging = dest.with_name(dest.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(staging, filter="data")
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(staging)
    else:
        raise ValueError(f"不认识的压缩格式：{archive}")
    entries = list(staging.iterdir())
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
    shutil.move(str(root), str(dest))
    if staging.exists():
        shutil.rmtree(staging)


def fetch_all(update_lock: bool, plat: str) -> dict:
    lock = json.loads(DEPS_JSON.read_text(encoding="utf-8"))
    changed = False
    for name, dep in lock["deps"].items():
        # 预编译二进制按平台区分：只下载当前平台的，解压到 .deps/<platform>/<name>
        if "platforms" in dep:
            if plat not in dep["platforms"]:
                print(f"[deps] {name} 没有 {plat} 平台的包，跳过")
                continue
            d = dep["platforms"][plat]
            d.setdefault("version", dep["version"])
            target = DEPS_DIR / plat / name
        else:
            d = dep
            target = DEPS_DIR / "src" / name
        url = d["url"]
        fname = f"{name}-{d['version'][:12]}-{url.rsplit('/', 1)[-1]}"
        dl = DEPS_DIR / "dl" / fname
        download(url, dl)
        got = sha256_of(dl)
        if not d.get("sha256"):
            if not update_lock:
                sys.exit(f"[deps] {name} 在 deps.json 里没有 sha256。首次引入请加 --update-lock，并在 docs/05 §6 登记。")
            d["sha256"] = got
            changed = True
            print(f"[deps] 记录 {name} sha256={got}")
        elif d["sha256"] != got:
            sys.exit(f"[deps] {name} 的 sha256 不匹配：期望 {d['sha256']}，实际 {got}")
        if "file" in d:
            out = DEPS_DIR / "src" / "include" / d["file"]
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                shutil.copyfile(dl, out)
        else:
            extract(dl, target)
    if changed:
        DEPS_JSON.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lock


def build_opencv(plat: str, generator: str | None, extra: list[str]) -> Path:
    src = DEPS_DIR / "src" / "opencv"
    build = DEPS_DIR / plat / "opencv-build"
    install = DEPS_DIR / plat / "opencv"
    if (install / ".done").exists():
        print(f"[deps] OpenCV 已编译：{install}")
        return install
    cfg = ["cmake", "-S", str(src), "-B", str(build), f"-DCMAKE_INSTALL_PREFIX={install}",
           "-DCMAKE_BUILD_TYPE=Release", *OPENCV_FLAGS, *extra]
    if generator:
        cfg += ["-G", generator]
    print("[deps] " + " ".join(cfg))
    subprocess.check_call(cfg)
    subprocess.check_call(["cmake", "--build", str(build), "--config", "Release", "--parallel"])
    subprocess.check_call(["cmake", "--install", str(build), "--config", "Release"])
    (install / ".done").write_text("ok\n")
    return install


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-lock", action="store_true", help="把缺失的 sha256 写回 deps.json")
    ap.add_argument("--build-opencv", action="store_true")
    ap.add_argument("--platform", default=default_platform(), help="平台标签，决定安装目录，默认自动判断")
    ap.add_argument("--generator", default="Ninja" if shutil.which("ninja") else None)
    ap.add_argument("--opencv-arg", action="append", default=[], help="额外传给 OpenCV 的 cmake 参数（交叉编译用）")
    args = ap.parse_args()
    fetch_all(args.update_lock, args.platform)
    if args.build_opencv:
        install = build_opencv(args.platform, args.generator, args.opencv_arg)
        print(f"[deps] OpenCV_DIR 候选：{install}")


if __name__ == "__main__":
    main()
