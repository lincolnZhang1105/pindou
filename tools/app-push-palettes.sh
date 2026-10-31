#!/usr/bin/env bash
# 把 bench 用的测试色卡（含 MARD 等来源存疑的国内品牌）推到手机上原型 App 的专属目录。
# 只用于 PO 自己的手机测试；这份色卡不进安装包（AGENTS 规则 1、07「例外：安卓原型」）。
# 推送后重启 App 生效；删掉手机上的这个文件就回到内置色卡。
set -euo pipefail
cd "$(dirname "$0")/.."
PKG="${PKG:-dev.pindou.proto}"
SRC=bench/palettes_unverified/build/palettes_bench.json
[ -f "$SRC" ] || PYTHONUTF8=1 .venv/Scripts/python -m bench.palettes_unverified.import_unverified
export MSYS_NO_PATHCONV=1
adb get-state >/dev/null
adb shell mkdir -p "/sdcard/Android/data/$PKG/files"
adb push "$SRC" "/sdcard/Android/data/$PKG/files/palettes.json"
adb shell am force-stop "$PKG" || true
echo "[app] 已推送测试色卡；重新打开 App 后底部会显示「测试色卡」"
