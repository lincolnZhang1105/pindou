#!/usr/bin/env bash
# T0.14 真机测速：把 Android arm64 版 beadcore-cli 推到手机上，重复识别并记录耗时。
#
#   tools/android-bench.sh CLI LIBORT [IMG...]
#     CLI    tools/build-android.sh 编出的 beadcore-cli
#     LIBORT ONNX Runtime 的 libonnxruntime.so（arm64-v8a）
#     IMG    测速图纸，默认 build/bench_104.png（由 tools/make_bench_image.py 生成）
#
# 需要 adb 已连上手机（开发者选项里打开 USB 调试）。结果追加到 build/android-bench.md。
set -euo pipefail
cd "$(dirname "$0")/.."
CLI="${1:?缺少 beadcore-cli 路径}"
ORT="${2:?缺少 libonnxruntime.so 路径}"
shift 2
IMGS=("$@")
if [ ${#IMGS[@]} -eq 0 ]; then
  [ -f build/bench_104.png ] || python tools/make_bench_image.py --out build/bench_104.png
  IMGS=(build/bench_104.png)
fi
PAL=bench/palettes_unverified/build/palettes_bench.json
[ -f "$PAL" ] || PAL=palettes/build/palettes.json
DEV=/data/local/tmp/beadcore
REPEAT="${REPEAT:-5}"

export MSYS_NO_PATHCONV=1  # Git Bash：别把 /data/... 改写成 Windows 路径
adb get-state >/dev/null
MODEL="$(adb shell getprop ro.product.model | tr -d '\r')"
SOC="$(adb shell getprop ro.soc.model | tr -d '\r')"
echo "[bench] 设备：$MODEL（$SOC）"
adb shell "rm -rf $DEV && mkdir -p $DEV/models"
adb push "$CLI" "$DEV/beadcore-cli" >/dev/null
adb push "$ORT" "$DEV/libonnxruntime.so" >/dev/null
adb push "$PAL" "$DEV/palettes.json" >/dev/null
for f in core/models/*.onnx core/models/*.json; do adb push "$f" "$DEV/models/" >/dev/null; done
adb shell "chmod 755 $DEV/beadcore-cli"

mkdir -p build
{
  echo ""
  echo "## $(date '+%Y-%m-%d %H:%M') · $MODEL（$SOC）"
  echo ""
  echo "| 图纸 | 网格 | 各次总耗时（ms） | 中位数（ms） | 引擎内各阶段（最后一次，ms） |"
  echo "|---|---|---|---|---|"
} >> build/android-bench.md
for img in "${IMGS[@]}"; do
  name="$(basename "$img")"
  adb push "$img" "$DEV/$name" >/dev/null
  out="$(adb shell "cd $DEV && ORT_DISABLE_TELEMETRY=1 LD_LIBRARY_PATH=. ./beadcore-cli recognize $name --models models --palettes palettes.json --repeat $REPEAT" | tr -d '\r')"
  python - "$name" "$out" <<'PY' | tee -a build/android-bench.md
import json, statistics, sys
name, out = sys.argv[1], sys.argv[2]
j = json.loads(out)
if "error" in j:
    print(f"| {name} | 失败：{j['error']['code']} | | | |")
else:
    ms = [round(x) for x in j.get("cli_wall_ms", [])]
    stages = ", ".join(f"{k} {v}" for k, v in j["timing_ms"].items())
    print(f"| {name} | {j['grid']['cols']}×{j['grid']['rows']} | {ms} | {round(statistics.median(ms)) if ms else '-'} | {stages} |")
PY
done
echo "[bench] 结果已追加到 build/android-bench.md（门槛：中端机 104×104 ≤ 3 s，04 §0.1 第 5 条）"
