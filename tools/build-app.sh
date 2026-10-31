#!/usr/bin/env bash
# 打包安卓原型 APK（07「例外：安卓原型」），在 Windows 的 Git Bash 里运行：
#   tools/build-app.sh                              # 产物：app/build/app/outputs/flutter-apk/app-release.apk（只有 arm64）
#   APP_ABIS="arm64-v8a x86_64" tools/build-app.sh  # 另带 x86_64，在电脑上的安卓模拟器里测（先 ABI=x86_64 tools/build-android.sh）
# 前置：
#   1. WSL 里先跑过 tools/build-android.sh（产出 libbeadcore.so；ONNX Runtime 的 libonnxruntime.so 由它下载）
#   2. Flutter SDK：环境变量 FLUTTER 指向 flutter.bat（默认 D:/AIworkspace/_tools/flutter/bin/flutter.bat）
# 拷进 app/ 的原生库、模型、色卡、测速图都在 gitignore 里，不进仓库。
set -euo pipefail
cd "$(dirname "$0")/.."
WSL_HOME="${WSL_HOME:-//wsl.localhost/Ubuntu-24.04/home/dministrator}"
FLUTTER="${FLUTTER:-/d/AIworkspace/_tools/flutter/bin/flutter.bat}"
PY=".venv/Scripts/python"
export PYTHONUTF8=1

APP_ABIS="${APP_ABIS:-arm64-v8a}"
rm -rf app/android/app/src/main/jniLibs
for abi in $APP_ABIS; do
  JNI="app/android/app/src/main/jniLibs/$abi"
  build="$WSL_HOME/pindou-build/android-$abi"
  [ "$abi" = arm64-v8a ] && build="$WSL_HOME/pindou-build/android-arm64"
  mkdir -p "$JNI"
  cp "$build/libbeadcore-stripped.so" "$JNI/libbeadcore.so"
  cp "$WSL_HOME/pindou-deps/android-$abi/onnxruntime/jni/$abi/libonnxruntime.so" "$JNI/libonnxruntime.so"
done
mkdir -p app/assets/models

cp core/models/models.json core/models/ocr_v3.onnx core/models/ocr_v3.json app/assets/models/
# 安装包里只放 palettes/ 里核实过来源的色卡（AGENTS 规则 1）；MARD 等测试色卡用 tools/app-push-palettes.sh 推到手机上
[ -f palettes/build/palettes.json ] || "$PY" -m palettes.build_palettes
cp palettes/build/palettes.json app/assets/palettes.json
# 程序生成的测速图（没有版权问题），用来在真机上测 T0.14
[ -f build/bench_104.png ] || "$PY" tools/make_bench_image.py --out build/bench_104.png
cp build/bench_104.png app/assets/bench_104.png

cd app
"$FLUTTER" pub get
TARGETS=""
for abi in $APP_ABIS; do
  case "$abi" in arm64-v8a) t=android-arm64 ;; x86_64) t=android-x64 ;; *) echo "不支持的 ABI：$abi"; exit 2 ;; esac
  TARGETS="${TARGETS:+$TARGETS,}$t"
done
"$FLUTTER" build apk --release --target-platform "$TARGETS" --android-project-arg "pindou.abis=$(echo $APP_ABIS | tr ' ' ',')"
APK=build/app/outputs/flutter-apk/app-release.apk
ls -la "$APK"
echo "[app] 装到手机：adb install -r app/$APK"
