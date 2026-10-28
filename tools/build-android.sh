#!/usr/bin/env bash
# 交叉编译 Android arm64 版 beadcore-cli（T0.14 真机测速用），在 Linux / WSL 上运行：
#   tools/build-android.sh            # 产物在 $BUILD_DIR（默认 ~/pindou-build/android-arm64）
# 依赖：python3、cmake、ninja；首次运行会下载 Android NDK（约 700 MB）到 ~/android/。
# OpenCV 关闭 KleidiCV / Carotene 这两个 ARM 加速库，保证结果和桌面端一致（App 是否打开，测速后再定）。
set -euo pipefail
cd "$(dirname "$0")/.."
NDK_VER="${NDK_VER:-r27c}"
NDK_DIR="${ANDROID_NDK:-$HOME/android/android-ndk-$NDK_VER}"
API="${ANDROID_API:-26}"
ABI=arm64-v8a
export BC_DEPS_DIR="${BC_DEPS_DIR:-$HOME/pindou-deps}"
BUILD_DIR="${BUILD_DIR:-$HOME/pindou-build/android-arm64}"

if [ ! -x "$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" ]; then
  mkdir -p "$(dirname "$NDK_DIR")"
  zip="$(dirname "$NDK_DIR")/android-ndk-$NDK_VER-linux.zip"
  [ -f "$zip" ] || curl -fL "https://dl.google.com/android/repository/android-ndk-$NDK_VER-linux.zip" -o "$zip"
  # 没有 unzip 时用 Python 解压，并按 zip 里记录的 unix 权限恢复可执行位
  python3 - "$zip" "$(dirname "$NDK_DIR")" <<'PY'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as z:
    for info in z.infolist():
        path = z.extract(info, dst)
        mode = info.external_attr >> 16
        if mode and not info.is_dir():
            if (mode & 0o170000) == 0o120000:  # 符号链接
                target = open(path).read()
                os.remove(path)
                os.symlink(target, path)
            else:
                os.chmod(path, mode & 0o7777)
PY
fi
TOOLCHAIN="$NDK_DIR/build/cmake/android.toolchain.cmake"
echo "[android] NDK: $NDK_DIR"

python3 core/scripts/fetch_deps.py --platform "android-$ABI" --build-opencv --generator Ninja \
  --opencv-arg="-DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN" --opencv-arg="-DANDROID_ABI=$ABI" \
  --opencv-arg="-DANDROID_PLATFORM=android-$API" --opencv-arg="-DBUILD_ANDROID_PROJECTS=OFF" \
  --opencv-arg="-DBUILD_ANDROID_EXAMPLES=OFF" --opencv-arg="-DWITH_KLEIDICV=OFF" --opencv-arg="-DWITH_CAROTENE=OFF" \
  --opencv-arg="-DBUILD_FAT_JAVA_LIB=OFF"

cmake -S core -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" -DANDROID_ABI=$ABI -DANDROID_PLATFORM="android-$API" \
  -DANDROID_STL=c++_static -DBC_PLATFORM="android-$ABI" -DBC_BUILD_TESTS=OFF
cmake --build "$BUILD_DIR" --parallel
echo "[android] 产物：$BUILD_DIR/beadcore-cli，依赖库：$BC_DEPS_DIR/android-$ABI/onnxruntime/jni/$ABI/libonnxruntime.so"
STRIP="$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
"$STRIP" -o "$BUILD_DIR/beadcore-cli-stripped" "$BUILD_DIR/beadcore-cli"
echo "[android] 去掉调试信息的版本：$BUILD_DIR/beadcore-cli-stripped（推到手机用这个）"
