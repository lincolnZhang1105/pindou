#!/usr/bin/env bash
# Windows 本地编译 core（MSVC + Ninja），输出到 build/win。用法：tools/build-win.sh [Release|Debug]
set -e
cd "$(dirname "$0")/.."
if [ ! -f build/win/build.ninja ]; then
  cmd //c "tools\msvc.cmd cmake -S core -B build\win -G Ninja -DCMAKE_BUILD_TYPE=${1:-Release}" > build/cfg-win.log 2>&1 || { cat build/cfg-win.log; exit 1; }
fi
cmd //c "tools\msvc.cmd cmake --build build\win --parallel" 2>&1 | iconv -f gbk -t utf-8 > build/build-win.log || true
if grep -aqE ": error|FAILED" build/build-win.log; then grep -aE ": error|FAILED" build/build-win.log | head -40; exit 1; fi
grep -aE "warning C" build/build-win.log | grep -v "_deps" | head -20 || true
echo "build ok"
