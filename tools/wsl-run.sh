#!/usr/bin/env bash
# 从 Windows 的 Git Bash 调用 WSL Ubuntu 执行命令（本地开发用）。
# MSYS_NO_PATHCONV 防止 Git Bash 把 /mnt/... 之类的参数改写成 Windows 路径。
export MSYS_NO_PATHCONV=1
wsl.exe -d "${PINDOU_WSL_DISTRO:-Ubuntu-24.04}" --exec bash -lc "$*" 2>&1 | tr -d '\0' | grep -av '^w s l\|^wsl:'
exit "${PIPESTATUS[0]}"
