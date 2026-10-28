"""python -m bench.label serve [--port 8765]：启动本地标注工具（只监听 127.0.0.1）。"""
from __future__ import annotations

import argparse
import sys
import webbrowser

from bench.common import formats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.label", description="真值标注工具（T0.4）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve", help="启动本地网页服务")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="启动后用默认浏览器打开")
    p.add_argument("--verbose", action="store_true", help="打印每个请求")
    a = ap.parse_args(argv)

    from bench.label.server import make_server

    root = formats.data_dir()
    srv = make_server(root, a.port, verbose=a.verbose)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"标注工具：{url}   数据目录：{root}   （Ctrl+C 退出）")
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
