"""标注工具的本地 HTTP 服务：标准库 http.server，只监听 127.0.0.1。

安全：
- 只绑定 127.0.0.1，并校验 Host 头（防 DNS rebinding）；
- 写操作只接受 `Content-Type: application/json`（跨站的简单请求带不了这个头，会被 CORS 预检拦下）；
- URL 里的 key / image_id / run 名必须匹配 `[A-Za-z0-9_-]`，且必须在 manifest 里；静态文件只认 static/ 下的文件名，
  防路径穿越。

接口（JSON）：
  GET  /api/tasks                    任务列表（含状态）
  GET  /api/task/<key>               任务详情（图片、manifest 元数据、草稿或已导出的真值）
  GET  /api/palettes                 色卡（下拉框和色号校验）
  GET  /img/<image_id>               原图字节（HEIC 返回 collect convert 转出的 PNG）
  GET  /api/sample/<key>?image_index=&x=&y=&r=   取色（方块中位数）
  GET  /api/preds/<key>              有这张图预测结果的 run 列表
  GET  /api/pred/<key>/<run>         预测结果转成的初稿
  PUT  /api/draft/<key>              保存草稿
  POST /api/autofill/<key>           按颜色自动填充
  POST /api/snap/<key>               网格线吸附
  POST /api/export/<key>             导出真值 / 复核通过
"""
from __future__ import annotations

import json
import mimetypes
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from bench.common import dataset as ds
from bench.label import imaging
from bench.label.store import Store

STATIC = Path(__file__).resolve().parent / "static"
_STATIC_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
MAX_BODY = 64 * 1024 * 1024
_IMG_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif",
              "bmp": "image/bmp", "tiff": "image/tiff"}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Handler(BaseHTTPRequestHandler):
    server_version = "PindouLabel/1.0"
    store: Store  # 由 make_server 注入

    # ------------------------------------------------------------------ 基础
    def log_message(self, fmt, *args):  # noqa: D401 —— 安静一点：只记错误
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _check_host(self) -> None:
        host = (self.headers.get("Host") or "").strip().lower()
        port = self.server.server_address[1]
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            raise ApiError(HTTPStatus.FORBIDDEN, "Host 不合法（只允许 127.0.0.1 / localhost）")

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", {"Cache-Control": "no-store"})

    def _body(self) -> dict:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "只接受 application/json")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            raise ApiError(HTTPStatus.BAD_REQUEST, "请求体为空或过大")
        try:
            obj = json.loads(self.rfile.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"JSON 解析失败：{e}") from e
        if not isinstance(obj, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "请求体必须是 JSON 对象")
        return obj

    def _dispatch(self, method: str) -> None:
        try:
            self._check_host()
            url = urlsplit(self.path)
            parts = [unquote(p) for p in url.path.split("/") if p]
            query = parse_qs(url.query)
            route = getattr(self, f"_{method}_{parts[0] if parts else 'index'}".replace("-", "_"), None)
            if route is None:
                raise ApiError(HTTPStatus.NOT_FOUND, "没有这个地址")
            route(parts[1:], query)
        except ApiError as e:
            self._json({"error": e.message}, e.status)
        except Exception as e:  # noqa: BLE001
            self.log_error("内部错误：%r", e)
            self._json({"error": f"内部错误：{e!r}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self):  # noqa: N802
        self._dispatch("get")

    def do_HEAD(self):  # noqa: N802
        self._dispatch("get")

    def do_PUT(self):  # noqa: N802
        self._dispatch("put")

    def do_POST(self):  # noqa: N802
        self._dispatch("post")

    # ------------------------------------------------------------------ 工具
    def _key(self, parts: list[str], i: int = 0) -> str:
        if len(parts) <= i or not ds.valid_id(parts[i]):
            raise ApiError(HTTPStatus.BAD_REQUEST, "id 不合法")
        return parts[i]

    def _pages(self, key: str) -> list[dict]:
        pages = self.store.pages_of(key)
        if pages is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"没有任务 {key}")
        return pages

    def _inside_root(self, p: Path) -> Path:
        """manifest 里的 file 字段拼出来的路径必须还在数据目录里（防 ../ 穿越）。"""
        root = self.store.root.resolve()
        rp = p.resolve()
        if rp != root and root not in rp.parents:
            raise ApiError(HTTPStatus.FORBIDDEN, "图片路径不在数据目录里")
        return rp

    def _image_array(self, key: str, image_index: int):
        pages = self._pages(key)
        if not 0 <= image_index < len(pages):
            raise ApiError(HTTPStatus.BAD_REQUEST, "image_index 越界")
        p = self._inside_root(ds.readable_image_path(pages[image_index], self.store.root))
        if not p.is_file():
            raise ApiError(HTTPStatus.CONFLICT, f"图片不存在或还没转换：{p.name}（HEIC 先运行 python -m bench.collect convert）")
        return imaging.load_rgb(p)

    # ------------------------------------------------------------------ 静态文件和图片
    def _get_index(self, parts, query):
        self._static("index.html")

    def _get_static(self, parts, query):
        if len(parts) != 1 or not _STATIC_NAME.match(parts[0]):
            raise ApiError(HTTPStatus.NOT_FOUND, "没有这个文件")
        self._static(parts[0])

    def _static(self, name: str) -> None:
        p = (STATIC / name).resolve()
        if p.parent != STATIC.resolve() or not p.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "没有这个文件")
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype, {"Cache-Control": "no-cache"})

    def _get_img(self, parts, query):
        iid = self._key(parts)
        rec = self.store.manifest().get(iid)
        if rec is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"manifest 里没有 {iid}")
        p = self._inside_root(ds.readable_image_path(rec, self.store.root))
        if not p.is_file():
            raise ApiError(HTTPStatus.CONFLICT, f"图片不存在或还没转换：{p.name}（HEIC 先运行 python -m bench.collect convert）")
        fmt = "png" if p.suffix.lower() == ".png" else rec.get("format")
        etag = f'"{rec.get("sha256", "")[:32]}-{p.stat().st_mtime_ns}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        self._send(200, p.read_bytes(), _IMG_TYPES.get(fmt, "application/octet-stream"),
                   {"ETag": etag, "Cache-Control": "no-cache"})

    # ------------------------------------------------------------------ API
    def _get_api(self, parts, query):
        if not parts:
            raise ApiError(HTTPStatus.NOT_FOUND, "没有这个地址")
        name, rest = parts[0], parts[1:]
        if name == "tasks":
            return self._json({"tasks": self.store.task_list()})
        if name == "palettes":
            return self._json(self.store.palettes())
        if name == "task":
            t = self.store.task(self._key(rest))
            if t is None:
                raise ApiError(HTTPStatus.NOT_FOUND, "没有这个任务")
            return self._json(t)
        if name == "sample":
            key = self._key(rest)
            try:
                ii = int(query.get("image_index", ["0"])[0])
                x, y = float(query["x"][0]), float(query["y"][0])
                r = max(0, min(12, int(query.get("r", ["4"])[0])))
            except (KeyError, ValueError) as e:
                raise ApiError(HTTPStatus.BAD_REQUEST, "参数不合法") from e
            try:
                rgb = imaging.sample(self._image_array(key, ii), x, y, r)
            except ValueError as e:
                raise ApiError(HTTPStatus.BAD_REQUEST, str(e)) from e
            return self._json({"rgb": rgb})
        if name == "preds":
            key = self._key(rest)
            self._pages(key)
            return self._json({"runs": self.store.pred_runs(key)})
        if name == "pred":
            key, run = self._key(rest, 0), self._key(rest, 1)
            try:
                return self._json(self.store.pred_draft(key, run))
            except KeyError as e:
                raise ApiError(HTTPStatus.NOT_FOUND, f"没有预测结果 {run}/{key}") from e
        raise ApiError(HTTPStatus.NOT_FOUND, "没有这个地址")

    def _put_api(self, parts, query):
        if len(parts) == 2 and parts[0] == "draft":
            key = self._key(parts, 1)
            self._pages(key)
            draft = self._body()
            saved = self.store.save_draft(key, draft)
            return self._json({"ok": True, "rev": saved["rev"], "updated_at": saved["updated_at"],
                               "status": self.store.status(key, self._pages(key))})
        raise ApiError(HTTPStatus.NOT_FOUND, "没有这个地址")

    def _post_api(self, parts, query):
        if len(parts) != 2:
            raise ApiError(HTTPStatus.NOT_FOUND, "没有这个地址")
        name, key = parts[0], self._key(parts, 1)
        self._pages(key)
        body = self._body()
        if name == "export":
            draft = body.get("draft")
            if not isinstance(draft, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "缺少 draft")
            res = self.store.export(key, draft, body.get("user"), review=bool(body.get("review")))
            res["status"] = self.store.status(key, self._pages(key))
            return self._json(res, 200 if res["ok"] else HTTPStatus.UNPROCESSABLE_ENTITY)
        if name == "autofill":
            return self._json(self._autofill(key, body))
        if name == "snap":
            p = body.get("page") or {}
            try:
                xl = [float(v) for v in p["x_lines"]]
                yl = [float(v) for v in p["y_lines"]]
                ii = int(p.get("image_index", 0))
                frac = float(body.get("frac", 0.3))
            except (KeyError, TypeError, ValueError) as e:
                raise ApiError(HTTPStatus.BAD_REQUEST, "page 参数不合法") from e
            if len(xl) < 2 or len(yl) < 2:
                raise ApiError(HTTPStatus.BAD_REQUEST, "网格线太少")
            nx, ny, moved = imaging.snap_lines(self._image_array(key, ii), xl, yl, max(0.05, min(0.45, frac)))
            return self._json({"x_lines": nx, "y_lines": ny, "moved": moved})
        raise ApiError(HTTPStatus.NOT_FOUND, "没有这个地址")

    def _autofill(self, key: str, body: dict) -> dict:
        try:
            grid = body["grid"]
            cols, rows = int(grid["cols"]), int(grid["rows"])
            pages = body["pages"]
            sw = [(None if s.get("code") in (None, "") else str(s["code"]), [int(v) for v in s["rgb"]])
                  for s in body["swatches"]]
            opt = body.get("options") or {}
            inner = float(opt.get("inner", 0.5))
            ink_thr = float(opt.get("ink_thr", 80))
            ink_frac = float(opt.get("ink_frac", 0.04))
            max_de = float(opt.get("max_de", 12))
            overwrite = bool(opt.get("overwrite", True))
        except (KeyError, TypeError, ValueError) as e:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"参数不合法：{e}") from e
        if not sw:
            raise ApiError(HTTPStatus.BAD_REQUEST, "没有色块：先给图例里的色号取色")
        cells = body.get("cells")
        if not (isinstance(cells, list) and len(cells) == rows and all(isinstance(r, list) and len(r) == cols
                                                                         for r in cells)):
            cells = [[None] * cols for _ in range(rows)]
        flags = []
        filled = 0
        for p in pages:
            try:
                xl = [float(v) for v in p["x_lines"]]
                yl = [float(v) for v in p["y_lines"]]
                co, ro = int(p.get("col_offset", 0)), int(p.get("row_offset", 0))
                img = self._image_array(key, int(p.get("image_index", 0)))
            except (KeyError, TypeError, ValueError) as e:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"页参数不合法：{e}") from e
            med, ink = imaging.cell_stats(img, xl, yl, inner=inner, ink_thr=ink_thr)
            codes, fl = imaging.assign(med, ink, sw, max_de=max_de, ink_frac=ink_frac)
            for r, row in enumerate(codes):
                gr = r + ro
                if not 0 <= gr < rows:
                    continue
                for c, code in enumerate(row):
                    gc = c + co
                    if not 0 <= gc < cols:
                        continue
                    if overwrite or cells[gr][gc] in (None, ""):
                        cells[gr][gc] = code
                        filled += 1
            flags.extend([r + ro, c + co, why] for r, c, why in fl if 0 <= r + ro < rows and 0 <= c + co < cols)
        return {"cells": cells, "flags": flags, "filled": filled}


def make_server(root: Path, port: int = 8765, verbose: bool = False) -> ThreadingHTTPServer:
    """创建（不启动）服务。port=0 时随机分配端口（测试用）。"""
    handler = type("BoundHandler", (Handler,), {"store": Store(root)})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    srv.verbose = verbose
    return srv


def serve_in_thread(root: Path, port: int = 0) -> tuple[ThreadingHTTPServer, threading.Thread]:
    srv = make_server(root, port)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    return srv, t
