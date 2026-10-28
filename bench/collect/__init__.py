"""bench.collect：manifest 工具（T0.3）。用法见 bench/collect/README.md，或 `python -m bench.collect -h`。

**只处理人工保存下来的图片，不写任何抓取、爬虫、自动下载代码**（AGENTS 规则 5）。
"""
from bench.collect.lock import check_lock

__all__ = ["check_lock"]
