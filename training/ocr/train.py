"""训练 OCR 模型（T0.11），导出 ONNX 到 core/models/。

    python -m training.ocr.train --steps 20000 --out training/runs/ocr_v1
    python -m training.ocr.train --export-only training/runs/ocr_v1/best.pt

完成标准（07 T0.11）：合成验证集上准确率 ≥ 99.5%；模型合计 < 3 MB；C++ 与 PyTorch 输出误差 < 1e-3。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from training.ocr.common import BLANK, CHARSET, INPUT_H, decode_greedy, encode
from training.ocr.gen import make_set, render
from training.ocr.model import CRNN

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "core" / "models"


class Pool:
    """synth_pool.py 生成的整页合成图文字裁剪。末尾 holdout 比例的样本留作验证。"""

    def __init__(self, path: Path, holdout: float = 0.03):
        d = np.load(path)
        self.flat, self.widths, self.labels, self.kinds = d["flat"], d["widths"], d["labels"], d["kinds"]
        self.offsets = np.concatenate([[0], np.cumsum(self.widths.astype(np.int64) * INPUT_H)])
        n = len(self.widths)
        self.n_train = int(n * (1 - holdout))
        self.n = n
        # 坐标数字远多于色号和颗数：训练时按种类分层抽样
        self.by_kind = {k: np.nonzero(self.kinds[: self.n_train] == k)[0] for k in ("code", "count", "coord", "cell")}
        has_cells = len(self.by_kind["cell"]) > 0
        self.kind_p = ({"code": 0.3, "count": 0.25, "coord": 0.2, "cell": 0.25} if has_cells
                       else {"code": 0.4, "count": 0.35, "coord": 0.25})

    def sample(self, rng: random.Random):
        r, acc = rng.random(), 0.0
        for k, pk in self.kind_p.items():
            acc += pk
            if r < acc and len(self.by_kind[k]):
                return self.get(int(self.by_kind[k][rng.randrange(len(self.by_kind[k]))]))
        return self.get(rng.randrange(self.n_train))

    def get(self, i: int):
        w = int(self.widths[i])
        x = self.flat[self.offsets[i] : self.offsets[i] + INPUT_H * w].reshape(INPUT_H, w).astype(np.float32) / 255.0
        return x, str(self.labels[i]), str(self.kinds[i])


class Stream(IterableDataset):
    def __init__(self, seed: int, pool_path: Path | None, pool_frac: float):
        self.seed = seed
        self.pool_path = pool_path
        self.pool_frac = pool_frac

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid)
        pool = Pool(self.pool_path) if self.pool_path else None
        while True:
            if pool is not None and rng.random() < self.pool_frac:
                x, label, _ = pool.sample(rng)
            else:
                x, label, _ = render(rng)
            yield x, label


def collate(batch):
    W = max(x.shape[1] for x, _ in batch)
    W = int(math.ceil(W / 4) * 4)
    xs = np.zeros((len(batch), 1, INPUT_H, W), np.float32)
    targets, tlen, ilen, labels = [], [], [], []
    for i, (x, label) in enumerate(batch):
        xs[i, 0, :, : x.shape[1]] = x
        t = encode(label)
        targets += t
        tlen.append(len(t))
        ilen.append(x.shape[1] // 4)
        labels.append(label)
    return (torch.from_numpy(xs), torch.tensor(targets, dtype=torch.long), torch.tensor(ilen, dtype=torch.long),
            torch.tensor(tlen, dtype=torch.long), labels)


@torch.no_grad()
def evaluate(model: CRNN, val: list) -> dict:
    model.eval()
    ok, n = 0, 0
    by_kind: dict[str, list[int]] = {}
    for i in range(0, len(val), 256):
        chunk = val[i : i + 256]
        xb, _, ilen, _, labels = collate([(x, l) for x, l, _ in chunk])
        out = model(xb).argmax(-1).T  # [N, T]
        for j, (_, label, kind) in enumerate(chunk):
            pred = decode_greedy(out[j, : ilen[j]].tolist())
            hit = int(pred == label)
            ok += hit
            n += 1
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][0] += hit
            by_kind[kind][1] += 1
    model.train()
    return {"acc": ok / n, **{f"acc_{k}": v[0] / v[1] for k, v in by_kind.items()}}


def export(model: CRNN, path: Path, meta: dict) -> None:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, INPUT_H, 64)
    torch.onnx.export(model, dummy, str(path), input_names=["x"], output_names=["logp"],
                      dynamic_axes={"x": {0: "n", 3: "w"}, "logp": {0: "t", 1: "n"}}, opset_version=17,
                      dynamo=False)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"导出 {path}（{path.stat().st_size / 1024:.0f} KB）和 {meta_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=REPO / "training" / "runs" / "ocr_v1")
    ap.add_argument("--export-only", type=Path)
    ap.add_argument("--name", default="ocr_v1")
    ap.add_argument("--pool", type=Path, help="synth_pool.py 生成的 npz")
    ap.add_argument("--pool-frac", type=float, default=0.4)
    ap.add_argument("--init", type=Path, help="从已有检查点继续训练（字符集相同时）")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    model = CRNN()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {"name": args.name, "charset": CHARSET, "blank": BLANK, "input_h": INPUT_H, "params": n_params,
            "preprocess": "training/ocr/common.py:ink_crop", "trained_on": "合成数据（training/ocr/gen.py，OFL 字体）"}
    if args.export_only:
        ck = torch.load(args.export_only, map_location="cpu")
        model.load_state_dict(ck["model"])
        meta.update(ck.get("meta", {}))
        export(model, MODELS / f"{args.name}.onnx", meta)
        return

    print(f"参数量 {n_params}，生成验证集 {args.val} 张……")
    val = make_set(args.val, seed=10_000 + args.seed)
    val_pool = []
    if args.pool:
        pool = Pool(args.pool)
        # 验证：holdout 里每类最多 1500 个
        val_pool, cnt = [], {}
        for i in range(pool.n_train, pool.n):
            x, label, kind = pool.get(i)
            if cnt.get(kind, 0) < 1500:
                val_pool.append((x, label, kind))
                cnt[kind] = cnt.get(kind, 0) + 1
        meta["trained_on"] += f"；整页合成图裁剪 {args.pool.name}（{pool.n_train} 个，占比 {args.pool_frac}）"
        print(f"整页合成图裁剪：训练 {pool.n_train}，验证 {len(val_pool)}")
    loader = DataLoader(Stream(args.seed, args.pool, args.pool_frac), batch_size=args.batch, num_workers=args.workers,
                        collate_fn=collate,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        own = model.state_dict()
        # 字符集变了时输出层形状不同：只加载形状一致的参数
        own.update({k: v for k, v in ck["model"].items() if k in own and own[k].shape == v.shape})
        model.load_state_dict(own)
        print(f"从 {args.init} 初始化")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    args.out.mkdir(parents=True, exist_ok=True)
    best, t0, step, run_loss = -1.0, time.time(), 0, 0.0
    for xb, tg, il, tl, _ in loader:
        logp = model(xb)
        loss = ctc(logp, tg, il, tl)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        run_loss = 0.98 * run_loss + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0 or step == args.steps:
            r = evaluate(model, val)
            if val_pool:
                rp = evaluate(model, val_pool)
                r.update({f"pool_{k}": v for k, v in rp.items()})
            print(f"step {step} loss {run_loss:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in r.items())
                  + f" {time.time() - t0:.0f}s", flush=True)
            # 以整页合成图裁剪（更接近线上）的准确率为准；没有时用渲染样本
            score = r.get("pool_acc", r["acc"])
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "meta": {"val": r, "step": step}}, args.out / "best.pt")
        if step >= args.steps:
            break
    ck = torch.load(args.out / "best.pt", map_location="cpu")
    model.load_state_dict(ck["model"])
    meta["val"] = ck["meta"]["val"]
    meta["step"] = ck["meta"]["step"]
    export(model, MODELS / f"{args.name}.onnx", meta)


if __name__ == "__main__":
    main()
