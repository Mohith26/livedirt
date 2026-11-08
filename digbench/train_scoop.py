"""Train the per-scoop soil model (digbench/scoop_model.py) on MPM bite samples.

  python -m digbench.train_scoop --data ~/v4/scoops/* --heldout-runs sg15-s1 --out ~/v4/models --tag s1

Held-out split is by RUN (whole trench episodes), never by bite, so no held-out bite shares a trench with training.
Reports, on held-out bites: final-surface RMSE over disturbed cells for (a) zero change, (b) the cut+relax prior,
(c) the model; carried-volume error; volume balance. Writes scoop-<tag>.{pt,onnx,json}.
"""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .scoop_model import CELL, load_sample, relax


def conv(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(), nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin=7, ch=32):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = conv(cin, ch), conv(ch, 2 * ch), conv(2 * ch, 4 * ch), conv(4 * ch, 8 * ch)
        self.d3, self.d2, self.d1 = conv(12 * ch, 4 * ch), conv(6 * ch, 2 * ch), conv(3 * ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)
        self.head = nn.Sequential(nn.Linear(16 * ch, 128), nn.SiLU(), nn.Linear(128, 3))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(Fn.max_pool2d(e1, 2))
        e3 = self.e3(Fn.max_pool2d(e2, 2))
        e4 = self.e4(Fn.max_pool2d(e3, 2))
        g = torch.cat([e4.mean((2, 3)), e4.amax((2, 3))], 1)
        d3 = self.d3(torch.cat([Fn.interpolate(e4, size=e3.shape[2:]), e3], 1))
        d2 = self.d2(torch.cat([Fn.interpolate(d3, size=e2.shape[2:]), e2], 1))
        d1 = self.d1(torch.cat([Fn.interpolate(d2, size=e1.shape[2:]), e1], 1))
        return self.out(d1)[:, 0], self.head(g)


def pad8(a):
    nx, ny = a.shape[-2:]
    px, py = (-nx) % 8, (-ny) % 8
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, px), (0, py)], mode="edge"), (nx, ny)


def batchify(S, flip=False):
    X = np.stack([s["X"] for s in S]); P = np.stack([s["prior"] for s in S]); D = np.stack([s["delta"] for s in S])
    if flip:
        X, P, D = X[..., ::-1].copy(), P[..., ::-1].copy(), D[..., ::-1].copy()
    X, (nx, ny) = pad8(X); P, _ = pad8(P[:, None]); D, _ = pad8(D[:, None])
    return torch.tensor(X), torch.tensor(P[:, 0]), torch.tensor(D[:, 0]), torch.tensor(np.stack([s["scalars"] for s in S])), (nx, ny)


def disturbed(d_true, d_pred, thr=0.01):
    return (np.abs(d_true) > thr) | (np.abs(d_pred) > thr)


def rmse(a, b, m):
    return float(np.sqrt(((a - b) ** 2)[m].mean())) if m.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--heldout-runs", default="", help="comma list of run-name prefixes held out")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--vol-w", type=float, default=1.0)
    ap.add_argument("--no-prior", action="store_true", help="ablation: predict delta directly (prior channel zeroed, no residual)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    files = sorted(set(f for d in args.data for f in glob.glob(str(Path(d) / "*.npz"))))
    t0 = time.time()
    S, seen = [], set()
    for f in files:
        s_ = load_sample(f)
        key = hash((s_["X"][0].tobytes(), s_["X"][1].tobytes()))  # identical terrain + swept envelope = same bite replayed
        if key in seen:
            continue
        seen.add(key); S.append(s_)
    if args.no_prior:
        for s in S:
            s["X"][3] = 0.0; s["prior"] = np.zeros_like(s["prior"])
    ho = [h for h in args.heldout_runs.split(",") if h]
    run = lambda s: s["name"].rsplit("_", 1)[0]
    test = [s for s in S if any(run(s).startswith(h) for h in ho)]
    train = [s for s in S if s not in test]
    print(f"samples {len(S)} train {len(train)} test {len(test)} (load {time.time() - t0:.0f}s)", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet(S[0]["X"].shape[0], args.ch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * max(1, math.ceil(len(train) / args.bs)))
    C2 = CELL * CELL
    for ep in range(args.epochs):
        net.train(); np.random.shuffle(train); tl = 0.0
        for b in range(0, len(train), args.bs):
            X, P, D, Y, (nx, ny) = batchify(train[b:b + args.bs], flip=np.random.rand() < 0.5)
            X, P, D, Y = X.to(dev), P.to(dev), D.to(dev), Y.to(dev)
            r, ys = net(X)
            pred = P + r
            w = 1.0 + 9.0 * ((D.abs() > 0.01) | (P.abs() > 0.01)).float()
            l_map = (w * (pred - D) ** 2)[:, :nx, :ny].mean()
            vol_pred = pred[:, :nx, :ny].sum((1, 2)) * C2
            l_vol = ((vol_pred + ys[:, 0]) ** 2).mean()  # removed from the ground = carried (volume-corrected MPM)
            l_s = ((ys - Y) ** 2 * torch.tensor([10.0, 1.0, 0.05], device=dev)).mean()
            loss = l_map + args.vol_w * l_vol + l_s
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += float(loss)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep} loss {tl / max(1, len(train) // args.bs):.5f}", flush=True)
    # evaluation on held-out bites
    net.eval(); rows = []
    with torch.no_grad():
        for s in test:
            X, P, D, Y, (nx, ny) = batchify([s])
            r, ys = net(X.to(dev))
            pred = (P.to(dev) + r)[0, :nx, :ny].cpu().numpy()
            d = s["delta"]; pr = s["prior"]
            m = disturbed(d, pred) | disturbed(d, pr)
            rows.append({"name": s["name"], "rmse_zero": rmse(np.zeros_like(d), d, m), "rmse_prior": rmse(pr, d, m), "rmse_model": rmse(pred, d, m),
                         "carried_true": float(s["scalars"][0]), "carried_model": float(ys[0, 0]), "carried_prior": float(s["prior_vol"]),
                         "vol_balance_model": float(pred.sum() * C2 + ys[0, 0]), "cells": int(m.sum())})
    agg = lambda k: float(np.mean([r[k] for r in rows])) if rows else None
    summ = {"n_test": len(rows), "n_train": len(train), "rmse_zero": agg("rmse_zero"), "rmse_prior": agg("rmse_prior"), "rmse_model": agg("rmse_model"),
            "carried_mae_model": float(np.mean([abs(r["carried_model"] - r["carried_true"]) for r in rows])) if rows else None,
            "carried_mae_prior": float(np.mean([abs(r["carried_prior"] - r["carried_true"]) for r in rows])) if rows else None,
            "vol_balance_abs_model": float(np.mean([abs(r["vol_balance_model"]) for r in rows])) if rows else None}
    print(json.dumps(summ), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / f"scoop-{args.tag}.pt")
    (out / f"scoop-{args.tag}.json").write_text(json.dumps({"args": vars(args), "summary": summ, "rows": rows, "files": len(files)}, indent=1))
    try:
        X, *_ = batchify([S[0]])
        torch.onnx.export(net.cpu(), X, out / f"scoop-{args.tag}.onnx", input_names=["x"], output_names=["residual", "scalars"],
                          dynamic_axes={"x": {2: "nx", 3: "ny"}, "residual": {1: "nx", 2: "ny"}}, opset_version=17)
    except Exception as e:  # ONNX export is optional for evaluation
        print("onnx export failed:", e)


if __name__ == "__main__":
    main()
