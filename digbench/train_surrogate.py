"""Phase 4: learned terrain + force surrogate trained on MPM rollouts, evaluated on held-out episodes.

State on a 5 cm grid (half bed, mirrored at y = 0): H (ground surface). Action: bucket raster T_t, T_{t+1}
(lowest point of the bucket solid per column). Soil parameter: mu. Model predicts dH over 0.1 s and the soil
force on the bucket at t+1. Trained with multi-step rollouts (pushforward) and noise injection.

Baselines on the same held-out episodes and the same bucket raster:
  * LiveDirt-style: cut H to min(H, T_{t+1}) where the bucket is, then LiveDirt relax (repose = bank = atan mu).
  * Persistence: H stays at H_0 (lower bound on usefulness).

  python -m digbench.train_surrogate --data /content/surrogate --epochs 60 --out digbench/results
"""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np

SURF = 1.2


def load(files):
    eps = []
    for f in files:
        d = np.load(f)
        eps.append({"H": d["H"].astype(np.float32), "T": d["T"].astype(np.float32), "F": d["F"].astype(np.float32),
                    "mu": float(d["mu"]), "carried": d["carried"].astype(np.float32), "name": Path(f).stem})
    return eps


def features(H, Tn, Tn1, mu, torch):
    """H, Tn, Tn1: (B, nx, ny) tensors (NaN = no bucket). Returns (B, 6, nx, 2*ny) mirrored input."""
    def tool(T):
        occ = torch.isfinite(T).float()
        pen = torch.where(torch.isfinite(T), (T - H).clamp(-1.0, 1.0), torch.ones_like(H))
        return pen, occ
    p0, o0 = tool(Tn)
    p1, o1 = tool(Tn1)
    m = torch.full_like(H, 1.0) * (mu.view(-1, 1, 1) - 0.5) * 4.0
    x = torch.stack([(H - SURF) * 2.0, p0, o0, p1, o1, m], 1)
    return torch.cat([x.flip(-1), x], -1)


def build_model(torch, ch=48):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(s, i, o):
            super().__init__()
            s.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())

        def forward(s, x):
            return s.c(x)

    class UNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.e1, s.e2, s.e3 = Block(6, ch), Block(ch, 2 * ch), Block(2 * ch, 4 * ch)
            s.d2, s.d1 = Block(6 * ch, 2 * ch), Block(3 * ch, ch)
            s.out = nn.Conv2d(ch, 1, 1)
            s.force = nn.Sequential(nn.Linear(4 * ch, 128), nn.GELU(), nn.Linear(128, 3))
            s.pool = nn.AvgPool2d(2)

        def forward(s, x):
            nx, ny = x.shape[-2:]
            px, py = (-nx) % 4, (-ny) % 4
            x = nn.functional.pad(x, (0, py, 0, px), mode="replicate")
            a = s.e1(x)
            b = s.e2(s.pool(a))
            c = s.e3(s.pool(b))
            u = nn.functional.interpolate(c, size=b.shape[-2:], mode="nearest")
            d = s.d2(torch.cat([u, b], 1))
            u = nn.functional.interpolate(d, size=a.shape[-2:], mode="nearest")
            e = s.d1(torch.cat([u, a], 1))
            dh = s.out(e)[..., :nx, :ny] * 0.05
            f = s.force(c.mean(dim=(-2, -1))) * 1e5
            return dh, f

    return UNet()


def unmirror(y, ny):
    return 0.5 * (y[..., ny:] + y[..., :ny].flip(-1))


def livedirt_rollout(ep, cell=0.05):
    from .solvers.heightfield import Heightfield
    ang = math.degrees(math.atan(ep["mu"]))
    H = ep["H"][0].astype(np.float64)
    out = [H.copy()]
    for t in range(len(ep["H"]) - 1):
        T = ep["T"][t + 1]
        cut = np.where(np.isfinite(T), np.minimum(H, T), H)
        full = np.concatenate([cut[:, ::-1], cut], 1)
        hf = Heightfield(np.full_like(full, SURF), cell, ang, ang)
        hf.B[:] = full
        for _ in range(20):  # LiveDirt runs relax every frame; 20 sweeps per 0.1 s at 5 cm cells
            if hf.relax(2) <= 2e-4:
                break
        H = hf.h()[:, cut.shape[1]:]
        out.append(H.copy())
    return np.array(out)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default="digbench/results")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(f"{args.data}/*.npz"))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    nval = max(2, int(len(files) * args.val_frac))
    val, train = load(files[:nval]), load(files[nval:])
    print(f"train {len(train)} episodes, val {len(val)} episodes, device {dev}", flush=True)
    model = build_model(torch).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    tens = lambda a: torch.tensor(a, device=dev)
    # flatten training windows
    wins = [(e, t) for e in range(len(train)) for t in range(len(train[e]["H"]) - args.unroll)]
    t_train = time.perf_counter()
    for ep_i in range(args.epochs):
        rng.shuffle(wins)
        tot = 0.0
        for b0 in range(0, len(wins), 64):
            batch = wins[b0:b0 + 64]
            H = tens(np.stack([train[e]["H"][t] for e, t in batch]))
            mu = tens(np.array([train[e]["mu"] for e, t in batch], np.float32))
            H = H + args.noise * torch.randn_like(H)
            loss = 0.0
            for k in range(args.unroll):
                Tn = tens(np.stack([train[e]["T"][t + k] for e, t in batch]))
                Tn1 = tens(np.stack([train[e]["T"][t + k + 1] for e, t in batch]))
                Ht = tens(np.stack([train[e]["H"][t + k + 1] for e, t in batch]))
                Ft = tens(np.stack([train[e]["F"][t + k + 1] for e, t in batch]))
                dh, f = model(features(H, Tn, Tn1, mu, torch))
                H = H + unmirror(dh[:, 0], H.shape[-1])
                loss = loss + ((H - Ht) ** 2).mean() / 0.01 ** 2 + (((f - Ft) / 1e4) ** 2).mean() * 0.1
                if k < args.unroll - 1:
                    H = H.detach() * 0.5 + H * 0.5  # partial pushforward gradient
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        sched.step()
        if ep_i % 5 == 0 or ep_i == args.epochs - 1:
            print(f"epoch {ep_i}: loss {tot / max(1, len(wins) // 64):.3f}", flush=True)
    t_train = time.perf_counter() - t_train

    # ---- evaluation: autoregressive rollouts on held-out episodes ----
    model.eval()
    rows = []
    # LiveDirt baseline rollouts are pure-Python CPU work: run them in parallel up front
    from concurrent.futures import ProcessPoolExecutor
    import os
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        lv_all = list(pool.map(livedirt_rollout, val))
    with torch.no_grad():
        for vi, ep in enumerate(val):
            n = len(ep["H"])
            H = tens(ep["H"][0][None])
            mu = tens(np.array([ep["mu"]], np.float32))
            pred, forces = [ep["H"][0]], []
            t0 = time.perf_counter()
            for t in range(n - 1):
                dh, f = model(features(H, tens(ep["T"][t][None]), tens(ep["T"][t + 1][None]), mu, torch))
                H = H + unmirror(dh[:, 0], H.shape[-1])
                pred.append(H[0].cpu().numpy()); forces.append(f[0].cpu().numpy())
            if dev == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / (n - 1) * 1e3
            pred = np.array(pred); forces = np.array(forces)
            lv = lv_all[vi]
            truth = ep["H"]
            dist = np.abs(truth[-1] - truth[0]) > 0.02
            rm = lambda A: float(np.sqrt(((A[-1] - truth[-1]) ** 2)[dist].mean())) if dist.any() else float("nan")
            rm_all = lambda A: float(np.sqrt(((A - truth) ** 2).mean()))
            Fx_t = ep["F"][1:, 0]
            drag = np.abs(Fx_t) > 1e3
            rows.append({
                "episode": ep["name"], "mu": ep["mu"], "frames": n, "ms_per_step": ms,
                "final_rmse_disturbed_m": {"surrogate": rm(pred), "livedirt": rm(lv), "persistence": rm(np.repeat(truth[:1], n, 0))},
                "rollout_rmse_all_m": {"surrogate": rm_all(pred), "livedirt": rm_all(lv), "persistence": rm_all(np.repeat(truth[:1], n, 0))},
                "force_x_mae_kN": float(np.abs(forces[:, 0] - Fx_t).mean() / 1e3),
                "force_x_mean_true_kN": float(np.abs(Fx_t[drag]).mean() / 1e3) if drag.any() else 0.0,
                "force_x_rel_err_drag": float(np.abs(forces[drag, 0] - Fx_t[drag]).mean() / np.abs(Fx_t[drag]).mean()) if drag.any() else None,
            })
            print(json.dumps(rows[-1]), flush=True)
    agg = lambda key, m: float(np.mean([r[key][m] for r in rows]))
    summary = {
        "params": nparam, "train_episodes": len(train), "val_episodes": len(val), "train_s": t_train,
        "final_rmse_disturbed_m": {m: agg("final_rmse_disturbed_m", m) for m in ("surrogate", "livedirt", "persistence")},
        "rollout_rmse_all_m": {m: agg("rollout_rmse_all_m", m) for m in ("surrogate", "livedirt", "persistence")},
        "force_x_rel_err_drag": float(np.mean([r["force_x_rel_err_drag"] for r in rows if r["force_x_rel_err_drag"] is not None])),
        "ms_per_step": float(np.mean([r["ms_per_step"] for r in rows])), "device": dev,
    }
    out = {"task": "surrogate", "tag": args.tag, "summary": summary, "episodes": rows, "params": vars(args)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"surrogate-{args.tag}.json").write_text(json.dumps(out, indent=1))
    torch.save(model.state_dict(), Path(args.out) / f"surrogate-{args.tag}.pt")
    try:
        dummy = features(tens(val[0]["H"][0][None]), tens(val[0]["T"][0][None]), tens(val[0]["T"][1][None]),
                         tens(np.array([0.5], np.float32)), torch)
        torch.onnx.export(model, dummy, str(Path(args.out) / f"surrogate-{args.tag}.onnx"), input_names=["x"],
                          output_names=["dh", "force"], opset_version=17, dynamo=False)
    except Exception as e:
        print("onnx export failed:", e)
    # parity fixture for the browser port (app/src/sim/neuralSoil.ts): full-width (mirrored) H, T0, T1, mu, the
    # model input x, and the PyTorch outputs
    with torch.no_grad():
        ep = val[0]
        t = len(ep["H"]) // 3
        H = tens(ep["H"][t][None]); T0 = tens(ep["T"][t][None]); T1 = tens(ep["T"][t + 1][None]); mu = tens(np.array([ep["mu"]], np.float32))
        x = features(H, T0, T1, mu, torch)
        dh, f = model(x)
        full = lambda a: torch.cat([a.flip(-1), a], -1)[0].cpu().numpy()
        fixture = {"nx": int(x.shape[-2]), "ny": int(x.shape[-1]), "mu": float(ep["mu"]),
                   "H": full(H).ravel().tolist(), "T0": [None if not np.isfinite(v) else float(v) for v in full(T0).ravel()],
                   "T1": [None if not np.isfinite(v) else float(v) for v in full(T1).ravel()],
                   "x": x[0].cpu().numpy().ravel().tolist(), "dh": dh[0, 0].cpu().numpy().ravel().tolist(), "force": f[0].cpu().numpy().tolist()}
        (Path(args.out) / f"surrogate-{args.tag}-fixture.json").write_text(json.dumps(fixture))
    print("summary:", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
