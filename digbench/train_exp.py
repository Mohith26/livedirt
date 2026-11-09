"""Training-method experiments for the per-scoop model (same data split and recipe as train_scoop, one change per flag).

  --xflip          mirror augmentation along x (drag axis) as well as y; physics is reflection-symmetric, so a mirrored
                   bite (terrain, path, result all mirrored) is a valid new sample. Covers the rarer +x drag direction.
  --prior seq      physics prior that fills the bucket in path order (cells cut in first-touch order until 0.95 m^3),
                   instead of scaling the whole cut down uniformly when it exceeds capacity
  --ema D          also keep an exponential moving average of the weights (decay D) and report it
  --subset-frac F  train on a fixed random fraction F of the train tier ("old data") for the incremental experiment
  --init P         warm start from a checkpoint; --shrink L --perturb S applies shrink-and-perturb (w = L w + S w_fresh)
Evaluation (pooled RMS over disturbed cells, the judge's metric) on the test tier, with zones, optional flip TTA, and
the per-bite predictions saved for ensemble / churn / disagreement analysis.

  python -m digbench.train_exp --tag base-s0 --seed 0 --out ~/v4/wsp/exp
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch

from digbench.scoop_model import CELL, load_sample, relax, swept_envelope
from digbench.train_scoop import UNet, batchify

SPLITS = Path("~/v4/wsp/splits.json").expanduser()


def seq_prior(terrain, env, first, phi_deg, cap=0.95, C=CELL):
    cut = np.where(np.isfinite(env), np.maximum(0.0, terrain - env), 0.0)
    order = np.argsort(np.where(np.isfinite(first), first, np.inf), axis=None, kind="stable")
    vol = cut.ravel()[order] * C * C; cum = np.cumsum(vol)
    keep = np.clip((cap - (cum - vol)) / np.maximum(vol, 1e-12), 0.0, 1.0)
    k = np.zeros(cut.size); k[order] = keep
    cut = cut * k.reshape(cut.shape)
    H1 = relax(terrain - cut, phi_deg, C); d = H1 - terrain
    near = np.isfinite(env)
    for _ in range(int(0.6 / C)):
        near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
    return np.where(near, d, 0.0).astype(np.float32)


def load(files, prior):
    S, seen = [], set()
    for f in files:
        s = load_sample(f)
        key = hash((s["X"][0].tobytes(), s["X"][1].tobytes()))
        if key in seen:
            continue
        seen.add(key)
        z = np.load(f); s["completion"] = json.loads(str(z["meta"]))["completion"]
        if prior == "seq":
            from digbench.wsp.stop_point import _walk_file
            (_, _, _, first), _ = _walk_file(f); T = z["terrain"].astype(float)
            env = swept_envelope(z["poses"], *T.shape, float(z["margin"]))
            phi = float(s["soil"].get("phi", math.degrees(math.atan(s["soil"].get("mu", 0.6)))))
            p = seq_prior(T, env, first, phi); s["prior"] = p; s["X"][3] = p
        S.append(s)
    return S


def xflip(s):
    return {**s, "X": s["X"][:, ::-1, :].copy(), "prior": s["prior"][::-1, :].copy(), "delta": s["delta"][::-1, :].copy()}


@torch.no_grad()
def infer(net, s, dev, tta=""):
    """Prediction (nx, ny); tta: '' none, 'y' average with y-flip, 'xy' average over the 4 mirror images."""
    views = [(False, False)] + ([(False, True)] if tta in ("y", "xy") else []) + ([(True, False), (True, True)] if tta == "xy" else [])
    acc = 0.0
    for fx, fy in views:
        s2 = xflip(s) if fx else s
        X, P, D, Y, (nx, ny) = batchify([s2], flip=fy)
        r, ys = net(X.to(dev)); p = (P.to(dev) + r)[0, :nx, :ny].cpu().numpy()
        if fy:
            p = p[:, ::-1]
        if fx:
            p = p[::-1, :]
        acc = acc + p
    return acc / len(views)


def pooled(preds, test):
    acc = {}
    for p, s in zip(preds, test):
        d = s["delta"]; m = (np.abs(d) > 0.01) | (np.abs(p) > 0.01); e2 = (p - d) ** 2; sw = s["X"][2] < 0.5
        zones = {"all": m, "deep": m & sw & (s["X"][1] < -0.6), "stalled" if s["completion"] < 0.9 else "completed": m}
        for k, z in zones.items():
            a = acc.setdefault(k, [0.0, 0]); a[0] += e2[z].sum(); a[1] += z.sum()
    return {k: round(100 * math.sqrt(v[0] / max(v[1], 1)), 3) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ch", type=int, default=32); ap.add_argument("--xflip", action="store_true"); ap.add_argument("--prior", default="cutrelax")
    ap.add_argument("--ema", type=float, default=0.0); ap.add_argument("--subset-frac", type=float, default=1.0)
    ap.add_argument("--subset-file", default="", help="json list of train-sample names to use")
    ap.add_argument("--init", default=""); ap.add_argument("--shrink", type=float, default=1.0); ap.add_argument("--perturb", type=float, default=0.0)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    sp = json.loads(SPLITS.read_text()); t0 = time.time()
    train, test = load(sp["train"], a.prior), load(sp["validation"], a.prior)
    if a.subset_frac < 1.0:
        rng = np.random.default_rng(123); idx = rng.permutation(len(train))[: int(round(a.subset_frac * len(train)))]
        train = [train[i] for i in sorted(idx)]
    if a.subset_file:
        keep = set(json.loads(Path(a.subset_file).expanduser().read_text())); train = [s for s in train if s["name"] in keep]
    t_load = time.time() - t0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet(7, a.ch).to(dev)
    if a.init:
        sd = torch.load(Path(a.init).expanduser(), map_location=dev)
        if a.shrink != 1.0 or a.perturb:
            fresh = UNet(7, a.ch).to(dev).state_dict()
            sd = {k: (a.shrink * v + a.perturb * fresh[k]) if v.dtype.is_floating_point else v for k, v in sd.items()}
        net.load_state_dict(sd)
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema)) if a.ema else None
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * max(1, math.ceil(len(train) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    C2 = CELL * CELL; t1 = time.time()
    for ep in range(a.epochs):
        net.train(); order = np.random.permutation(len(train))
        for b in range(0, len(train), a.bs):
            batch = [train[i] for i in order[b:b + a.bs]]
            if a.xflip and np.random.rand() < 0.5:
                batch = [xflip(s) for s in batch]
            X, P, D, Y, (nx, ny) = batchify(batch, flip=np.random.rand() < 0.5)
            X, P, D, Y = X.to(dev), P.to(dev), D.to(dev), Y.to(dev)
            r, ys = net(X); pred = P + r
            w = 1.0 + 9.0 * ((D.abs() > 0.01) | (P.abs() > 0.01)).float()
            l_map = (w * (pred - D) ** 2)[:, :nx, :ny].mean()
            l_vol = ((pred[:, :nx, :ny].sum((1, 2)) * C2 - D[:, :nx, :ny].sum((1, 2)) * C2) ** 2).mean()
            l_s = ((ys - Y) ** 2 * torch.tensor([10.0, 1.0, 0.05], device=dev)).mean()
            loss = l_map + l_vol + l_s
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if ema is not None:
                ema.update_parameters(net)
    t_train = time.time() - t1
    net.eval(); res = {"tag": a.tag, "args": vars(a), "n_train": len(train), "n_test": len(test), "load_s": round(t_load), "train_s": round(t_train)}
    models = {"net": net} | ({"ema": ema.module} if ema is not None else {})
    for name, m in models.items():
        m.eval()
        for tta in ("", "y", "xy"):
            preds = [infer(m, s, dev, tta) for s in test]
            res[f"{name}{'_tta' + tta if tta else ''}"] = pooled(preds, test)
            if tta == "":
                np.savez_compressed(out / f"{a.tag}-{name}-pred.npz", **{s["name"]: p.astype(np.float16) for s, p in zip(test, preds)})
    torch.save(net.state_dict(), out / f"{a.tag}.pt")
    (out / f"{a.tag}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
