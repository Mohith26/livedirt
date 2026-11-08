"""v3 learned-soil surrogate: deployment-matched training and one frozen held-out evaluation for every model.

Data: full-width open-ground windows from digbench/gen_replay.py (no mirroring; 68 x 40 at 5 cm, window frame anchored
as the app). Inputs and outputs match app/src/sim/neuralSoil.ts exactly: x = [(H-1.2)*2, pen0, occ0, pen1, occ1,
(mu-0.5)*4] -> dh (m per 0.1 s), force (Fx, Fz, Ty).

Model options (each is an ablation arm):
  --head free   : dh predicted directly (v1/v2 behaviour)
  --head flux   : dh = -div(q) - s * mask: q = signed fluxes across cell edges (zero across the window boundary),
                  s = sink/source allowed only under bucket-occupied cells (dilated 1 cell). Volume is conserved by
                  construction except what enters/leaves the bucket (= predicted carried volume).
  --floor-train : the deploy-time floor projection (no cell below min(anchored height, lowest bucket point that passed
                  over it)) is applied inside training rollouts too.
  --curriculum 4,16,32 : unroll length per stage (equal epochs per stage), gradients through the whole unroll.
  --noise auto  : input noise std for each stage = rollout RMSE measured at the end of the previous stage.
  --pool occ    : force head pools bottleneck features over bucket-occupied cells (plus global mean).
  --sym         : symmetrise every training window about the bucket centreline (cause-1 ablation: a model that has
                  only seen mirror-symmetric inputs, like v1/v2).

Evaluation (--heldout DIR): the shared frozen set; each episode is rolled out autoregressively from frame 0 with the
deploy floor projection, against LiveDirt cut+relax and persistence on the same bucket rasters. Old ONNX models
(v1, v2) are evaluated with --eval-onnx.

  python -m digbench.train_v3 --data ~/v3/train --heldout ~/v3/heldout --head flux --floor-train \
      --curriculum 4,16,32 --noise auto --pool occ --epochs 30 --tag v3a --out digbench/results
  python -m digbench.train_v3 --heldout ~/v3/heldout --eval-onnx app/public/models/soil-v2.onnx --tag v2
"""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np

SURF = 1.2
CELL = 0.05


# ---------------------------------------------------------------- data
def load(files, sym=False):
    eps = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        H, T = d["H"].astype(np.float32), d["T"].astype(np.float32)
        if sym:
            H = 0.5 * (H + H[:, :, ::-1])
            Tf = T[:, :, ::-1]
            T = np.where(np.isfinite(T) & np.isfinite(Tf), np.fmin(T, Tf), np.where(np.isfinite(T), T, Tf))
        # cumulative floor per frame: min(anchored height, lowest bucket point that has passed over each cell)
        Tn = np.where(np.isfinite(T), T, np.inf)
        floor = np.minimum(H[0][None], np.minimum.accumulate(Tn, axis=0)).astype(np.float32)
        eps.append({"H": H, "T": T, "floor": floor, "F": d["F"].astype(np.float32), "mu": float(d["mu"]),
                    "carried": d["carried"].astype(np.float32), "name": Path(f).stem,
                    "meta": json.loads(str(d["meta"])) if "meta" in d.files else {}})
    return eps


def features(H, Tn, Tn1, mu, torch):
    def tool(T):
        occ = torch.isfinite(T).float()
        pen = torch.where(torch.isfinite(T), (T - H).clamp(-1.0, 1.0), torch.ones_like(H))
        return pen, occ
    p0, o0 = tool(Tn)
    p1, o1 = tool(Tn1)
    m = torch.ones_like(H) * (mu.view(-1, 1, 1) - 0.5) * 4.0
    return torch.stack([(H - SURF) * 2.0, p0, o0, p1, o1, m], 1)


# ---------------------------------------------------------------- model
def build_model(torch, ch=48, head="flux", pool="occ"):
    nn = torch.nn
    F = nn.functional

    class Block(nn.Module):
        def __init__(s, i, o):
            super().__init__()
            s.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())

        def forward(s, x):
            return s.c(x)

    class UNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.head, s.pooling = head, pool
            s.e1, s.e2, s.e3 = Block(6, ch), Block(ch, 2 * ch), Block(2 * ch, 4 * ch)
            s.d2, s.d1 = Block(6 * ch, 2 * ch), Block(3 * ch, ch)
            s.out = nn.Conv2d(ch, 3 if head == "flux" else 1, 1)
            s.force = nn.Sequential(nn.Linear(8 * ch if pool == "occ" else 4 * ch, 128), nn.GELU(), nn.Linear(128, 3))
            s.pool = nn.AvgPool2d(2)

        def forward(s, x):
            nx, ny = x.shape[-2:]
            px, py = (-nx) % 4, (-ny) % 4
            xp = F.pad(x, (0, py, 0, px), mode="replicate")
            a = s.e1(xp)
            b = s.e2(s.pool(a))
            c = s.e3(s.pool(b))
            u = F.interpolate(c, size=b.shape[-2:], mode="nearest")
            d = s.d2(torch.cat([u, b], 1))
            u = F.interpolate(d, size=a.shape[-2:], mode="nearest")
            e = s.d1(torch.cat([u, a], 1))
            o = s.out(e)[..., :nx, :ny] * 0.05
            occ = ((x[:, 2] + x[:, 4]) > 0.5).float()
            if s.head == "flux":
                qx, qy, sk = o[:, 0], o[:, 1], o[:, 2]
                # zero flux across the window boundary: edge i carries qx[i] between cells i and i+1
                qx = torch.cat([qx[:, :-1], torch.zeros_like(qx[:, -1:])], 1)
                qy = torch.cat([qy[:, :, :-1], torch.zeros_like(qy[:, :, -1:])], 2)
                div = qx - torch.cat([torch.zeros_like(qx[:, :1]), qx[:, :-1]], 1) \
                    + qy - torch.cat([torch.zeros_like(qy[:, :, :1]), qy[:, :, :-1]], 2)
                mask = F.max_pool2d(occ[:, None], 3, 1, 1)[:, 0]
                dh = -div - sk * mask
            else:
                dh = o[:, 0]
            g = c.mean(dim=(-2, -1))
            if s.pooling == "occ":
                m = F.max_pool2d(F.pad(occ, (0, py, 0, px))[:, None], 4)  # bottleneck resolution
                w = m / (m.sum(dim=(-2, -1), keepdim=True) + 1e-6)
                g = torch.cat([g, (c * w).sum(dim=(-2, -1))], 1)
            f = s.force(g) * 1e5
            return dh[:, None], f

    return UNet()


def floor_project(torch, Hn, Hprev, floor):
    return torch.maximum(Hn, torch.minimum(floor, Hprev))


# ---------------------------------------------------------------- baselines
def cut_relax_rollout(ep):
    """LiveDirt-style: cut H to min(H, T_{t+1}) under the bucket, then LiveDirt relax at the friction angle."""
    from .solvers.heightfield import Heightfield
    ang = math.degrees(math.atan(ep["mu"]))
    H = ep["H"][0].astype(np.float64)
    out = [H.copy()]
    for t in range(len(ep["H"]) - 1):
        T = ep["T"][t + 1]
        cut = np.where(np.isfinite(T), np.minimum(H, T), H)
        hf = Heightfield(np.full_like(cut, SURF), CELL, ang, ang)
        hf.B[:] = cut
        for _ in range(20):
            if hf.relax(2) <= 2e-4:
                break
        H = hf.h()
        out.append(H.copy())
    return np.array(out)


# ---------------------------------------------------------------- evaluation (shared frozen set)
def evaluate(predict, eps, floor_proj=True, baselines=None):
    """predict(H (1,nx,ny) np, T0, T1, mu) -> dh (nx,ny), f (3,). Returns per-episode rows + summary."""
    rows = []
    for vi, ep in enumerate(eps):
        n = len(ep["H"])
        H = ep["H"][0].astype(np.float64)
        floor = H.copy()
        pred, forces, clipped, floored = [H.copy()], [], 0, 0
        t0 = time.perf_counter()
        for t in range(n - 1):
            dh, f = predict(H, ep["T"][t], ep["T"][t + 1], ep["mu"])
            dh = np.asarray(dh, np.float64)
            bad = ~np.isfinite(dh)
            clipped += int(bad.sum() + (np.abs(dh) > 0.3).sum())
            dh = np.clip(np.where(bad, 0.0, dh), -0.3, 0.3)
            T1 = ep["T"][t + 1]
            floor = np.where(np.isfinite(T1), np.minimum(floor, T1), floor)
            Hn = np.clip(H + dh, SURF - 2.0, SURF + 1.5)
            if floor_proj:
                lim = np.minimum(floor, H)
                floored += int((Hn < lim).sum())
                Hn = np.maximum(Hn, lim)
            H = Hn
            pred.append(H.copy()); forces.append(np.asarray(f, np.float64))
        ms = (time.perf_counter() - t0) / max(1, n - 1) * 1e3
        pred, forces = np.array(pred), np.array(forces)
        truth = ep["H"].astype(np.float64)
        dist = np.abs(truth[-1] - truth[0]) > 0.02
        rm = lambda A: float(np.sqrt(((A[-1] - truth[-1]) ** 2)[dist].mean())) if dist.any() else float("nan")
        rm_all = lambda A: float(np.sqrt(((A - truth) ** 2).mean()))
        vrem = lambda A: float((A[0] - A[-1]).sum() * CELL * CELL)  # window volume removed (m^3)
        Fx = ep["F"][1:, 0].astype(np.float64)
        drag = np.abs(Fx) > 1e3
        lv = baselines[vi] if baselines is not None else cut_relax_rollout(ep)
        pers = np.repeat(truth[:1], n, 0)
        rows.append({
            "episode": ep["name"], "source": ep["meta"].get("source", ""), "mu": ep["mu"], "frames": n, "ms_per_step": ms,
            "final_rmse_disturbed_m": {"model": rm(pred), "cut_relax": rm(lv), "persistence": rm(pers)},
            "rollout_rmse_all_m": {"model": rm_all(pred), "cut_relax": rm_all(lv), "persistence": rm_all(pers)},
            "volume_removed_m3": {"truth": vrem(truth), "model": vrem(pred), "cut_relax": vrem(lv)},
            "volume_err_m3": {"model": vrem(pred) - vrem(truth), "cut_relax": vrem(lv) - vrem(truth)},
            "force_x_rel_err_drag": float(np.abs(forces[drag, 0] - Fx[drag]).mean() / np.abs(Fx[drag]).mean()) if drag.any() else None,
            "peak_Fx_kN": {"truth": float(Fx.max() / 1e3), "model": float(forces[:, 0].max() / 1e3)},
            "clipped": clipped, "floored": floored,
        })
    def agg(key, m, src=None):
        v = [r[key][m] for r in rows if (src is None or r["source"] == src) and r[key][m] == r[key][m]]
        return float(np.mean(v)) if v else None
    summary = {}
    for src in (None, "standard", "chained", "replay"):
        k = src or "all"
        summary[k] = {
            "episodes": sum(1 for r in rows if src is None or r["source"] == src),
            "final_rmse_disturbed_m": {m: agg("final_rmse_disturbed_m", m, src) for m in ("model", "cut_relax", "persistence")},
            "rollout_rmse_all_m": {m: agg("rollout_rmse_all_m", m, src) for m in ("model", "cut_relax", "persistence")},
            "abs_volume_err_m3": {m: (float(np.mean([abs(r["volume_err_m3"][m]) for r in rows if src is None or r["source"] == src])) if rows else None) for m in ("model", "cut_relax")},
            "force_x_rel_err_drag": (lambda v: float(np.mean(v)) if v else None)([r["force_x_rel_err_drag"] for r in rows if (src is None or r["source"] == src) and r["force_x_rel_err_drag"] is not None]),
            "clipped": int(sum(r["clipped"] for r in rows if src is None or r["source"] == src)),
            "floored": int(sum(r["floored"] for r in rows if src is None or r["source"] == src)),
        }
    summary["ms_per_step"] = float(np.mean([r["ms_per_step"] for r in rows])) if rows else None
    return rows, summary


def feature_stats(eps, n_samples=400000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [[] for _ in range(6)]
    per = max(1, n_samples // max(1, sum(len(e["H"]) for e in eps)))
    for e in eps:
        for t in range(len(e["H"]) - 1):
            H, T0, T1 = e["H"][t], e["T"][t], e["T"][t + 1]
            k = rng.integers(0, H.size, per)
            h = H.ravel()[k]
            vals[0].extend(((h - SURF) * 2).tolist())
            for c, T in ((1, T0), (3, T1)):
                tv = T.ravel()[k]
                ok = np.isfinite(tv)
                vals[c].extend(np.clip(tv[ok] - h[ok], -1, 1).tolist())
                vals[c + 1].extend(ok.astype(float).tolist())
            vals[5].extend([(e["mu"] - 0.5) * 4] * per)
    names = ["H", "pen0", "occ0", "pen1", "occ1", "mu"]
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else None
    return {"channels": [{"channel": names[c], "n": len(vals[c]), "p001": q(vals[c], 0.1), "p01": q(vals[c], 1), "p50": q(vals[c], 50),
                          "p99": q(vals[c], 99), "p999": q(vals[c], 99.9)} for c in range(6)]}


# ---------------------------------------------------------------- training
def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--eval-onnx", default=None, help="evaluate an existing ONNX model on the held-out set and exit")
    ap.add_argument("--head", choices=["free", "flux"], default="flux")
    ap.add_argument("--pool", choices=["global", "occ"], default="occ")
    ap.add_argument("--floor-train", action="store_true")
    ap.add_argument("--curriculum", default="4,16,32")
    ap.add_argument("--noise", default="auto")
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-eps", type=int, default=0)
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--out", default="digbench/results")
    args = ap.parse_args()
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    held = load(sorted(f for f in glob.glob(f"{args.heldout}/*.npz") if not Path(f).name.startswith("_")))
    print(f"held-out: {len(held)} episodes ({', '.join(sorted(set(e['meta'].get('source', '?') for e in held)))})", flush=True)
    base_cache = Path(args.heldout) / "_cut_relax_cache.npz"
    if base_cache.exists():
        z = np.load(base_cache)
        baselines = [z[e["name"]] for e in held]
    else:
        baselines = [cut_relax_rollout(e) for e in held]
        np.savez_compressed(base_cache, **{e["name"]: b for e, b in zip(held, baselines)})

    if args.eval_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.eval_onnx, providers=["CPUExecutionProvider"])

        def predict(H, T0, T1, mu):
            x = features(torch.tensor(H[None], dtype=torch.float32), torch.tensor(T0[None]), torch.tensor(T1[None]),
                         torch.tensor([mu], dtype=torch.float32), torch).numpy()
            dh, f = sess.run(None, {"x": x})
            return dh[0, 0], f[0]
        rows, summary = evaluate(predict, held, baselines=baselines)
        (outd / f"heldout-{args.tag}.json").write_text(json.dumps({"tag": args.tag, "model": args.eval_onnx, "summary": summary, "episodes": rows}, indent=1))
        print(json.dumps(summary, indent=1))
        return

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(f for f in sum([glob.glob(f"{d}/*.npz") for d in args.data], []) if not Path(f).name.startswith("_"))
    rng = np.random.default_rng(args.seed)
    if args.max_train_eps:
        rng.shuffle(files); files = sorted(files[:args.max_train_eps])
    train = load(files, sym=args.sym)
    stats = feature_stats(train)
    print(f"train {len(train)} episodes on {dev}; head={args.head} pool={args.pool} floor_train={args.floor_train} "
          f"curriculum={args.curriculum} noise={args.noise} sym={args.sym}", flush=True)
    model = build_model(torch, head=args.head, pool=args.pool).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    tens = lambda a: torch.tensor(np.asarray(a), device=dev)
    stages = [int(s) for s in args.curriculum.split(",")]
    per_stage = [args.epochs // len(stages)] * len(stages)
    per_stage[-1] += args.epochs - sum(per_stage)
    noise = 0.005 if args.noise == "auto" else float(args.noise)
    log, t_train, ep_i = [], time.perf_counter(), 0
    for si, U in enumerate(stages):
        wins = [(e, t) for e in range(len(train)) for t in range(len(train[e]["H"]) - U)]
        bs = max(4, args.batch * 4 // max(4, U))  # keep memory ~constant as the unroll grows
        for _ in range(per_stage[si]):
            rng.shuffle(wins)
            tot, nb = 0.0, 0
            for b0 in range(0, len(wins), bs):
                batch = wins[b0:b0 + bs]
                g = lambda key, k: tens(np.stack([train[e][key][t + k] for e, t in batch]))
                H = g("H", 0)
                mu = tens(np.array([train[e]["mu"] for e, t in batch], np.float32))
                H = H + noise * torch.randn_like(H)
                floor = g("floor", 0)
                c0 = tens(np.array([train[e]["carried"][t] for e, t in batch], np.float32))
                loss = 0.0
                for k in range(U):
                    Tn, Tn1 = g("T", k), g("T", k + 1)
                    dh, f = model(features(H, Tn, Tn1, mu, torch))
                    Hn = H + dh[:, 0]
                    if args.floor_train:
                        floor = torch.minimum(floor, torch.where(torch.isfinite(Tn1), Tn1, torch.full_like(Tn1, 1e9)))
                        Hn = floor_project(torch, Hn, H, floor)
                    H = Hn
                    Ht, Ft = g("H", k + 1), g("F", k + 1)
                    loss = loss + ((H - Ht) ** 2).mean() / 0.01 ** 2 + (((f - Ft) / 1e4) ** 2).mean() * 0.1
                loss = loss / U
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss); nb += 1
            sched.step()
            log.append({"epoch": ep_i, "unroll": U, "noise": noise, "loss": tot / max(1, nb)})
            print(f"epoch {ep_i} (unroll {U}, noise {noise:.4f}, batch {bs}): loss {tot / max(1, nb):.3f}", flush=True)
            ep_i += 1
        # measured rollout error over this stage's horizon on a training subset -> noise for the next stage
        if args.noise == "auto" and si < len(stages) - 1:
            model.eval()
            with torch.no_grad():
                errs = []
                for e in rng.choice(len(train), min(40, len(train)), replace=False):
                    ep = train[e]
                    if len(ep["H"]) <= stages[si + 1]:
                        continue
                    t = int(rng.integers(0, len(ep["H"]) - stages[si + 1]))
                    H = tens(ep["H"][t][None]); mu = tens(np.array([ep["mu"]], np.float32)); floor = tens(ep["floor"][t][None])
                    for k in range(stages[si]):
                        T0, T1 = tens(ep["T"][t + k][None]), tens(ep["T"][t + k + 1][None])
                        dh, _ = model(features(H, T0, T1, mu, torch))
                        Hn = H + dh[:, 0]
                        floor = torch.minimum(floor, torch.where(torch.isfinite(T1), T1, torch.full_like(T1, 1e9)))
                        H = floor_project(torch, Hn, H, floor) if args.floor_train else Hn
                    errs.append(float(((H[0] - tens(ep["H"][t + stages[si]])) ** 2).mean().sqrt()))
                noise = float(min(0.05, np.mean(errs))) if errs else noise
            model.train()
    t_train = time.perf_counter() - t_train

    model.eval()

    def predict(H, T0, T1, mu):
        with torch.no_grad():
            x = features(tens(np.asarray(H, np.float32)[None]), tens(T0[None]), tens(T1[None]), tens(np.array([mu], np.float32)), torch)
            dh, f = model(x)
            return dh[0, 0].cpu().numpy(), f[0].cpu().numpy()
    rows, summary = evaluate(predict, held, baselines=baselines)
    res = {"tag": args.tag, "summary": summary, "episodes": rows, "params": vars(args), "n_params": nparam,
           "train_episodes": len(train), "train_s": t_train, "device": dev, "log": log}
    (outd / f"heldout-{args.tag}.json").write_text(json.dumps(res, indent=1))
    (outd / f"soil-{args.tag}.stats.json").write_text(json.dumps(stats, indent=1))
    torch.save(model.state_dict(), outd / f"soil-{args.tag}.pt")
    model.cpu()
    ex = features(torch.tensor(held[0]["H"][0][None]), torch.tensor(held[0]["T"][0][None]), torch.tensor(held[0]["T"][1][None]),
                  torch.tensor([0.5]), torch)
    torch.onnx.export(model, ex, str(outd / f"soil-{args.tag}.onnx"), input_names=["x"], output_names=["dh", "force"],
                      opset_version=17, dynamo=False)
    print("summary:", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
