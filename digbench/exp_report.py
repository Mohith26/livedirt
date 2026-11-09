"""Summarise the train_exp grid: per method (mean +- sd over seeds) pooled RMS, EMA and TTA variants, seed ensembles,
incremental-retraining churn, and whether ensemble disagreement predicts per-bite error (active-learning signal)."""
import glob, json, math
from pathlib import Path

import numpy as np

from digbench.train_exp import load, pooled, SPLITS

O = Path("~/v4/wsp/exp").expanduser()


def runs(prefix):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(O / f"{prefix}-s[0-9].json")))]


def ms(v):
    return f"{np.mean(v):.2f} +- {np.std(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def main():
    sp = json.loads(SPLITS.read_text())
    test = {"cutrelax": load(sp["validation"], "cutrelax")}
    print("method      | n | all (net) | all (EMA) | all (EMA+TTAxy) | deep | stalled | completed | train s")
    for m in ["base", "xflip", "seq", "xflipseq", "old", "warm", "sp"]:
        R = runs(m)
        if not R:
            continue
        g = lambda k, z: [r[k][z] for r in R if k in r and z in r[k]]
        print(f"{m:11s} | {len(R)} | {ms(g('net','all'))} | {ms(g('ema','all'))} | {ms(g('ema_taxy' if False else 'ema_ttaxy','all'))} | "
              f"{ms(g('ema','deep'))} | {ms(g('ema','stalled'))} | {ms(g('ema','completed'))} | {ms([r['train_s'] for r in R])}")
    # seed ensembles (net weights, no TTA): average the saved predictions
    for m in ["base", "xflip", "seq", "xflipseq"]:
        fs = sorted(glob.glob(str(O / f"{m}-s[0-9]-net-pred.npz")))
        if len(fs) < 2:
            continue
        P = [np.load(f) for f in fs]; prior = "seq" if "seq" in m else "cutrelax"
        if prior not in test:
            test[prior] = load(sp["validation"], prior)
        T = test[prior]
        ens = [np.mean([p[s["name"]].astype(np.float32) for p in P], 0) for s in T]
        print(f"ensemble of {len(fs)} {m}: {pooled(ens, T)}")
    # incremental retraining: churn vs the old model and per-bite regressions
    T = test["cutrelax"]
    for new in ["base", "warm", "sp"]:
        ch, worse, better = [], 0, 0
        for s_ in range(3):
            fo, fn = O / f"old-s{s_}-net-pred.npz", O / f"{new}-s{s_}-net-pred.npz"
            if not (fo.exists() and fn.exists()):
                continue
            po, pn = np.load(fo), np.load(fn)
            for s in T:
                a, b = po[s["name"]].astype(np.float32), pn[s["name"]].astype(np.float32); d = s["delta"]
                m = (np.abs(d) > 0.01) | (np.abs(a) > 0.01) | (np.abs(b) > 0.01)
                ch.append(np.sqrt(((a - b)[m] ** 2).mean()))
                ea, eb = np.sqrt(((a - d)[m] ** 2).mean()), np.sqrt(((b - d)[m] ** 2).mean())
                worse += eb > ea + 0.01; better += eb < ea - 0.01
        if ch:
            print(f"old(75%) -> {new}: churn RMS {100*np.mean(ch):.2f} cm per bite; bites >1 cm worse {worse}, >1 cm better {better}")
    # active-learning signal: does seed disagreement predict per-bite error?
    fs = sorted(glob.glob(str(O / "base-s[0-9]-net-pred.npz")))
    if len(fs) >= 3:
        P = [np.load(f) for f in fs]; dis, err = [], []
        for s in T:
            ps = np.stack([p[s["name"]].astype(np.float32) for p in P]); mean = ps.mean(0); d = s["delta"]
            m = (np.abs(d) > 0.01) | (np.abs(mean) > 0.01)
            dis.append(float(ps.std(0)[m].mean())); err.append(float(np.sqrt(((mean - d)[m] ** 2).mean())))
        dis, err = np.array(dis), np.array(err)
        rk = lambda v: np.argsort(np.argsort(v))
        rho = np.corrcoef(rk(dis), rk(err))[0, 1]
        k = len(err) // 4; top = np.argsort(-dis)[:k]; rnd = np.mean(err)
        print(f"disagreement vs error over {len(err)} validation bites: Spearman {rho:.2f}; mean error of the 25% most-disputed "
              f"bites {100*err[top].mean():.1f} cm vs {100*rnd:.1f} cm all (picking those for labelling targets the hard cases)")


if __name__ == "__main__":
    main()
