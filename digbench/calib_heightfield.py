"""Item 5: tune the simple (heightfield-style) bite model to the MPM teacher, per bite.

Simple model with two parameters, applied to a bite's swept envelope on the app terrain:
  * soil above the envelope is cut; a fraction `f` of it is carried (up to the bucket's capacity), the rest is left as
    loose spill in a ring around the swept area (the berm the bucket pushes aside),
  * the surface then relaxes to `repose` (deg).
Grid search over (f, repose offset from the preset's angle) minimising the final-surface RMSE on disturbed cells over
MPM bite samples; reports the fitted values with the carried-volume error, against the untuned cut+relax prior.

  python -m digbench.calib_heightfield --data ~/v4/scoops/* --out results/v4/calib-heightfield.json
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from .scoop_model import CELL, relax, swept_envelope


def simple_bite(H, env, f, repose, cap=0.95, ring=0.3):
    touched = np.isfinite(env)
    cut = np.where(touched, np.maximum(0.0, H - env), 0.0)
    vol = cut.sum() * CELL * CELL
    carried = min(cap, f * vol)
    spill = vol - carried
    H1 = H - cut
    if spill > 0:
        ringm = touched.copy()
        for _ in range(int(ring / CELL)):
            ringm = ringm | np.roll(ringm, 1, 0) | np.roll(ringm, -1, 0) | np.roll(ringm, 1, 1) | np.roll(ringm, -1, 1)
        ringm &= ~touched
        if ringm.any():
            H1 = H1 + ringm * spill / (ringm.sum() * CELL * CELL)
    H1 = relax(H1, repose)
    return H1 - H, carried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(set(f for d in args.data for f in glob.glob(str(Path(d) / "*.npz"))))
    S, seen = [], set()
    for f in files:
        z = np.load(f)
        soil = json.loads(str(z["soil"])); meta = json.loads(str(z["meta"]))
        T = z["terrain"].astype(float)
        key = hash(T.tobytes())
        if key in seen:
            continue
        seen.add(key)
        env = swept_envelope(z["poses"], *T.shape, float(z["margin"]))
        S.append({"T": T, "env": env, "d": z["delta"].astype(float), "carried": meta["carried"], "phi": soil.get("phi", 30.0), "soil": soil.get("preset", soil.get("phi"))})
    print(f"{len(S)} bites", flush=True)

    def score(f, dphi):
        e, ce = [], []
        for s in S:
            d, c = simple_bite(s["T"], s["env"], f, s["phi"] + dphi)
            m = (np.abs(d) > 0.01) | (np.abs(s["d"]) > 0.01)
            e.append(float(np.sqrt(((d - s["d"]) ** 2)[m].mean())) if m.any() else 0.0)
            ce.append(abs(c - s["carried"]))
        return float(np.mean(e)), float(np.mean(ce))

    base = score(1.0, 0.0)
    grid = [(f, dp) for f in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0) for dp in (-6.0, -3.0, 0.0, 3.0)]
    res = {f"{f}|{dp}": score(f, dp) for f, dp in grid}
    best = min(res, key=lambda k: res[k][0])
    out = {"n_bites": len(S), "untuned_prior": {"f": 1.0, "dphi": 0.0, "rmse_m": base[0], "carried_mae_m3": base[1]},
           "best": {"f": float(best.split("|")[0]), "dphi": float(best.split("|")[1]), "rmse_m": res[best][0], "carried_mae_m3": res[best][1]},
           "grid": {k: {"rmse_m": v[0], "carried_mae_m3": v[1]} for k, v in res.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_bites", "untuned_prior", "best")}))


if __name__ == "__main__":
    main()
