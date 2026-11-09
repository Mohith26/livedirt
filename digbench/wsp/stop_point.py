"""Stop-point calculator: where along the planned path will the bucket stop? Uses only what is known before the scoop
(terrain, planned poses, soil); fitted on the TRAIN tier only, scored on VALIDATION.

Walk the planned poses in time order. At each pose k:
  V_k  soil the bucket would have had to remove so far = sum over cells of max(0, terrain - lowest bucket point so far)
  F_k  passive-wedge digging force at the lip (fundamental earthmoving equation, per metre of width):
       0.5 rho g d^2 Kp + 2 c d sqrt(Kp), d = lip depth below the terrain under the bucket centreline
Predicted stop = first pose where V_k > V_th or max(F_0..F_k) > F_th; predicted completion = t_stop / t_end (1 if never).

The input channel ("beyond the predicted stop", 0..1) marks every cell the bucket would first reach AFTER the predicted
stop: clip((first-touch time fraction - predicted completion) / 0.05, 0, 1); 0 on cells reached before it and untouched.

  python -m digbench.wsp.stop_point --fit      # fit V_th, F_th on train, report train + validation, write params
"""
import argparse, json, math
from functools import lru_cache
from pathlib import Path

import numpy as np

from digbench.scoop_model import CELL, _bucket_points
from digbench.attitude import lip_offset

PARAMS = Path("~/v4/wsp/stop_params.json").expanduser()
SPLITS = Path("~/v4/wsp/splits.json").expanduser()


def walk(terrain, poses, soil, M, C=CELL, nsub=150):
    """Per sub-sampled pose: time fraction, removable volume so far, peak force so far; plus first-touch map."""
    import digbench.scoop_model as sm
    if sm._BP is None:
        sm._BP = _bucket_points()
    BP = sm._BP
    P = np.asarray(poses, np.float64); nx, ny = terrain.shape
    t_end = float(P[-1, 0]) if P[-1, 0] > 0 else 1.0
    idx = list(range(0, len(P), max(1, len(P) // nsub)))
    if idx[-1] != len(P) - 1:
        idx.append(len(P) - 1)
    phi = math.radians(float(soil.get("phi", math.degrees(math.atan(soil.get("mu", 0.6))))))
    c = float(soil.get("c", 0.0)); rho = float(soil.get("rho", 1600.0)); Kp = math.tan(math.pi / 4 + phi / 2) ** 2
    jc = int((1.0 + M) / C)
    env = np.full((nx, ny), np.inf); first = np.full((nx, ny), np.nan)
    tf, V, F = [], [], []; fmax = 0.0
    for k in idx:
        px, pz, pa = P[k, 1], P[k, 2], P[k, 3]
        ca, sa = math.cos(pa), math.sin(pa)
        x = ca * BP[:, 0] - sa * BP[:, 2]; z = sa * BP[:, 0] + ca * BP[:, 2]; y = BP[:, 1]
        ro = P[k, 5] if P.shape[1] > 5 else 0.0
        if ro != 0.0:
            lx, lz = lip_offset(pa); cr, sr = math.cos(ro), math.sin(ro); yy, zz = y, z - lz
            y, z = cr * yy - sr * zz, sr * yy + cr * zz + lz
        i = np.floor((px + x + M) / C).astype(int); j = np.floor((y + 1.0 + M) / C).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        new = np.zeros((nx, ny), bool); new[i[ok], j[ok]] = True
        t = float(P[k, 0]) / t_end
        first[new & np.isnan(first)] = t
        np.minimum.at(env, (i[ok], j[ok]), pz + z[ok])
        V.append(float(np.where(np.isfinite(env), np.maximum(0.0, terrain - env), 0.0).sum() * C * C))
        lx, lz = lip_offset(pa); il = int((px + lx + M) / C)
        d = max(0.0, float(terrain[il, max(0, jc - 8):jc + 9].mean()) - (pz + lz)) if 0 <= il < nx else 0.0
        fmax = max(fmax, 0.5 * rho * 9.81 * d * d * Kp + 2 * c * d * math.sqrt(Kp))
        tf.append(t); F.append(fmax / 1e3)
    return np.array(tf), np.array(V), np.array(F), first


def predict_completion(tf, V, F, v_th, f_th):
    hit = (V > v_th) | (F > f_th)
    return float(tf[np.argmax(hit)]) if hit.any() else 1.0


def channel(first, comp_pred, ramp=0.05):
    return np.where(np.isfinite(first), np.clip((first - comp_pred) / ramp, 0.0, 1.0), 0.0).astype(np.float32)


@lru_cache(maxsize=None)
def _walk_file(f):
    z = np.load(f); soil = json.loads(str(z["soil"]))
    return walk(z["terrain"].astype(float), z["poses"], soil, float(z["margin"])), json.loads(str(z["meta"]))["completion"]


def params():
    return json.loads(PARAMS.read_text())


def for_file(f):
    """(beyond-stop channel, predicted completion) for one sample file, with the fitted parameters."""
    (tf, V, F, first), _ = _walk_file(f); p = params()
    cp = predict_completion(tf, V, F, p["v_th"], p["f_th"])
    return channel(first, cp), cp


def score(files, v_th, f_th):
    pc, tc = [], []
    for f in files:
        (tf, V, F, _), comp = _walk_file(f)
        pc.append(predict_completion(tf, V, F, v_th, f_th)); tc.append(min(comp, 1.0))
    pc, tc = np.array(pc), np.array(tc); st = tc < 0.9; ps = pc < 0.9
    return {"n": len(files), "stalled": int(st.sum()), "stall_accuracy": float((ps == st).mean()),
            "recall": float((ps & st).sum() / max(1, st.sum())), "precision": float((ps & st).sum() / max(1, ps.sum())),
            "mae_completion": float(np.abs(pc - tc).mean()),
            "stalled_median_stop_error": float(np.median(np.abs(pc - tc)[st])) if st.any() else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fit", action="store_true"); a = ap.parse_args()
    sp = json.loads(SPLITS.read_text())
    best = None
    for v_th in np.r_[np.arange(0.3, 3.01, 0.05), 99.0]:
        for f_th in np.r_[np.arange(2.0, 80.0, 1.0), 9999.0]:
            s = score(sp["train"], v_th, f_th); key = s["mae_completion"]
            if best is None or key < best[0]:
                best = (key, float(v_th), float(f_th), s)
    _, v_th, f_th, s_tr = best
    out = {"v_th": v_th, "f_th": f_th, "fit_on": "train", "train": s_tr,
           "practice": score(sp["practice"], v_th, f_th), "validation": score(sp["validation"], v_th, f_th),
           "baseline_never_stop_validation": score(sp["validation"], 99.0, 9999.0)}
    PARAMS.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
