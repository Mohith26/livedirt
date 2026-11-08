"""Calibrate the uncertainty gate on the frozen held-out set (never on the loop gate).

For an ensemble of ONNX members, roll each held-out episode with the ensemble mean (deploy-identical floor projection)
and record, per 0.1 s step, the mean per-cell spread of dh over bucket-near cells (dilated 2 cells, as in
app/src/sim/neuralSoil.ts) and the fraction of bucket-near cells whose inputs fall outside the training range
[p0.1, p99.9] (from soil-*.stats.json). The gate thresholds are the q-th percentile of each over held-out steps.

  python -m digbench.gate_calib --heldout /tmp/v3/heldout --members a.onnx,b.onnx,c.onnx --stats a.stats.json --q 95
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from .train_v3 import SURF, load


def feats_np(H, T0, T1, mu):
    def tool(T):
        occ = np.isfinite(T).astype(np.float32)
        pen = np.where(np.isfinite(T), np.clip(T - H, -1, 1), 1.0).astype(np.float32)
        return pen, occ
    p0, o0 = tool(T0)
    p1, o1 = tool(T1)
    return np.stack([(H - SURF) * 2, p0, o0, p1, o1, np.full_like(H, (mu - 0.5) * 4)], 0)[None].astype(np.float32)


def near_mask(x):
    occ = (x[0, 2] > 0.5) | (x[0, 4] > 0.5)
    m = occ.copy()
    for di in range(-2, 3):
        for dj in range(-2, 3):
            m |= np.roll(np.roll(occ, di, 0), dj, 1)
    return m


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--members", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--q", type=float, default=95.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sess = [ort.InferenceSession(m, providers=["CPUExecutionProvider"]) for m in args.members.split(",")]
    st = json.loads(Path(args.stats).read_text())["channels"]
    lo = np.array([c["p001"] for c in st]); hi = np.array([c["p999"] for c in st])
    eps = load(sorted(f for f in glob.glob(f"{args.heldout}/*.npz") if not Path(f).name.startswith("_")))
    stds, oors = [], []
    for ep in eps:
        H = ep["H"][0].astype(np.float64); floor = H.copy()
        for t in range(len(ep["H"]) - 1):
            x = feats_np(H.astype(np.float32), ep["T"][t], ep["T"][t + 1], ep["mu"])
            outs = [s.run(None, {"x": x})[0][0, 0] for s in sess]
            dh = np.mean(outs, 0); sd = np.std(outs, 0)
            nm = near_mask(x)
            if nm.any():
                stds.append(float(sd[nm].mean()))
                bad = np.zeros_like(nm)
                for c in (0, 1, 3):
                    bad |= (x[0, c] < lo[c]) | (x[0, c] > hi[c])
                oors.append(float((bad & nm).sum() / nm.sum()))
            T1 = ep["T"][t + 1]
            floor = np.where(np.isfinite(T1), np.minimum(floor, T1), floor)
            Hn = np.clip(H + np.clip(np.nan_to_num(dh), -0.3, 0.3), SURF - 2.0, SURF + 1.5)
            H = np.maximum(Hn, np.minimum(floor, H))
    res = {"members": args.members.split(","), "q": args.q, "steps": len(stds),
           "maxStd": float(np.percentile(stds, args.q)), "maxOOR": float(np.percentile(oors, args.q)),
           "std_percentiles": {p: float(np.percentile(stds, p)) for p in (50, 90, 95, 99)},
           "oor_percentiles": {p: float(np.percentile(oors, p)) for p in (50, 90, 95, 99)}}
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
