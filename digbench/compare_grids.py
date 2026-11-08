"""Compare final terrains of closed-loop runs (bench --dump-grid JSONs) against a reference run (the MPM-in-the-loop
teacher). Same scenario, same controller, different soil backend: how close does each backend's closed-loop outcome
come to the teacher's?

  python -m digbench.compare_grids ref.json other1.json other2.json ...
Reports per file: RMSE of the final surface vs the reference over (a) design cells, (b) all cells within 1 m of the
design (the disturbed neighbourhood), mean signed difference, and the cut-volume difference.
"""
import json
import sys

import numpy as np


def load(f):
    g = json.load(open(f))
    sh = (g["nj"], g["ni"])
    return {k: np.array(g[k], float).reshape(sh) for k in ("H", "D", "H0", "F")} | {"cell": g["cell"], "i0": g["i0"], "j0": g["j0"]}


def compare(ref, o):
    assert ref["H"].shape == o["H"].shape and ref["i0"] == o["i0"] and ref["j0"] == o["j0"], "grids differ (different scenario?)"
    des = ref["D"] > -900
    near = des.copy()
    for _ in range(int(round(1.0 / ref["cell"]))):
        near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
    d = o["H"] - ref["H"]
    a = ref["cell"] ** 2
    return {"rmse_design_m": float(np.sqrt((d[des] ** 2).mean())), "rmse_near_m": float(np.sqrt((d[near] ** 2).mean())),
            "mean_diff_design_m": float(d[des].mean()), "cut_volume_diff_m3": float(((ref["H0"] - o["H"]) - (ref["H0"] - ref["H"]))[near].sum() * a),
            "ref_cut_m3": float((ref["H0"] - ref["H"])[near].clip(0).sum() * a)}


if __name__ == "__main__":
    ref = load(sys.argv[1])
    for f in sys.argv[2:]:
        print(f, json.dumps({k: round(v, 4) for k, v in compare(ref, load(f)).items()}))
