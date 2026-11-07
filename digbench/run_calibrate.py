"""DigBench task 5: per-site calibration. Observe one slump, infer soil friction, predict digging force.

Protocol (synthetic ground truth, so the answer is known):
  1. A hidden soil friction mu_true generates an "observed" final surface profile of a column collapse (a = 2),
     plus Gaussian noise (sigma = 1 cm, roughly drone photogrammetry) on a 1 cm grid.
  2. mu is recovered by golden-section search on the profile RMSE (2D MPM, same constitutive convention as 3D).
  3. The recovered mu predicts the full-scale bucket drag force (3D MPM, task 3 cycle); compared with the
     force produced by mu_true. Also reported: what you'd predict with a textbook default mu = 0.5.

Also (real data): fit the friction angle of the quasi-2D collapse model to the Sarlin et al. 2021 fits and
compare with their measured repose angle (23.5 +/- 1.2 deg).

  python -m digbench.run_calibrate --arch cuda
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import metrics, references
from .run_collapse import device_name, git_rev
from .solvers import mpm_dp

ROOT = Path(__file__).resolve().parent
OBS_CASE = {"a": 2.0, "H0": 0.2, "L0": 0.1}


def collapse_profile(mu, arch, dx=0.0025, case=OBS_CASE, quasi2d=False):
    kw = dict(dx=dx, phi_deg=math.degrees(math.atan(mu)), yield_fit="dp_tan", arch=arch)
    if quasi2d:
        kw.update(wall_mu=0.21, side_W=0.15, side_mu=0.21)
    r = mpm_dp.run_column_collapse(case, **kw)
    pos = r["positions"]
    xc, h = metrics.profile_from_particles(pos[:, 0], pos[:, 1], dx, r["spacing"])
    grid = np.arange(0.0, 1.0, 0.01)
    return np.interp(grid, xc, h, right=0.0), xc, h


def golden(f, lo, hi, tol=0.004):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    evals = 2
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = f(d)
        evals += 1
    return (a + b) / 2, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--truths", default="0.3,0.45,0.6")
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--bucket-dx", type=float, default=0.04)
    ap.add_argument("--skip-bucket", action="store_true")
    ap.add_argument("--tag", default="calib")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = {"task": "calibration", "tag": args.tag, "device": device_name(args.arch), "git": git_rev(), "params": vars(args),
           "synthetic": [], "real": {}}
    pth = Path(args.out) / f"calib-{args.tag}.json"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # --- real data: fit friction angle of the quasi-2D model to Sarlin et al. fits ---
    def real_loss(phi):
        errs = []
        for c in references.COLLAPSE_CASES:
            r = mpm_dp.run_column_collapse(c, dx=0.0025, phi_deg=phi, wall_mu=0.21, side_W=0.15, side_mu=0.21, arch=args.arch)
            pos = r["positions"]
            xc, h = metrics.profile_from_particles(pos[:, 0], pos[:, 1], 0.0025, r["spacing"])
            dep = metrics.deposit(xc, h, c["L0"], references.COLLAPSE["grain_d"])
            sc = metrics.score(c, dep, references.collapse_runout, references.collapse_height)
            errs.append(abs(sc["runout_relerr"]) + abs(sc["height_relerr"]))
        return float(np.mean(errs))
    phi_fit, n_eval = golden(real_loss, 18.0, 34.0, tol=0.5)
    out["real"] = {"phi_fit_deg": phi_fit, "loss_at_fit": real_loss(phi_fit), "measured_repose_deg": 23.5, "measured_sd_deg": 1.2,
                   "evals": n_eval, "note": "plane-strain MC convention (sin phi) with quasi-2D sidewall friction, as in task 1"}
    print("real-data fit:", json.dumps(out["real"]), flush=True)
    pth.write_text(json.dumps(out))

    # --- synthetic inversion + transfer to bucket force ---
    if not args.skip_bucket:
        from .run_bucket import BED, SURF, bucket_boxes, path
        from .solvers.sim3d import Sim3D

        def bucket_drag(mu):
            sim = Sim3D(BED, args.bucket_dx, mu_f=mu, arch=args.arch, tool_speed_max=1.5,
                        max_particles=int(1.15 * BED[0] * BED[1] * SURF / (args.bucket_dx / 2) ** 3) + 1000)
            sim.fill_box((0, 0, 0), (BED[0], BED[1], SURF))
            sim.set_tool(bucket_boxes(args.bucket_dx))
            p0, a0 = path(0.0)
            sim.set_pose(p0, a0)
            sim.settle(0.3)
            ser = sim.drive(lambda t: path(t), 3.0, sample_every=0.02)["series"]
            m = (ser[:, 1] > 1.2) & (ser[:, 1] < 3.0)
            return float(ser[m, 6].mean())
        f_default = bucket_drag(0.5)
        out["bucket_drag_default_mu0.5_N"] = f_default

    for mu_t in [float(s) for s in args.truths.split(",")]:
        clean, _, _ = collapse_profile(mu_t, args.arch)
        f_true = bucket_drag(mu_t) if not args.skip_bucket else None
        for rep in range(args.reps):
            obs = clean + rng.normal(0, args.noise, clean.shape) * (clean > 0)
            loss = lambda mu: float(np.sqrt(((collapse_profile(mu, args.arch)[0] - obs) ** 2).mean()))
            mu_hat, n_eval = golden(loss, 0.15, 0.9)
            row = {"mu_true": mu_t, "rep": rep, "mu_hat": mu_hat, "mu_err": mu_hat - mu_t, "evals": n_eval}
            if not args.skip_bucket:
                f_hat = bucket_drag(mu_hat)
                row.update({"drag_true_N": f_true, "drag_pred_N": f_hat, "drag_err": (f_hat - f_true) / f_true,
                            "drag_err_default_mu0.5": (f_default - f_true) / f_true})
            out["synthetic"].append(row)
            print(json.dumps(row), flush=True)
            pth.write_text(json.dumps(out))
    print("->", pth)


if __name__ == "__main__":
    main()
