"""DigBench task 2: plate intrusion stress maps vs Li et al. 2013 (3 mm glass spheres).

  python -m digbench.run_plate --arch cuda --grid coarse --tag mpm3d
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import references
from .run_collapse import device_name, git_rev
from .solvers import mpm3d

ROOT = Path(__file__).resolve().parent
GRIDS = {
    "smoke": [(0, 90), (90, 0)],
    "coarse": [(b, g) for b in (-60, -30, 0, 30, 60, 90) for g in (-60, -30, 0, 30, 60, 90)],
}


def alpha_from_series(ser, A, skip=0.015):
    """Least-squares slope through the origin of stress vs depth, after `skip` m of travel, depth > 1 cm."""
    t, path, z, fx, fz = ser.T
    m = (path > skip) & (z > 0.01)
    if m.sum() < 5:
        return float("nan"), float("nan"), 0
    sx, sz = fx[m] / A, fz[m] / A
    zz = z[m]
    ax = float((sx * zz).sum() / (zz * zz).sum()) / 1e6
    az = float((sz * zz).sum() / (zz * zz).sum()) / 1e6
    return ax, az, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--grid", default="coarse")
    ap.add_argument("--dx", type=float, default=0.003)
    ap.add_argument("--speed", type=float, default=0.1)
    ap.add_argument("--mu-f", type=float, default=references.RFT_3MM_CP["mu_f"])
    ap.add_argument("--mu-tool", type=float, default=references.RFT_3MM_CP["plate_mu"])
    ap.add_argument("--E", type=float, default=1.0e6)
    ap.add_argument("--tag", default="mpm3d")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    M = references.RFT_3MM_CP
    A = 0.0381 * 0.0254
    rows = []
    for b, g in GRIDS[args.grid]:
        r = mpm3d.run_plate(beta_deg=b, gamma_deg=g, mu_f=args.mu_f, mu_tool=args.mu_tool, rho_bulk=M["rho_bulk"],
                            E=args.E, dx=args.dx, speed=args.speed, arch=args.arch)
        ser = r.pop("series")
        ax, az, n = alpha_from_series(ser, A)
        # Convention: the simulator's beta rotates the plate CCW from +x with motion toward -x, so beta > 0 tilts the
        # leading face up. Li et al.'s beta > 0 tilts it down (their large positive drag-induced lift at beta=+45,
        # gamma=0). Same physical plate => Li beta = -beta here. Checked on the convention-invariant cases
        # (beta = 0, 90), which matched before any mapping.
        rx, rz = references.rft_alpha(math.radians(-b), math.radians(g))
        rows.append({"beta": b, "gamma": g, "alpha_x": ax, "alpha_z": az, "ref_x": rx, "ref_z": rz, "n_samples": n,
                     **{k: r[k] for k in ("n_particles", "grid", "dt", "steps", "wall_s", "rtf", "start_depth", "travel")},
                     "series": ser[::2].round(6).tolist()})
        print(f"beta={b:>4} gamma={g:>4}: alpha_x {ax:+.3f} (ref {rx:+.3f})  alpha_z {az:+.3f} (ref {rz:+.3f}) N/cm^3  "
              f"N={r['n_particles']} steps={r['steps']} wall={r['wall_s']:.0f}s", flush=True)
    ok = [x for x in rows if not math.isnan(x["alpha_z"])]
    sx = np.array([x["alpha_x"] for x in ok]); sz = np.array([x["alpha_z"] for x in ok])
    rx = np.array([x["ref_x"] for x in ok]); rz = np.array([x["ref_z"] for x in ok])
    sim = np.concatenate([sx, sz]); ref = np.concatenate([rx, rz])
    peak = max(abs(ref).max(), 1e-9)
    summary = {
        "nrmse": float(np.sqrt(((sim - ref) ** 2).mean()) / peak),  # RMSE / max |ref|
        "corr": float(np.corrcoef(sim, ref)[0, 1]) if len(ok) > 2 else float("nan"),
        "gain": float((sim * ref).sum() / (ref * ref).sum()),  # best scalar k with sim ~ k*ref
        "alpha_z_0_90": next((x["alpha_z"] for x in rows if x["beta"] == 0 and x["gamma"] == 90), None),
        "alpha_z_0_90_measured": M["measured_alpha_z_0_90"],
    }
    out = {"task": "plate_rft", "solver": "mls-mpm-dp-3d", "tag": args.tag, "reference": M["source"],
           "device": device_name(args.arch), "git": git_rev(), "params": {**vars(args)}, "summary": summary, "cases": rows}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"plate-{args.tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print("summary:", json.dumps(summary), "->", p)


if __name__ == "__main__":
    main()
