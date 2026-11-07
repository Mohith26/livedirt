"""DigBench task 1: quasi-2D granular column collapse vs Sarlin et al. 2021.

Usage:
  python -m digbench.run_collapse --arch gpu --dx 0.0025 --tag mpm-baseline
  python -m digbench.run_collapse --arch cpu --cases 1,3 --dx 0.005 --tag smoke
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from . import metrics, references
from .solvers import mpm_dp

ROOT = Path(__file__).resolve().parent


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def device_name(arch):
    if arch in ("gpu", "cuda"):
        try:
            return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
        except Exception:
            pass
    return f"{platform.machine()} {arch}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--dx", type=float, default=0.0025)
    ap.add_argument("--ppc", type=int, default=2, help="particles per cell side")
    ap.add_argument("--E", type=float, default=1.0e6)
    ap.add_argument("--phi", type=float, default=references.COLLAPSE["repose_deg"])
    ap.add_argument("--yield-fit", default="plane_strain", choices=["plane_strain", "klar3d"])
    ap.add_argument("--floor", default="sticky", choices=["sticky", "coulomb"])
    ap.add_argument("--floor-mu", type=float, default=None)
    ap.add_argument("--cases", default="all", help="comma list of aspect ratios, or all")
    ap.add_argument("--tag", default="mpm")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    cases = references.COLLAPSE_CASES
    if args.cases != "all":
        want = {float(s) for s in args.cases.split(",")}
        cases = [c for c in cases if c["a"] in want]

    rows = []
    for case in cases:
        t0 = time.perf_counter()
        r = mpm_dp.run_column_collapse(case, dx=args.dx, ppc_side=args.ppc, E=args.E, phi_deg=args.phi, yield_fit=args.yield_fit,
                                       floor=args.floor, floor_mu=args.floor_mu, arch=args.arch)
        pos = r.pop("positions")
        r.pop("snapshots")
        xc, h = metrics.profile_from_particles(pos[:, 0], pos[:, 1], args.dx, r["spacing"])
        dep = metrics.deposit(xc, h, case["L0"], references.COLLAPSE["grain_d"])
        sc = metrics.score(case, dep, references.collapse_runout, references.collapse_height)
        area = float(np.trapezoid(h, xc)) if hasattr(np, "trapezoid") else float(np.trapz(h, xc))
        row = {**case, **dep, **sc, **{k: r[k] for k in ("n_particles", "grid", "dt", "steps", "t_sim", "wall_s", "rtf", "stopped", "volume_ratio")},
               "area_ratio": area / (case["H0"] * case["L0"]), "total_s": time.perf_counter() - t0,
               "profile": {"x": xc[::2].round(5).tolist(), "h": h[::2].round(5).tolist()}}
        rows.append(row)
        print(f"a={case['a']:>5}: runout {sc['runout_sim']:.2f} vs {sc['runout_ref']:.2f} ({sc['runout_relerr']:+.1%})  "
              f"height {sc['height_sim']:.2f} vs {sc['height_ref']:.2f} ({sc['height_relerr']:+.1%})  "
              f"N={r['n_particles']} steps={r['steps']} wall={r['wall_s']:.1f}s rtf={r['rtf']:.3f} {r['stopped']}", flush=True)

    out = {
        "task": "column_collapse",
        "solver": "mls-mpm-drucker-prager-2d",
        "tag": args.tag,
        "reference": references.COLLAPSE["source"],
        "device": device_name(args.arch),
        "git": git_rev(),
        "params": {**r["params"]},
        "summary": {
            "runout_mae_rel": float(np.mean([abs(x["runout_relerr"]) for x in rows])),
            "height_mae_rel": float(np.mean([abs(x["height_relerr"]) for x in rows])),
        },
        "cases": rows,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"collapse-{args.tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"summary: runout MAE {out['summary']['runout_mae_rel']:.1%}, height MAE {out['summary']['height_mae_rel']:.1%} -> {p}")


if __name__ == "__main__":
    main()
