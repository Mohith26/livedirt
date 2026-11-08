"""DigBench Project Chrono baseline runner (external open-source solver). Runs inside the `chrono` conda env, which has
pychrono but not taichi, so the taichi-dependent MPM modules are stubbed for import (only pure helpers are used).

  python -m digbench.run_chrono bucket-scm --delta 0.05 --tag scm-d5cm
  python -m digbench.run_chrono plate-probe
  python -m digbench.run_chrono collapse-crm --cases 1,3 --tag crm
  python -m digbench.run_chrono bucket-crm --tag crm
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

for _m in ("taichi", "taichi.lang", "taichi.types"):
    sys.modules.setdefault(_m, mock.MagicMock())

from . import metrics, references  # noqa: E402
from .attitude import physical_path  # noqa: E402

ROOT = Path(__file__).resolve().parent


def git_rev():
    import os
    if os.environ.get("LIVEDIRT_GIT"):
        return os.environ["LIVEDIRT_GIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def device_name(gpu=False):
    if gpu:
        try:
            return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
        except Exception:
            pass
    import platform
    return f"{platform.machine()} cpu"


def save(out, name, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    p = Path(outdir) / name
    p.write_text(json.dumps(out))
    print("->", p)


def bucket_scm(args):
    from .solvers import chrono_baseline as cb
    from .run_bucket import SURF, bucket_boxes
    soil = {}
    if args.cohesion is not None:
        soil["Mohr_cohesion"] = args.cohesion
    if args.kphi_scale != 1.0:
        soil["Bekker_Kphi"] = cb.SCM_SOIL["Bekker_Kphi"] * args.kphi_scale
    path_fn, T = physical_path(args.depth, SURF)
    res = cb.run_scm_bucket(path_fn, T if args.duration is None else args.duration, delta=args.delta, dt=args.dt,
                            soil=soil, bulldoze=not args.no_bulldoze, boxes=bucket_boxes(0.04))
    ser = res["series"]  # t, px, pz, pitch, Fx, Fy, Fz, Ty, npatches
    t = ser[:, 0]
    drag = (t >= 1.2) & (t < 3.0)  # same window convention as run_bucket (after penetration settles, before curl)
    H0, H1 = res["H0"], res["H1"]
    cell = cb.CELL
    fp = (slice(int(0.6 / cell), int(1.4 / cell)), slice(0, int(0.4 / cell)))
    zone = np.zeros(H1.shape, bool)
    zone[int(0.6 / cell):int(2.9 / cell), :int(0.8 / cell)] = True
    vol_change = float(2 * (H1 - H0)[:, :].sum() * cell * cell)  # both halves, bank volume
    summary = {
        "drag_force_mean_kN": float(ser[drag, 4].mean() / 1e3),
        "drag_force_peak_kN": float(ser[:, 4].max() / 1e3),
        "vertical_force_peak_kN": float(np.abs(ser[:, 6]).max() / 1e3),
        "vertical_force_mean_drag_kN": float(ser[drag, 6].mean() / 1e3),
        "carried_bank_m3": None,  # not modeled: SCM is a heightfield surface model, it cannot carry soil in a bucket
        "bucket_geometric_m3": 0.9 * 0.9 * 1.0,
        "trench_depth_footprint_scm_m": float(cb.SURF - H1[fp].mean()),
        "terrain_min_height_m": float(H1.min()),
        "terrain_volume_change_bank_m3": vol_change,
        "wall_s": res["wall_s"], "steps": res["steps"], "rtf": res["steps"] * res["dt"] / res["wall_s"],
        "bulldozing": not args.no_bulldoze,
    }
    out = {"task": "bucket_dig", "solver": "chrono-scm", "tag": args.tag, "device": device_name(), "git": git_rev(),
           "chrono_version": "pychrono 10.0.0 (conda projectchrono)",
           "params": {**vars(args), "soil": {**cb.SCM_SOIL, **soil}, "bulldozing": cb.BULLDOZE, "bucket": bucket_boxes(0.04),
                      "n_collision_boxes_incl_mirror": res["n_shapes"], "surface": cb.SURF},
           "summary": summary, "series_cols": ["t", "pivot_x", "pivot_z", "pitch", "Fx_N", "Fy_N", "Fz_N", "Ty_Nm", "n_patches"],
           "series": ser[::2].round(4).tolist(), "H_scm": H1.round(4).tolist(), "H0_scm": H0.round(4).tolist()}
    print(json.dumps(summary, indent=1))
    save(out, f"chrono-bucket-{args.tag}.json", args.out)


def plate_probe(args):
    from .solvers import chrono_baseline as cb
    A = 0.0381 * 0.0254
    rows = []
    for b, g in [(0, 0), (0, 90), (90, 0), (30, 0), (-30, 0), (60, 90)]:
        t0 = time.perf_counter()
        s = cb.run_scm_plate(b, g, delta=args.delta)
        w = time.perf_counter() - t0
        rows.append({"beta": b, "gamma": g, "Fx_mean_N": float(s[:, 1].mean()), "Fz_mean_N": float(s[:, 2].mean()),
                     "Fx_max_abs_N": float(np.abs(s[:, 1]).max()), "Fz_max_abs_N": float(np.abs(s[:, 2]).max()),
                     "max_patches": int(s[:, 3].max()), "wall_s": w})
        print(rows[-1], flush=True)
    out = {"task": "plate_rft", "solver": "chrono-scm", "tag": "probe", "device": device_name(), "git": git_rev(),
           "status": "not scored: SCM cannot represent a submerged plate (probe only)",
           "params": {"delta": args.delta, "plate_m": [0.0381, 0.0254, 0.0064], "depth0_m": 0.05, "soil": cb.SCM_SOIL},
           "cases": rows}
    save(out, "chrono-plate-probe.json", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bucket-scm", "plate-probe", "collapse-crm", "bucket-crm"])
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--depth", type=float, default=0.6)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--cohesion", type=float, default=None)
    ap.add_argument("--kphi-scale", type=float, default=1.0)
    ap.add_argument("--no-bulldoze", action="store_true")
    ap.add_argument("--cases", default="1,3")
    ap.add_argument("--spacing", type=float, default=0.005)
    ap.add_argument("--tag", default="scm")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    if args.cmd == "bucket-scm":
        bucket_scm(args)
    elif args.cmd == "plate-probe":
        plate_probe(args)
    else:
        from .solvers import chrono_crm
        chrono_crm.run(args, save, device_name, git_rev)


if __name__ == "__main__":
    main()
