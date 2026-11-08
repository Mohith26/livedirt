"""DigBench teacher check: cohesion verification on a vertical cut (plane strain).

A soil block of height H with a vertical free face is loaded by gravity. Classical limit analysis brackets the critical
height of an unsupported vertical cut in a Mohr-Coulomb soil (tension allowed up to the MC apex):
    H_c = N c / gamma * tan(45 deg + phi/2),  N = 3.83 (upper bound, Chen 1975 log-spiral) ... 4.0 (Culmann plane wedge)
and the lower bound with a full-depth tension crack is N = 2.0 (Terzaghi). A correct cohesive MC implementation should
stand at c above c_crit and collapse below it, with the threshold N in [2, 4] and, for a tension-capable model, near 3.83.

We sweep c around the theory value and record the crest settlement after undamped time; the threshold N is where the
crest displacement jumps.

  python -m digbench.run_cut --arch cuda --phi 20 --H 1.0 --dx 0.025
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .run_collapse import device_name, git_rev
from .solvers.sim3d import Sim3D

ROOT = Path(__file__).resolve().parent


def run_one(args, c):
    H, dx = args.H, args.dx
    Lx, Ly, Lz = 3.0 * H, 4 * dx, 1.3 * H
    sim = Sim3D((Lx, Ly, Lz), dx, mu_f=math.tan(math.radians(args.phi)), rho_bulk=args.rho, E=args.E, nu=0.3,
                arch=args.arch, symmetry_y=True, wall_mu=0.0, yield_model="mc", phi_deg=args.phi, cohesion=c,
                max_particles=int(1.1 * 1.5 * H * Ly * H / (dx / 2) ** 3) + 1000)
    n0 = sim.fill_box((0, 0, 0), (1.5 * H, Ly, H))
    # gravity with heavy damping first (geostatic state), then undamped: a cut above its critical height keeps moving
    sim.settle(args.settle, use_tool=0)
    p0 = sim.particles()
    band = lambda q: q[(q[:, 0] > 1.25 * H) & (q[:, 0] < 1.5 * H)][:, 2]
    z_crest0 = float(np.percentile(band(p0), 99))
    x_face0 = float(np.percentile(p0[:, 0], 99.5))
    t0 = time.perf_counter()
    n = int(args.run / sim.dt)
    for _ in range(n):
        sim._substep(0)
    p1 = sim.particles()
    wall = time.perf_counter() - t0
    # crest settlement: top of the soil within 2 dx of the original face, vs the same columns after
    b1 = band(p1)
    zc = float(np.percentile(b1, 99)) if len(b1) else 0.0
    runout = float(np.percentile(p1[:, 0], 99.5)) - x_face0
    return {"c": c, "crest_drop_m": z_crest0 - zc, "runout_m": runout, "wall_s": wall, "particles": n0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--phi", type=float, default=20.0)
    ap.add_argument("--H", type=float, default=1.0)
    ap.add_argument("--dx", type=float, default=0.025)
    ap.add_argument("--rho", type=float, default=1600.0)
    ap.add_argument("--E", type=float, default=5e6)
    ap.add_argument("--settle", type=float, default=0.3)
    ap.add_argument("--run", type=float, default=2.0)
    ap.add_argument("--N", type=str, default="1.5,2.0,2.5,3.0,3.5,3.83,4.2,5.0,6.0", help="stability numbers to test")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    gam = args.rho * 9.81
    Kp = math.tan(math.radians(45 + args.phi / 2))
    rows = []
    for N in [float(v) for v in args.N.split(",")]:
        # cut stands iff H <= N_true c / gamma * Kp, i.e. testing c = gamma H / (N Kp) probes "is N_true >= N"
        c = gam * args.H / (N * Kp)
        r = run_one(args, c)
        r["N_tested"] = N
        rows.append(r)
        print(f"N={N:.2f} c={c:.0f} Pa crest_drop={r['crest_drop_m']:.3f} m runout={r['runout_m']:.3f} m wall={r['wall_s']:.0f}s", flush=True)
    # cut stands iff N_tested <= N_true (larger c for smaller N): N_true lies in [largest N that stands, smallest N that fails]
    stands = [r["N_tested"] for r in rows if r["crest_drop_m"] < 0.05 * args.H]
    fails = [r["N_tested"] for r in rows if r["crest_drop_m"] >= 0.05 * args.H]
    out = {"task": "cut_stability", "tag": args.tag, "device": device_name(args.arch), "git": git_rev(), "params": vars(args),
           "theory": {"N_upper_bound_chen": 3.83, "N_culmann": 4.0, "N_tension_crack": 2.0, "Kp": Kp,
                      "c_crit_chen_Pa": gam * args.H / (3.83 * Kp)},
           "rows": rows,
           "summary": {"N_stand_max": max(stands, default=None), "N_fail_min": min(fails, default=None),
                       "consistent": (max(stands, default=0) < min(fails, default=99))}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pth = Path(args.out) / f"cut-{args.tag}.json"
    pth.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"]), "->", pth)


if __name__ == "__main__":
    main()
