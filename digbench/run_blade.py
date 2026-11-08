"""DigBench teacher check: Coetzee 2004 flat-blade benchmark (thesis Sec 6.2.5, Fig 6.14).

A flat vertical blade (350 mm high, ~200 mm wide = glass spacing) is pushed horizontally at 10 mm/s through corn at a
FIXED depth h (no vertical motion; arm locked), h = 150 / 250 / 350 mm. Measured average draft force over 15-35 mm of
blade displacement: 96.6 / 205.8 / 386.1 N. Unlike the bucket test, the path is fully specified, so this isolates the
solver (material model, tool contact, side-wall friction, grid).

Two references:
  * code verification: plane strain (no side-wall friction) vs the Sokolovski passive-pressure solution the thesis
    reports for the same material (63.0 / 171.4 / 334.3 N);
  * validation: glass side walls (12 deg) vs the experiment.
Published comparisons (labelled published): nonpolar Cosserat-MPM (plane strain) -25 / -11 / -9 %, DEM +9 / +11 / -5 %.

ASSUMPTIONS (not in the thesis): blade thickness 10 mm (raised to 2.5 dx on coarse grids), bed bottom 0.15 m below the
blade tip, soil on both sides of the blade, drag speed 0.1 m/s (no rate effect 0-100 mm/s per the thesis, p95),
initial stress from damped gravity settling.

  python -m digbench.run_blade --arch cuda --dx 0.004 --yield-model mc --glass-deg 12
"""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np

from .run_collapse import device_name, git_rev
from .solvers.sim3d import Sim3D

ROOT = Path(__file__).resolve().parent


def load_ref():
    s = importlib.util.spec_from_file_location("coetzee", ROOT / "refs" / "coetzee.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def run_depth(args, h, mat):
    W2 = 0.100  # half width (symmetry plane at y = 0, glass wall at y = 0.1)
    thick = max(args.thick, 2.5 * args.dx)
    below = args.below
    Lz = h + below + 0.10
    xb = 0.15  # blade front face start
    Lx = xb + args.ahead
    bed = h + below
    E, nu, rho = mat["E_MPa"] * 1e6, 0.2, mat["density_kg_m3"]
    sim = Sim3D((Lx, W2, Lz), args.dx, mu_f=math.tan(math.radians(mat["friction_angle_deg"])), rho_bulk=rho, E=E, nu=nu,
                mu_tool=math.tan(math.radians(mat["friction_with_steel_deg"])), arch=args.arch, symmetry_y=True,
                wall_mu=math.tan(math.radians(args.glass_deg)), tool_speed_max=max(0.2, 1.5 * args.speed),
                volume_correction=args.volume_correction, yield_model=args.yield_model, phi_deg=mat["friction_angle_deg"],
                max_particles=int(1.05 * Lx * W2 * bed / (args.dx / 2) ** 3) + 1000)
    n0 = sim.fill_box((0, 0, 0), (Lx, W2, bed))
    # blade in tool frame: front face at x = 0, tip at z = 0, height 0.35 m upward, plate behind the face
    sim.set_tool([((-thick / 2, 0.0, 0.175), (thick / 2, W2, 0.175), 0.0)])
    z_tip = bed - h
    T = args.drag / args.speed

    def path(t):
        u = min(max(t, 0.0), T) * args.speed
        return np.array([xb + u, 0.0, z_tip]), 0.0

    sim.set_pose(*path(0.0))
    sim.remove_inside_tool()
    sim.settle(args.settle)
    t0 = time.perf_counter()
    res = sim.drive(path, T, sample_every=T / 200)
    wall = time.perf_counter() - t0
    ser = res["series"]
    disp = ser[:, 1] * args.speed * 1000.0
    draft = -ser[:, 6]
    win = (disp >= 15) & (disp <= 35)
    return {"h_mm": h * 1000, "avg_draft_15_35mm_N": float(draft[win].mean()), "vertical_15_35mm_N": float(ser[win, 7].mean()),
            "particles": n0, "wall_s": wall, "steps": res["steps"], "dt": res["dt"], "thick_m": thick,
            "curve_mm_N": np.stack([disp, draft], 1)[::2].round(2).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--dx", type=float, default=0.004)
    ap.add_argument("--depths", default="0.15,0.25,0.35")
    ap.add_argument("--yield-model", choices=["dp", "mc"], default="mc")
    ap.add_argument("--glass-deg", type=float, default=12.0)
    ap.add_argument("--thick", type=float, default=0.010, help="blade thickness, ASSUMPTION")
    ap.add_argument("--below", type=float, default=0.15, help="bed below the blade tip, ASSUMPTION")
    ap.add_argument("--ahead", type=float, default=0.90)
    ap.add_argument("--speed", type=float, default=0.10)
    ap.add_argument("--drag", type=float, default=0.040)
    ap.add_argument("--settle", type=float, default=0.2)
    ap.add_argument("--volume-correction", action="store_true")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    ref = load_ref()
    mat = ref.THESIS_MATERIAL["measured"]
    meas = ref.BLADE_MEASURED
    rows = []
    for h in [float(v) for v in args.depths.split(",")]:
        r = run_depth(args, h, mat)
        i = meas["depth_h_mm"].index(int(round(h * 1000)))
        r["exp_N"] = meas["experiment_avg_draft_force_N"][i]
        r["sokolovski_N"] = meas["Sokolovski_N_at_depths"][i]
        r["rel_err_exp"] = r["avg_draft_15_35mm_N"] / r["exp_N"] - 1
        r["rel_err_sokolovski"] = r["avg_draft_15_35mm_N"] / r["sokolovski_N"] - 1
        rows.append(r)
        print(f"h={h:.2f} sim={r['avg_draft_15_35mm_N']:.1f} N exp={r['exp_N']} ({r['rel_err_exp']:+.0%}) "
              f"sokolovski={r['sokolovski_N']} ({r['rel_err_sokolovski']:+.0%}) wall={r['wall_s']:.0f}s particles={r['particles']}", flush=True)
    out = {"task": "coetzee_blade", "tag": args.tag, "device": device_name(args.arch), "git": git_rev(), "params": vars(args),
           "reference": {"source": meas["source"], "published_nonpolar_mpm": "-25/-11/-9 % [PUBLISHED]", "published_dem": "+9/+11/-5 % [PUBLISHED]"},
           "rows": rows,
           "summary": {"mean_rel_err_exp": float(np.mean([r["rel_err_exp"] for r in rows])),
                       "mean_rel_err_sokolovski": float(np.mean([r["rel_err_sokolovski"] for r in rows]))}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pth = Path(args.out) / f"blade-{args.tag}.json"
    pth.write_text(json.dumps(out))
    print(json.dumps(out["summary"]), "->", pth)


if __name__ == "__main__":
    main()
