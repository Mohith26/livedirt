"""DigBench task 3b: measured lab bucket filling (Coetzee 2004 PhD thesis Sec 6.2, Fig 6.25; Coetzee, Basson & Vermeer
2007, J. Terramechanics 44:177-186). A dragline bucket profile is dragged 800 mm through cohesionless corn between two
glass panels 200 mm apart; the draft force vs displacement was measured (single run, +-5-10 N trace noise).

Reproduced in the 3D MLS-MPM solver (solvers/sim3d.py) with ONLY the published inputs - no fitting:
  * bed exactly 200 mm wide, both side walls with the measured corn-glass friction (12 deg);
  * corn: E 2.76 MPa, nu 0.2 (assumed in the thesis), bulk density 778 kg/m^3, friction 26 deg (shear box), cohesion 0;
    bucket (steel) friction 14 deg (all Table 6.1, thesis p95);
  * bucket: Fig 6.3 profile (tooth 40 mm at 40 deg, 190 mm floor, R100 arc, 140 mm wall), floor horizontal, tooth
    engaged at the start (floor at the free surface);
  * path: horizontal drag 800 mm with the thesis' constant vertical sink, -0.20 mm per mm of drag (measured from the
    thesis simulation figures, +-0.04; refs/coetzee.py VERTICAL_PATH).
ASSUMPTIONS (not in the thesis; reported with sensitivity runs): bed depth (default 0.30 m), plate thickness 15 mm
(an unlabelled 15 mm dimension on the drawing), drag speed 100 mm/s instead of 10 mm/s (the thesis reports no
significant effect of drag speed on draft force over 0-100 mm/s, p95; quasi-static), dilatancy (psi = 2 deg in the
thesis was a fit to the free surface; omitted here).

Published comparisons (labelled published): nonpolar MPM 12% low at 800 mm (text; ~16% from the plotted curve),
DEM ~20 N low up to 600 mm, polar MPM lower than both.

  python -m digbench.run_coetzee --arch cuda --dx 0.006
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


def profile_boxes(poly_mm, width, thick, n_arc=6):
    """Bucket profile polyline (mm, tooth tip at origin, +x = drag direction, floor horizontal) -> list of rotated boxes
    (centre, half-extents, pitch) in the tool frame (m). Plates sit on the outer side of the inner-surface polyline."""
    P = np.asarray(poly_mm, float) / 1000.0
    # thin the 16-point arc to n_arc segments to keep the box count small (sim3d n_boxes = 8 by default; raised below)
    keep = [0, 1, 2] + list(np.linspace(2, len(P) - 2, n_arc + 1).round().astype(int)[1:]) + [len(P) - 1]
    keep = sorted(set(keep))
    P = P[keep]
    boxes = []
    for a, b in zip(P[:-1], P[1:]):
        d = b - a
        L = float(np.hypot(*d))
        ang = math.atan2(d[1], d[0])
        n = np.array([-d[1], d[0]]) / L  # left-hand normal of a->b: the polyline runs tip -> back, so this points outward (below the floor, behind the wall)
        c = (a + b) / 2 + n * thick / 2
        boxes.append(((float(c[0]), width / 2, float(c[1])), (L / 2 + thick / 2, width / 2, thick / 2), ang))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--dx", type=float, default=0.006)
    ap.add_argument("--bed-depth", type=float, default=0.30, help="ASSUMPTION (not in the thesis)")
    ap.add_argument("--thick", type=float, default=0.015, help="plate thickness, ASSUMPTION")
    ap.add_argument("--sink", type=float, default=-0.20, help="vertical mm per mm of drag (thesis figs, +-0.04)")
    ap.add_argument("--speed", type=float, default=0.10, help="drag speed m/s (experiment 0.01; no effect 0-0.1 per thesis)")
    ap.add_argument("--glass-deg", type=float, default=12.0)
    ap.add_argument("--tool-mu", type=float, default=None, help="override bucket-corn friction (default tan 14 deg = 0.249)")
    ap.add_argument("--drag", type=float, default=0.800, help="drag length (m); shorter for grid studies")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    ref = load_ref()
    mat = ref.THESIS_MATERIAL["measured"]
    poly = ref.BUCKET_PROFILE_MM["polyline_inner_surface_mm"]
    W = 0.200
    drag = args.drag
    x_tip0 = 0.40  # tooth tip start (bucket body extends 0.33 m behind it, toward -x)
    Lx = x_tip0 + drag + 0.35
    Lz = args.bed_depth + 0.40
    E, nu, rho = mat["E_MPa"] * 1e6, 0.2, mat["density_kg_m3"]
    sim = Sim3D((Lx, W, Lz), args.dx, mu_f=math.tan(math.radians(mat["friction_angle_deg"])), rho_bulk=rho, E=E, nu=nu,
                mu_tool=args.tool_mu if args.tool_mu is not None else math.tan(math.radians(mat["friction_with_steel_deg"])), arch=args.arch, symmetry_y=False,
                wall_mu=math.tan(math.radians(args.glass_deg)), tool_speed_max=max(0.2, args.speed * 1.5), n_boxes=16,
                max_particles=int(1.05 * Lx * W * args.bed_depth / (args.dx / 2) ** 3) + 1000)
    n0 = sim.fill_box((0, 0, 0), (Lx, W, args.bed_depth))
    boxes = profile_boxes(poly, W, args.thick)
    sim.set_tool(boxes)
    floor_z = poly[1][1] / 1000.0  # floor inner surface height above the tooth tip
    z_tip0 = args.bed_depth - floor_z  # floor at the free surface, tooth engaged
    T = drag / args.speed

    def path(t):
        u = min(max(t, 0.0), T) * args.speed
        return np.array([x_tip0 + u, 0.0, z_tip0 + args.sink * u]), 0.0

    sim.set_pose(*path(0.0))
    sim.settle(0.2)
    t0 = time.perf_counter()
    res = sim.drive(path, T, sample_every=T / 400)
    wall = time.perf_counter() - t0
    ser = res["series"]  # t, t_cmd, px, py, pz, pitch, Fx, Fz, Ty, s
    disp_mm = ser[:, 1] * args.speed * 1000.0
    draft = -ser[:, 6]  # soil force on the bucket opposing the +x drag
    meas = ref.THESIS_BUCKET_MEASURED
    xb = np.array(meas["x_displacement_mm_bin_centres_25mm"])
    fb = np.array(meas["experiment_draft_force_N_bin_mean"])
    keep = xb <= drag * 1000.0
    xb, fb = xb[keep], fb[keep]
    sim_b = np.array([draft[(disp_mm >= x - 12.5) & (disp_mm < x + 12.5)].mean() for x in xb])
    ok = xb >= 25  # first bin is the engagement transient
    err = sim_b - fb
    # fill: particles inside the bucket polygon envelope at the end (inner floor/wall box, width W)
    p = sim.particles()
    pv = ser[-1, 2:5]
    rel = p - np.array([pv[0], 0.0, pv[2]])
    inside = (rel[:, 0] < 0) & (rel[:, 0] > -0.33) & (rel[:, 2] > floor_z) & (rel[:, 2] < floor_z + 0.24)
    fill_m3 = float(inside.sum() * sim.sp ** 3)
    out = {
        "task": "coetzee_bucket", "solver": "mls-mpm-dp-3d", "tag": args.tag, "device": device_name(args.arch), "git": git_rev(),
        "params": {**vars(args), "particles": n0, "dt": sim.dt, "boxes": boxes, "E": E, "nu": nu, "rho": rho},
        "reference": {"citation": ref.CITATION, "source": meas["source"], "single_run_noise_N": "+-5-10"},
        "summary": {
            "draft_at_800mm_sim_N": float(sim_b[-1]), "draft_at_800mm_exp_N": float(fb[-1]),
            "rel_err_at_800mm": float(err[-1] / fb[-1]), "end_mm": float(xb[-1]),
            "mae_N": float(np.abs(err[ok]).mean()), "rmse_N": float(np.sqrt((err[ok] ** 2).mean())),
            "mean_rel_err": float((err[ok] / fb[ok]).mean()), "mean_abs_rel_err": float(np.abs(err[ok] / fb[ok]).mean()),
            "fill_m3_end": fill_m3, "wall_s": wall, "steps": res["steps"],
            "published_nonpolar_mpm_rel_err_at_800mm": "-12% (text) / ~-16% (plotted curve) [PUBLISHED]",
            "published_dem": "~20 N low up to 600 mm; 191.8 N at 800 mm vs 250 N [PUBLISHED, digitized]",
        },
        "curve": {"x_mm": xb.tolist(), "sim_N": sim_b.round(2).tolist(), "exp_N": fb.tolist()},
        "series": ser[::4].round(5).tolist(),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pth = Path(args.out) / f"coetzee-{args.tag}.json"
    pth.write_text(json.dumps(out))
    print(json.dumps(out["summary"], indent=1), "->", pth)


if __name__ == "__main__":
    main()
