"""DigBench task 3: full-scale bucket dig cycle (penetrate, drag, curl, lift) in sand.

No public measured data at this scale is available to us, so this task is scored against (a) the same
contact model that matched Li et al. 2013 plate stresses (task 2), (b) a finer-grid reference run (grid
convergence), and (c) conservation. The kinematic swept-volume heightfield (what real-time terrain uses,
including LiveDirt's dig) is scored against the MPM reference on final terrain and carried volume.

  python -m digbench.run_bucket --arch cuda --dx 0.04 --tag dx4cm
  python -m digbench.run_bucket --arch cuda --dx 0.04 --fmax 60000 --tag dx4cm-fmax60k
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .run_collapse import device_name, git_rev
from .solvers.heightfield import Heightfield
from .solvers.sim3d import Sim3D, connected_surface

ROOT = Path(__file__).resolve().parent

# bucket in tool frame: pivot at origin, opening toward -x and up; full width 1.0 m (y half 0.5), plates 4 cm
T = 0.02


def bucket_boxes(dx):
    """Shell plates built outward from a fixed 0.9 x 0.9 x 1.0 m interior (capacity unchanged), each at least
    2.5 grid cells thick: thinner shells let soil tunnel through (nodes inside a thin plate take the normal of
    the far face)."""
    # fixed 5 cm half-thickness (10 cm shell) at every resolution so grid-convergence runs share one geometry;
    # >= 2.5 cells thick for dx <= 4 cm (thinner shells let soil tunnel through)
    assert dx <= 0.0801, "bucket shell needs dx <= 4 cm (8 cm allowed for smoke tests)"
    t = 0.05
    return [
        ((t, 0.0, -0.45), (t, 0.5 + 2 * t, 0.45 + 2 * t), 0.0),            # back plate
        ((-0.45, 0.0, -0.9 - t), (0.45, 0.5 + 2 * t, t), 0.0),             # floor
        ((-0.45, 0.5 + t, -0.45), (0.45, t, 0.45), 0.0),                   # side plate (y = +0.5)
        ((-0.45, 0.0, t), (0.45, 0.5 + 2 * t, t), 0.0),                    # top shell: single mouth facing -x
    ]


BUCKET = [
    ((0.0, 0.0, -0.45), (T, 0.5, 0.45), 0.0),      # back plate
    ((-0.45, 0.0, -0.9), (0.45, 0.5, T), 0.0),     # floor
    ((-0.45, 0.5, -0.45), (0.45, T, 0.45), 0.0),   # side plate (y = +0.5; the y = -0.5 side is the mirror)
]
LIP = np.array([-0.9, -0.9])  # cutting edge in tool frame (x, z)
BUCKET_GEOM_VOL = 0.9 * 0.9 * 1.0
SURF = 1.2   # soil depth (m)
BED = (3.4, 1.0, SURF + 2.5)  # x, y (half, symmetry at y=0), z; ceiling clears the lifted bucket


def lip_world(pivot, pitch):
    ca, sa = math.cos(pitch), math.sin(pitch)
    return np.array([pivot[0] + ca * LIP[0] - sa * LIP[1], pivot[2] + sa * LIP[0] + ca * LIP[1]])


def path(t, depth=0.6, x_shift=0.0):
    """Commanded pivot pose vs time (s). Lip starts 0.3 m above the surface."""
    x0, z0 = 2.7 + x_shift, SURF + 1.2
    dz_pen = 0.3 + depth  # lip starts 0.3 m above the surface; lower it to `depth` below
    if t < 1.0:  # penetrate (down and slightly back)
        u = t / 1.0
        return np.array([x0 - 0.2 * u, 0.0, z0 - dz_pen * u]), 0.0
    x1, z1 = x0 - 0.2, z0 - dz_pen
    if t < 3.0:  # drag toward the machine
        u = (t - 1.0) / 2.0
        return np.array([x1 - 1.2 * u, 0.0, z1]), 0.0
    x2 = x1 - 1.2
    if t < 4.5:  # curl (lip up) while rising a little
        u = (t - 3.0) / 1.5
        return np.array([x2 - 0.1 * u, 0.0, z1 + 0.3 * u]), -math.radians(75) * u
    x3, z3 = x2 - 0.1, z1 + 0.3
    u = min(1.0, (t - 4.5) / 1.0)  # lift
    return np.array([x3, 0.0, z3 + 0.7 * u]), -math.radians(75)


def carried_volume(p, sp, pivot, pitch):
    """Bank volume of particles inside the bucket envelope (tool frame box) after the lift."""
    r = p - np.array([pivot[0], 0.0, pivot[2]])
    ca, sa = math.cos(pitch), math.sin(pitch)
    lx = ca * r[:, 0] + sa * r[:, 2]
    lz = -sa * r[:, 0] + ca * r[:, 2]
    inside = (lx > -0.9) & (lx < 0.0) & (lz > -0.9) & (lz < 0.0) & (r[:, 1] < 0.5)
    return 2 * inside.sum() * sp ** 3  # x2 for symmetry


# DX225LC-7 linkage (app/src/machine/kinematics.ts) placed so the bucket digs toward the machine: swing axis at
# x = -4 m on the bed surface; ISO digging forces from the Develon spec sheet.
DX225 = {"boom_foot": (0.18, 2.05), "boom": 5.7, "arm": 2.9, "x_machine": -4.0,
         "bucket_force": 161.8e3, "bucket_r": 1.45, "arm_force": 112.8e3}


def make_dx225_limiter(log, enforce=True, pressure=None):
    """Per-actuator DX225LC-7 limits (digbench/actuators.py): boom/arm/bucket cylinder torque from relief pressure and
    ISO-calibrated moment arms, plus front lift-off, forward tip and track sliding. Swing axis at x = -4 m on the bed
    surface; the rig pivot stands in for the bucket (wrist) pin. With enforce=False the loads are logged only."""
    from . import actuators as A
    xm = DX225["x_machine"]

    def limiter(fx, fz, ty, pb, ab):
        piv = (pb[0] - xm, pb[2] - SURF)
        lip = lip_world(pb, ab)
        d = A.demand(fx, fz, ty, piv, bucket_xy=(lip[0] - xm, lip[1] - SURF), pressure=pressure or A.P_RELIEF)
        log.append(d)
        return enforce and d["max_frac"] > 1.0

    return limiter


def kinematic_heightfield_baseline(cell, nxy, depth, path_fn=None):
    """Swept-volume cut: every column the bucket floor passes below the surface is cut down to the floor;
    removed bank volume is 'carried' up to the bucket's geometric capacity; overflow is lost (not modelled).
    This is the standard real-time approach (and how LiveDirt's dig() treats the bucket)."""
    H = np.full(nxy, SURF)
    removed = 0.0
    for t in np.linspace(0.0, 4.5, 2000):
        pv, a = path_fn(t) if path_fn else path(t, depth)
        ca, sa = math.cos(a), math.sin(a)
        # floor plate segment in world x-z
        for s in np.linspace(0.0, 1.0, 20):
            lx = -0.9 + 0.9 * s
            lz = -0.9
            wx, wz = pv[0] + ca * lx - sa * lz, pv[2] + sa * lx + ca * lz
            i = int(wx / cell)
            if 0 <= i < nxy[0]:
                for j in range(min(nxy[1], int(0.5 / cell) + 1)):
                    if H[i, j] > wz:
                        removed += (H[i, j] - max(wz, 0.0)) * cell * cell
                        H[i, j] = max(wz, 0.0)
    return H, 2 * removed


def stall_phase(ser):
    """Where the commanded trajectory stopped progressing (phase timing shared by the legacy and physical paths)."""
    s = ser[:, 9]
    tc = ser[:, 1]
    stalled = np.where(s < 0.05)[0]
    t = tc[stalled[0]] if len(stalled) else None
    if t is None:
        return None
    return "penetrate" if t < 1.0 else "drag" if t < 3.0 else "curl" if t < 4.5 else "lift"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--depth", type=float, default=0.6)
    ap.add_argument("--fmax", type=float, default=None, help="drive force limit (N); None = prescribed motion. "
                    "DX225LC-7 spec (ISO): arm digging force 11,500 kgf = 112.8 kN, bucket 16,500 kgf = 161.8 kN")
    ap.add_argument("--limits", choices=["none", "dx225"], default="none",
                    help="dx225: per-actuator cylinder limits + stability (digbench/actuators.py); none: loads logged only")
    ap.add_argument("--boost", action="store_true", help="power-boost relief pressure (36.3 MPa) instead of 34.3 MPa")
    ap.add_argument("--attitude", choices=["physical", "legacy"], default="physical",
                    help="physical: teeth-first penetration and easing attack angle (digbench/attitude.py); legacy: flat floor at pitch 0")
    ap.add_argument("--mu-f", type=float, default=0.38)
    ap.add_argument("--E", type=float, default=1.0e6)
    ap.add_argument("--rho", type=float, default=1600.0)
    ap.add_argument("--duration", type=float, default=5.5)
    ap.add_argument("--tag", default="bucket")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    sim = Sim3D(BED, args.dx, mu_f=args.mu_f, rho_bulk=args.rho, E=args.E, arch=args.arch, tool_speed_max=1.5,
                max_particles=int(1.15 * BED[0] * BED[1] * SURF / (args.dx / 2) ** 3) + 1000)
    n0 = sim.fill_box((0, 0, 0), (BED[0], BED[1], SURF))
    sim.set_tool(bucket_boxes(args.dx))
    from .attitude import physical_path
    if args.attitude == "physical":
        path_fn, _T = physical_path(args.depth, SURF)
    else:
        path_fn = lambda t: path(t, args.depth)
    p0, a0 = path_fn(0.0)
    sim.set_pose(p0, a0)
    sim.settle(0.3)
    cell = 0.1
    nxy = (int(BED[0] / cell), int(BED[1] / cell))
    H0 = sim.heightfield(cell, nxy, mode="volume")
    log = []
    from . import actuators as A
    limiter = make_dx225_limiter(log, enforce=args.limits == "dx225", pressure=A.P_BOOST if args.boost else A.P_RELIEF) if args.fmax is None else None
    res = sim.drive(path_fn, args.duration, f_max=args.fmax, limiter=limiter, sample_every=0.02)
    ser = res["series"]  # t, t_cmd, px, py, pz, pitch, Fx, Fz, Ty, s
    p = sim.particles()
    pv, pa = ser[-1, 2:5], ser[-1, 5]
    carried = carried_volume(p, sim.sp, pv, pa)
    H1_top = sim.heightfield(cell, nxy, mode="top")
    # terrain readout excludes everything in or on the bucket (padded envelope around the shell)
    r = p - np.array([pv[0], 0.0, pv[2]])
    ca, sa = math.cos(pa), math.sin(pa)
    lx = ca * r[:, 0] + sa * r[:, 2]; lz = -sa * r[:, 0] + ca * r[:, 2]
    on_bucket = (lx > -1.25) & (lx < 0.4) & (lz > -1.25) & (lz < 0.4) & (r[:, 1] < 0.85)
    ground = ~on_bucket
    Hg = connected_surface(p[ground], nxy, cell, sim.sp)
    vol_total = 2 * len(p) * sim.sp ** 3
    vol_on_bucket_outside = 2 * on_bucket.sum() * sim.sp ** 3 - carried
    drag = (ser[:, 0] > 1.2) & (ser[:, 1] < 3.0)
    i_pen = np.argmin(np.abs(ser[:, 1] - 1.0))
    lip_depth_end_pen = SURF - lip_world(ser[i_pen, 2:5], ser[i_pen, 5])[1]
    Hk, removed_k = kinematic_heightfield_baseline(cell, nxy, args.depth, path_fn)
    # LiveDirt-style: kinematic cut, then LiveDirt's relax (repose = bank = friction angle for dry sand),
    # mirrored across the symmetry plane so the y = 0 row relaxes like an interior row
    ang = math.degrees(math.atan(args.mu_f))
    full = np.concatenate([Hk[:, ::-1], Hk], axis=1)
    hf = Heightfield(np.concatenate([np.full_like(Hk, SURF)] * 2, axis=1), cell, ang, ang)
    hf.B[:] = full
    hf.settle()
    Hr = hf.h()[:, nxy[1]:]
    cut = Hk < SURF - 1e-6
    zone = np.zeros(nxy, bool)
    zone[max(0, int(0.6 / cell)):int(2.9 / cell), :int(0.8 / cell)] = True  # trench region incl. walls
    out = {
        "task": "bucket_dig", "solver": "mls-mpm-dp-3d", "tag": args.tag, "device": device_name(args.arch), "git": git_rev(),
        "params": {**vars(args), "bucket": bucket_boxes(args.dx), "bed": BED, "surface": SURF, "dt": sim.dt, "particles": n0},
        "summary": {
            "drag_force_mean_kN": float(ser[drag, 6].mean() / 1e3) if drag.any() else None,  # +x resists the -x drag
            "drag_force_peak_kN": float(ser[:, 6].max() / 1e3),
            "vertical_force_peak_kN": float(np.abs(ser[:, 7]).max() / 1e3),
            "carried_bank_m3": float(carried),
            "bucket_geometric_m3": BUCKET_GEOM_VOL,
            "kinematic_removed_bank_m3": float(removed_k),
            "lip_depth_after_penetration_m": float(lip_depth_end_pen),
            "trajectory_completion": float(ser[-1, 1] / args.duration),
            "min_speed_scale": float(ser[:, 9].min()),
            "terrain_rmse_kinematic_vs_mpm_m": float(np.sqrt(((Hk - Hg)[zone] ** 2).mean())),
            "terrain_rmse_livedirt_relax_vs_mpm_m": float(np.sqrt(((Hr - Hg)[zone] ** 2).mean())),
            # legacy single-cell readout (x = 1.8 m, y = 0): kept for comparison; it sits where the legacy flat
            # plunge compressed and heaved the soil, so it is not a trench-depth measure
            "trench_depth_cell_x1.8_mpm_m": float(SURF - Hg[int(1.8 / cell), 0]),
            # footprint mean over the drag path (lip x 0.6-1.4 m, |y| < 0.4 m)
            "trench_depth_footprint_mpm_m": float(SURF - Hg[int(0.6 / cell):int(1.4 / cell), :int(0.4 / cell)].mean()),
            "trench_depth_footprint_kinematic_m": float(SURF - Hk[int(0.6 / cell):int(1.4 / cell), :int(0.4 / cell)].mean()),
            "trench_depth_footprint_livedirt_relax_m": float(SURF - Hr[int(0.6 / cell):int(1.4 / cell), :int(0.4 / cell)].mean()),
            "ground_removed_bank_m3": float(2 * (H0[zone].sum() - np.minimum(Hg, H0)[zone].sum() + 0.0) * cell * cell),
            "volume_total_bank_m3": float(vol_total),
            "volume_on_bucket_outside_m3": float(vol_on_bucket_outside),
            "wall_s": res["wall_s"], "steps": res["steps"], "rtf": res["steps"] * res["dt"] / res["wall_s"],
            "stall_phase": stall_phase(ser),
            "attitude": args.attitude, "limits": args.limits if args.fmax is None else f"fmax {args.fmax}",
            **({f"peak_{k}_frac": float(max(d[k] for d in log)) for k in ("bucket", "arm", "boom", "front_liftoff", "forward_tip", "sliding")} if log else {}),
            "peak_governing": (max(log, key=lambda d: d["max_frac"])["governing"] if log else None),
            "would_tip_or_lift": bool(log and max(max(d["front_liftoff"], d["forward_tip"]) for d in log) > 1.0),
        },
        "series": ser[::2].round(5).tolist(),
        "H_mpm": Hg.round(4).tolist(), "H_kinematic": Hk.round(4).tolist(), "H_livedirt_relax": Hr.round(4).tolist(), "H0_volume": H0.round(4).tolist(),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pth = Path(args.out) / f"bucket-{args.tag}.json"
    pth.write_text(json.dumps(out))
    print(json.dumps(out["summary"], indent=1), "->", pth)


if __name__ == "__main__":
    main()
