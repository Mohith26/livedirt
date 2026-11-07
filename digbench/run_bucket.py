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
    t = max(T, 1.25 * dx)
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


def kinematic_heightfield_baseline(cell, nxy, depth):
    """Swept-volume cut: every column the bucket floor passes below the surface is cut down to the floor;
    removed bank volume is 'carried' up to the bucket's geometric capacity; overflow is lost (not modelled).
    This is the standard real-time approach (and how LiveDirt's dig() treats the bucket)."""
    H = np.full(nxy, SURF)
    removed = 0.0
    for t in np.linspace(0.0, 4.5, 2000):
        pv, a = path(t, depth)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--depth", type=float, default=0.6)
    ap.add_argument("--fmax", type=float, default=None, help="drive force limit (N); None = prescribed motion")
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
    p0, a0 = path(0.0, args.depth)
    sim.set_pose(p0, a0)
    sim.settle(0.3)
    cell = 0.1
    nxy = (int(BED[0] / cell), int(BED[1] / cell))
    H0 = sim.heightfield(cell, nxy, mode="volume")
    res = sim.drive(lambda t: path(t, args.depth), args.duration, f_max=args.fmax, sample_every=0.02)
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
    lip_depth_end_pen = SURF - lip_world(ser[np.argmin(np.abs(ser[:, 1] - 1.0)), 2:5], 0.0)[1]
    Hk, removed_k = kinematic_heightfield_baseline(cell, nxy, args.depth)
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
            "trench_depth_centre_mpm_m": float(SURF - Hg[int(1.8 / cell), 0]),
            "trench_depth_centre_kinematic_m": float(SURF - Hk[int(1.8 / cell), 0]),
            "trench_depth_centre_livedirt_relax_m": float(SURF - Hr[int(1.8 / cell), 0]),
            "volume_total_bank_m3": float(vol_total),
            "volume_on_bucket_outside_m3": float(vol_on_bucket_outside),
            "wall_s": res["wall_s"], "steps": res["steps"], "rtf": res["steps"] * res["dt"] / res["wall_s"],
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
