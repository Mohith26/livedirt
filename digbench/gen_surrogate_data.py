"""Generate MPM training data for the learned terrain surrogate (Phase 4).

Each episode: random dig cycle (start x, depth, drag length, curl angle, friction mu) of the task-3 bucket in a
3.4 m x 1.0 m (half) bed. Every 0.1 s we record, on a 5 cm grid:
  H     ground surface (top of soil, excluding soil in/on the bucket)
  Tlo   lowest point of the bucket solid in each column (NaN where the bucket is not above/in the column)
  pose  pivot x, z, pitch;  vel  pivot vx, vz, pitch rate;  F  soil force on bucket (Fx, Fz, Ty);  carried volume
  python -m digbench.gen_surrogate_data --arch cuda --episodes 120 --seed 0 --out /content/surrogate
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

from .run_bucket import BED, SURF, bucket_boxes, carried_volume
from .solvers.sim3d import Sim3D

CELL = 0.05


def make_path(rng):
    depth = rng.uniform(0.25, 0.8)
    drag = rng.uniform(0.6, 1.6)
    x0 = rng.uniform(2.5, 2.9)
    curl = math.radians(rng.uniform(55, 95))
    t_pen, t_drag, t_curl, t_lift = rng.uniform(0.7, 1.3), rng.uniform(1.4, 2.6), rng.uniform(1.0, 1.8), 1.0
    z0 = SURF + 1.2
    dz = 0.3 + depth

    def path(t):
        if t < t_pen:
            u = t / t_pen
            return np.array([x0 - 0.2 * u, 0.0, z0 - dz * u]), 0.0
        x1, z1 = x0 - 0.2, z0 - dz
        if t < t_pen + t_drag:
            u = (t - t_pen) / t_drag
            return np.array([x1 - drag * u, 0.0, z1]), 0.0
        x2 = x1 - drag
        if t < t_pen + t_drag + t_curl:
            u = (t - t_pen - t_drag) / t_curl
            return np.array([x2 - 0.1 * u, 0.0, z1 + 0.3 * u]), -curl * u
        u = min(1.0, (t - t_pen - t_drag - t_curl) / t_lift)
        return np.array([x2 - 0.1, 0.0, z1 + 0.3 + 0.7 * u]), -curl
    return path, t_pen + t_drag + t_curl + t_lift, {"depth": depth, "drag": drag, "x0": x0, "curl": curl}


def connected_surface(g, nxy, sp, zmax=4.0):
    """Top of the soil column connected to the bed: bin particles in z (bin = particle spacing) and stop at the
    first gap of two empty bins, so airborne spill is not counted as ground."""
    nz = int(zmax / sp) + 2
    occ = np.zeros((nxy[0], nxy[1], nz), dtype=bool)
    ix = np.clip((g[:, 0] / CELL).astype(int), 0, nxy[0] - 1)
    iy = np.clip((g[:, 1] / CELL).astype(int), 0, nxy[1] - 1)
    iz = np.clip((g[:, 2] / sp).astype(int), 0, nz - 1)
    occ[ix, iy, iz] = True
    gap = ~occ[:, :, :-1] & ~occ[:, :, 1:]
    first = np.where(gap.any(axis=2), gap.argmax(axis=2), nz - 1)
    return (first * sp).astype(np.float64)


def tool_points(boxes, n=14):
    pts = []
    for c, h, a in boxes:
        g = [np.linspace(-1, 1, n)] * 3
        X, Y, Z = np.meshgrid(*g, indexing="ij")
        pts.append(np.stack([c[0] + h[0] * X.ravel(), c[1] + h[1] * Y.ravel(), c[2] + h[2] * Z.ravel()], 1))
    return np.concatenate(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="surrogate_data")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nxy = (int(round(BED[0] / CELL)), int(round(BED[1] / CELL)))
    boxes = bucket_boxes(args.dx)
    TP = tool_points(boxes)
    sh = max(0.02, 1.25 * args.dx)  # shell thickness half-width used by bucket_boxes
    for ep in range(args.episodes):
        f = out / f"ep{args.seed:03d}_{ep:04d}.npz"
        if f.exists():
            continue
        rng = np.random.default_rng(args.seed * 100000 + ep)
        mu = float(rng.uniform(0.3, 0.7))
        path, T, meta = make_path(rng)
        t0 = time.perf_counter()
        sim = Sim3D(BED, args.dx, mu_f=mu, arch=args.arch, tool_speed_max=1.5,
                    max_particles=int(1.15 * BED[0] * BED[1] * SURF / (args.dx / 2) ** 3) + 1000)
        sim.fill_box((0, 0, 0), (BED[0], BED[1], SURF), rng_seed=ep)
        sim.set_tool(boxes)
        p0, a0 = path(0.0)
        sim.set_pose(p0, a0)
        sim.settle(0.3)
        Hs, Ts, poses, forces, carried = [], [], [], [], []

        def rec(s, row):
            pv, pa = np.array(row[2:5]), row[5]
            p = s.particles()
            r = p - np.array([pv[0], 0.0, pv[2]])
            ca, sa = math.cos(pa), math.sin(pa)
            lx = ca * r[:, 0] + sa * r[:, 2]; lz = -sa * r[:, 0] + ca * r[:, 2]
            on_b = (lx > -0.9 - 0.15) & (lx < 2 * sh + 0.15) & (lz > -0.9 - 2 * sh - 0.15) & (lz < 2 * sh + 0.15) & (r[:, 1] < 0.5 + 2 * sh + 0.15)
            H = connected_surface(p[~on_b], nxy, s.sp)
            wx = pv[0] + ca * TP[:, 0] - sa * TP[:, 2]; wz = pv[2] + sa * TP[:, 0] + ca * TP[:, 2]
            T_ = np.full(nxy, np.nan)
            ix = np.clip((wx / CELL).astype(int), 0, nxy[0] - 1); iy = np.clip((TP[:, 1] / CELL).astype(int), 0, nxy[1] - 1)
            ok = (wx >= 0) & (wx < BED[0]) & (TP[:, 1] >= 0)
            tmp = np.full(nxy, np.inf)
            np.minimum.at(tmp, (ix[ok], iy[ok]), wz[ok])
            T_[np.isfinite(tmp)] = tmp[np.isfinite(tmp)]
            Hs.append(H.astype(np.float32)); Ts.append(T_.astype(np.float32))
            poses.append([pv[0], pv[2], pa]); forces.append(row[6:9]); carried.append(carried_volume(p, s.sp, pv, pa))

        res = sim.drive(path, T, sample_every=0.1, on_sample=rec)
        np.savez_compressed(f, H=np.array(Hs), T=np.array(Ts), pose=np.array(poses, np.float32),
                            F=np.array(forces, np.float32), carried=np.array(carried, np.float32), mu=mu, dt=0.1,
                            cell=CELL, **{k: v for k, v in meta.items()})
        print(f"ep {ep}: mu={mu:.2f} depth={meta['depth']:.2f} drag={meta['drag']:.2f} frames={len(Hs)} "
              f"carried={carried[-1]:.2f} m3 wall={time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
