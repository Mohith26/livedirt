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
from .solvers.sim3d import Sim3D, connected_surface

CELL = 0.05


def make_path(rng, surf=SURF, depth_range=(0.25, 0.8)):
    depth = rng.uniform(*depth_range)
    drag = rng.uniform(0.6, 1.6)
    x0 = rng.uniform(2.5, 2.9)
    curl = math.radians(rng.uniform(55, 95))
    t_pen, t_drag, t_curl, t_lift = rng.uniform(0.7, 1.3), rng.uniform(1.4, 2.6), rng.uniform(1.0, 1.8), 1.0
    z0 = surf + 1.2
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


def tool_points(boxes, n=14):
    pts = []
    for c, h, a in boxes:
        g = [np.linspace(-1, 1, n)] * 3
        X, Y, Z = np.meshgrid(*g, indexing="ij")
        pts.append(np.stack([c[0] + h[0] * X.ravel(), c[1] + h[1] * Y.ravel(), c[2] + h[2] * Z.ravel()], 1))
    return np.concatenate(pts)


def record_episode(sim, path, T, nxy, TP, sh, mu, meta, f):
    Hs, Ts, poses, forces, carried = [], [], [], [], []

    def rec(s, row):
        pv, pa = np.array(row[2:5]), row[5]
        p = s.particles()
        r = p - np.array([pv[0], 0.0, pv[2]])
        ca, sa = math.cos(pa), math.sin(pa)
        lx = ca * r[:, 0] + sa * r[:, 2]; lz = -sa * r[:, 0] + ca * r[:, 2]
        on_b = (lx > -0.9 - 0.15) & (lx < 2 * sh + 0.15) & (lz > -0.9 - 2 * sh - 0.15) & (lz < 2 * sh + 0.15) & (r[:, 1] < 0.5 + 2 * sh + 0.15)
        H = connected_surface(p[~on_b], nxy, CELL, s.sp)
        wx = pv[0] + ca * TP[:, 0] - sa * TP[:, 2]; wz = pv[2] + sa * TP[:, 0] + ca * TP[:, 2]
        T_ = np.full(nxy, np.nan)
        ix = np.clip((wx / CELL).astype(int), 0, nxy[0] - 1); iy = np.clip((TP[:, 1] / CELL).astype(int), 0, nxy[1] - 1)
        ok = (wx >= 0) & (wx < BED[0]) & (TP[:, 1] >= 0)
        tmp = np.full(nxy, np.inf)
        np.minimum.at(tmp, (ix[ok], iy[ok]), wz[ok])
        T_[np.isfinite(tmp)] = tmp[np.isfinite(tmp)]
        Hs.append(H.astype(np.float32)); Ts.append(T_.astype(np.float32))
        poses.append([pv[0], pv[2], pa]); forces.append(row[6:9]); carried.append(carried_volume(p, s.sp, pv, pa))

    sim.drive(path, T, sample_every=0.1, on_sample=rec)
    np.savez_compressed(f, H=np.array(Hs), T=np.array(Ts), pose=np.array(poses, np.float32), F=np.array(forces, np.float32),
                        carried=np.array(carried, np.float32), mu=mu, dt=0.1, cell=CELL, **{k: v for k, v in meta.items()})
    return Hs, carried


def run_chain(args, out, nxy, boxes, TP, sh):
    """v2 data: rough initial terrain, several bites per bed with depth measured from the local surface, and the
    bucket emptied between bites, so episodes start from already-dug ground like the autonomy's multi-pass work."""
    for bed in range(args.episodes):
        if bed % args.num_workers != args.worker:
            continue
        if (out / f"chain{args.seed:03d}_{bed:04d}_{args.chain - 1}.npz").exists():
            continue
        rng = np.random.default_rng(10_000_000 + args.seed * 100000 + bed)
        mu = float(rng.uniform(0.3, 0.7))
        t0 = time.perf_counter()
        sim = Sim3D(BED, args.dx, mu_f=mu, arch=args.arch, tool_speed_max=1.5, mem_frac=0.85 / args.num_workers,
                    max_particles=int(1.4 * BED[0] * BED[1] * (SURF + args.rough) / (args.dx / 2) ** 3) + 1000)
        hx = np.arange(int(BED[0] / 0.05)) * 0.05
        hy = np.arange(int(BED[1] / 0.05)) * 0.05
        X, Y = np.meshgrid(hx, hy, indexing="ij")
        H0 = np.full(X.shape, SURF)
        for _ in range(4):  # smooth random field, wavelengths 0.8-3 m
            kx, ky = rng.uniform(2.0, 8.0), rng.uniform(0.0, 4.0)
            H0 += args.rough / 2 * math.sin(rng.uniform(0, 6.3)) * np.cos(kx * X + ky * Y + rng.uniform(0, 6.3))
        sim.fill_heightfield(H0, 0.05, rng_seed=bed)
        sim.set_tool(boxes)
        for b in range(args.chain):
            p = sim.particles()
            near = (np.abs(p[:, 0] - 2.7) < 0.3) & (p[:, 1] < 0.3)
            surf = float(np.percentile(p[near, 2], 98)) if near.any() else SURF
            path, T, meta = make_path(rng, surf=surf, depth_range=(0.15, 0.6))
            meta.update({"chain_index": b, "surface_at_start": surf})
            p0, a0 = path(0.0)
            sim.set_pose(p0, a0)
            sim.settle(0.25)
            Hs, carried = record_episode(sim, path, T, nxy, TP, sh, mu, meta, out / f"chain{args.seed:03d}_{bed:04d}_{b}.npz")
            # dump: remove everything inside the bucket envelope, then lift the bucket clear
            pe, ae = path(T)
            def in_bucket(x, pe=pe, ae=ae):
                r = x - np.array([pe[0], 0.0, pe[2]])
                ca, sa = math.cos(ae), math.sin(ae)
                lx = ca * r[:, 0] + sa * r[:, 2]; lz = -sa * r[:, 0] + ca * r[:, 2]
                return (lx > -0.95) & (lx < 0.05) & (lz > -0.95) & (lz < 0.1) & (r[:, 1] < 0.55)
            n_dumped = sim.remove_where(in_bucket)
            print(f"bed {bed} bite {b}: mu={mu:.2f} surf={surf:.2f} depth={meta['depth']:.2f} drag={meta['drag']:.2f} "
                  f"frames={len(Hs)} carried={carried[-1]:.2f} m3 dumped={n_dumped} wall={time.perf_counter()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="surrogate_data")
    ap.add_argument("--chain", type=int, default=0, help="v2: bites per bed (rough start, bucket dumped between bites)")
    ap.add_argument("--rough", type=float, default=0.15, help="v2: amplitude of the random initial terrain (m)")
    ap.add_argument("--standard", action="store_true", help="one episode: the task-3 cycle (depth 0.6, mu 0.38)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1, help="run N processes in parallel on one GPU to hide CPU overhead")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nxy = (int(round(BED[0] / CELL)), int(round(BED[1] / CELL)))
    boxes = bucket_boxes(args.dx)
    TP = tool_points(boxes)
    sh = 0.05  # shell half-thickness used by bucket_boxes
    if args.chain:
        run_chain(args, out, nxy, boxes, TP, sh)
        return
    for ep in range(args.episodes):
        if ep % args.num_workers != args.worker:
            continue
        f = out / f"ep{args.seed:03d}_{ep:04d}.npz"
        if f.exists() and not args.standard:
            continue
        rng = np.random.default_rng(args.seed * 100000 + ep)
        mu = float(rng.uniform(0.3, 0.7))
        path, T, meta = make_path(rng)
        if args.standard:
            from .run_bucket import path as std_path
            mu, T = 0.38, 5.5
            path = lambda t: std_path(t, 0.6)
            meta = {"depth": 0.6, "drag": 1.2, "x0": 2.7, "curl": math.radians(75)}
            f = out / "standard_task3.npz"
        t0 = time.perf_counter()
        sim = Sim3D(BED, args.dx, mu_f=mu, arch=args.arch, tool_speed_max=1.5, mem_frac=0.85 / args.num_workers,
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
            H = connected_surface(p[~on_b], nxy, CELL, s.sp)
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
