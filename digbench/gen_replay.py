"""v3 surrogate data: full-width, open-ground MPM beds that match the deployment loop.

* No symmetry plane: the bed is the full width, the bucket has both side plates, forces are not doubled.
* Open ground: the bed extends `--margin` (default 1.5 m) beyond the 3.4 m x 2.0 m model window on every side, and
  frames are cropped from the interior, so no wall is ever in view. `--margin 0` gives the walled ablation.
* Terrain and bucket poses come from the app: traces exported by `app/bench/autonomy.ts --trace` (the autonomy's
  actual bites on the captured site, through multi-pass trenches, lane offsets, and - for DAgger traces recorded with
  the model in the loop - the states the model itself visits). The bucket is driven along the trace with MPM's own
  per-actuator relief (digbench/actuators.py), so stalls and slowdowns are physically consistent.
* Frames are anchored exactly as app/src/sim/neuralSoil.ts: window frame z = world + shift with
  shift = 1.2 - mean(window surface at the first frame).
* Synthetic sets in the same format: `--standard` (task-3 cycle, physical attitude, flat ground, mu 0.38) and
  `--chained N` (rough ground, N random bites per bed, bucket dumped between bites, re-anchored per bite).

Every 0.1 s: H (68 x 40 window, connected surface excluding soil on/in the bucket), T (lowest bucket point per window
column, NaN elsewhere), pose (window frame pivot x, z, pitch), F (soil force on bucket Fx, Fz, Ty; full width),
carried (bank m^3 inside the bucket), s (relief speed scale), demand (max actuator fraction).

  python -m digbench.gen_replay --traces traces.json --out ~/v3/replay --arch cuda
  python -m digbench.gen_replay --standard --out ~/v3/heldout
  python -m digbench.gen_replay --chained 2 --beds 5 --seed 900 --out ~/v3/heldout
"""
import argparse
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np

from .attitude import physical_pitch, pivot_from_lip, lip_offset
from .run_bucket import bucket_boxes
from .solvers.sim3d import Sim3D, connected_surface
from . import actuators as A

CELL = 0.05
VOLCORR = [False]  # set from --volume-correction
# soil model for new beds: {"model": "dp"|"mc", "phi": deg or None (atan(mu)), "c": cohesion Pa}; set from CLI / server
YIELD = {"model": "dp", "phi": None, "c": 0.0, "rho": 1600.0}
NX, NY = 68, 40  # model window: 3.4 m (x, radial) x 2.0 m (lateral, -1..1 m)
SURF = 1.2


def bucket_boxes_full(dx):
    t = 0.05
    return bucket_boxes(dx) + [((-0.45, -(0.5 + t), -0.45), (0.45, t, 0.45), 0.0)]  # second side plate (y = -0.5)


def tool_points(boxes, n=14):
    pts = []
    for c, h, a in boxes:
        g = [np.linspace(-1, 1, n)] * 3
        X, Y, Z = np.meshgrid(*g, indexing="ij")
        pts.append(np.stack([c[0] + h[0] * X.ravel(), c[1] + h[1] * Y.ravel(), c[2] + h[2] * Z.ravel()], 1))
    return np.concatenate(pts)


def synthetic_poses(depth, drag, x_pivot0=2.7, surf=SURF, t_pen=1.0, t_drag=2.0, t_curl=1.5, curl_end=None):
    """Physical-attitude bite as a 0.1 s pose list in the window frame (pivot starts at x = 2.7 like the app)."""
    lip0 = np.array([0.0, surf + 0.3])
    piv0 = pivot_from_lip(lip0, physical_pitch("penetrate", 0))
    dx0 = x_pivot0 - piv0[0]
    z_cut = surf - depth
    out = []
    T = t_pen + t_drag + t_curl
    end_drag = None
    for k in range(int(round(T / 0.1)) + 1):
        t = k * 0.1
        if t < t_pen:
            u = t / t_pen
            lip, p = np.array([-0.2 * u, lip0[1] + (z_cut - lip0[1]) * u]), physical_pitch("penetrate", u)
            pv = pivot_from_lip(lip, p)
        elif t < t_pen + t_drag:
            u = (t - t_pen) / t_drag
            lip, p = np.array([-0.2 - drag * u, z_cut]), physical_pitch("drag", u)
            pv = pivot_from_lip(lip, p)
        else:
            if end_drag is None:
                end_drag = pivot_from_lip([-0.2 - drag, z_cut], physical_pitch("drag", 1.0))
            u = min(1.0, (t - t_pen - t_drag) / t_curl)
            pv, p = end_drag + np.array([-0.1 * u, 0.3 * u]), physical_pitch("curl", u)
        out.append([t, pv[0] + dx0, pv[1], p, 1.0])
    return out


class Episode:
    """One bed (MPM sim) that can run several bites; frames are cropped to the window and anchored per bite."""

    def __init__(self, terrain_win, margin, mu, dx, arch, num_workers, r0, ground_win, seed, min_floor_win, poses=None, col_depth=None):
        self.M = margin
        self.nxb, self.nyb = terrain_win.shape
        self.zfloor = float(min(terrain_win.min(), min_floor_win)) - 0.5  # bed z = window z - zfloor
        Hb = terrain_win - self.zfloor
        self.size = (self.nxb * CELL, self.nyb * CELL, float(Hb.max()) + 2.6)
        # per-column floor (speed): soil deeper than col_depth below the local surface is replaced by a sticky floor,
        # except within 1 m of anything the bucket sweeps, where the floor stays 0.5 m below the lowest lip point
        bottom = None
        if col_depth is not None:
            bottom = np.maximum(0.0, Hb - col_depth)
            if poses is not None:
                X, Y = np.meshgrid((np.arange(self.nxb) + 0.5) * CELL, (np.arange(self.nyb) + 0.5) * CELL, indexing="ij")
                near = np.zeros_like(Hb, bool)
                for pp in poses:
                    px = pp[1] + margin
                    near |= (np.abs(X - (px - 0.45)) < 0.45 + 1.0 + 0.3) & (np.abs(Y - (margin + 1.0)) < 0.6 + 1.0)
                bottom = np.where(near, 0.0, bottom)
        npart = int(1.12 * float((Hb - (bottom if bottom is not None else 0)).sum()) * CELL * CELL / (dx / 2) ** 3) + 20000
        self.sim = Sim3D(self.size, dx, mu_f=mu, arch=arch, symmetry_y=False, tool_speed_max=1.5, seed=seed, volume_correction=VOLCORR[0],
                         mem_frac=0.85 / num_workers, max_particles=npart, yield_model=YIELD["model"], phi_deg=YIELD["phi"],
                         cohesion=YIELD["c"], rho_bulk=YIELD.get("rho", 1600.0))
        self.n0 = self.sim.fill_heightfield(Hb, CELL, rng_seed=seed, bottom=bottom)
        if bottom is not None:
            self.sim.set_floor(bottom, CELL)
        self.boxes = bucket_boxes_full(dx)
        self.sim.set_tool(self.boxes)
        self.TP = tool_points(self.boxes)
        self.mu, self.r0, self.ground_win = mu, r0, ground_win
        self.i0, self.j0 = int(round(margin / CELL)), int(round(margin / CELL))
        self.yb = margin + 1.0  # bucket centreline (window lateral 0) in bed y

    def bed_pose(self, px, pz, pitch, roll=0.0):
        pos = np.array([px + self.M, self.yb, pz - self.zfloor])
        if roll != 0.0:
            # the app tilts the bucket about its cutting-edge centre; Sim3D rolls about the tool origin (the pivot), so
            # shift the pivot to keep the lip centre fixed: pivot += l - R_x(roll) l with l = lip offset (x, 0, z)
            lz = float(lip_offset(pitch)[1])
            pos = pos + np.array([0.0, math.sin(roll) * lz, lz * (1.0 - math.cos(roll))])
        return pos, pitch, roll

    def window_H(self, p, pv, pa, sh=0.05):
        r = p - np.array([pv[0], pv[1], pv[2]])
        ca, sa = math.cos(pa), math.sin(pa)
        lx = ca * r[:, 0] + sa * r[:, 2]; lz = -sa * r[:, 0] + ca * r[:, 2]
        # soil inside the bucket envelope (interior + shell, small mouth/top pad) is carried, not ground. Soil BELOW the
        # floor plate is ground: an earlier padded envelope (0.15 m below the shell, as in the v1/v2 generator) carved
        # the readout so the ground under the bucket was always >= 0.12 m below it, unlike the deployed window.
        on_b = (lx > -0.95) & (lx < 2 * sh + 0.05) & (lz > -0.9 - 2 * sh) & (lz < 2 * sh + 0.1) & (np.abs(r[:, 1]) < 0.5 + 2 * sh + 0.05)
        Hb = connected_surface(p[~on_b], (self.nxb, self.nyb), CELL, self.sim.sp, zmax=self.size[2])
        return Hb[self.i0:self.i0 + NX, self.j0:self.j0 + NY] + self.zfloor

    def window_T(self, pv, pa):
        ca, sa = math.cos(pa), math.sin(pa)
        TP = self.TP
        wx = pv[0] + ca * TP[:, 0] - sa * TP[:, 2] - self.M
        wy = TP[:, 1] + (pv[1] - self.yb) + 1.0
        wz = pv[2] + sa * TP[:, 0] + ca * TP[:, 2] + self.zfloor
        ix, iy = np.floor(wx / CELL).astype(int), np.floor(wy / CELL).astype(int)
        ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
        tmp = np.full((NX, NY), np.inf)
        np.minimum.at(tmp, (ix[ok], iy[ok]), wz[ok])
        T = np.full((NX, NY), np.nan)
        T[np.isfinite(tmp)] = tmp[np.isfinite(tmp)]
        return T

    def bed_surface(self, p, pv=None, pa=0.0, sh=0.05):
        """Connected soil surface over the whole bed (bed grid, window-frame z), excluding soil on/in the bucket."""
        if pv is not None:
            r = p - np.array([pv[0], pv[1], pv[2]])
            ca, sa = math.cos(pa), math.sin(pa)
            lx = ca * r[:, 0] + sa * r[:, 2]; lz = -sa * r[:, 0] + ca * r[:, 2]
            on_b = (lx > -0.95) & (lx < 2 * sh + 0.05) & (lz > -0.9 - 2 * sh) & (lz < 2 * sh + 0.1) & (np.abs(r[:, 1]) < 0.5 + 2 * sh + 0.05)
            p = p[~on_b]
        return connected_surface(p, (self.nxb, self.nyb), CELL, self.sim.sp, zmax=self.size[2]) + self.zfloor

    def run_bite_lean(self, poses, limits=True, settle=0.25):
        """One bite with no per-frame readout: settled surface before, surface after, carried bank volume, force stats,
        completion. For the MPM-in-the-loop test and per-scoop data."""
        P = np.array(poses, dtype=np.float64)
        T_end = float(P[-1, 0])

        def path(t):
            t = min(t, T_end)
            px, pz, pa = (np.interp(t, P[:, 0], P[:, c]) for c in (1, 2, 3))
            ro = float(np.interp(t, P[:, 0], P[:, 5])) if P.shape[1] > 5 else 0.0
            return self.bed_pose(px, pz, pa, ro)

        p0, a0, r0_ = path(0.0)
        self.sim.set_pose(p0, a0, r0_)
        t0 = time.perf_counter()
        self.sim.settle(settle)
        H_before = self.bed_surface(self.sim.particles(), np.asarray(p0), a0)
        dem_log, stall, nend, fx_log, fz_log = [], [0], [0], [], []

        def limiter(fx, fz, ty, pb, ab):
            piv = (self.r0 + pb[0] - self.M, pb[2] + self.zfloor - self.ground_win)
            lip = np.array([pb[0], pb[2]]) + lip_offset(ab)
            d = A.demand(fx, fz, ty, piv, bucket_xy=(self.r0 + lip[0] - self.M, lip[1] + self.zfloor - self.ground_win))
            dem_log.append(d["max_frac"])
            return limits and d["max_frac"] > 1.0

        last = {"tcmd": 0.0, "s": 1.0, "pv": p0, "pa": a0}

        def rec(s, row):
            last.update(tcmd=row[1], s=row[9], pv=np.array(row[2:5]), pa=row[5])
            fx_log.append(row[6]); fz_log.append(row[7])
            if row[1] >= T_end - 1e-6:
                nend[0] += 1
            stall[0] = stall[0] + 1 if row[9] <= 0.051 else 0
            return nend[0] > 3 or stall[0] >= 30

        res = self.sim.drive_fast(path, T_end * 1.6 + 0.5, limiter=limiter, sample_every=0.1, on_sample=rec, speed_floor=0.05)
        p = self.sim.particles()
        H_after = self.bed_surface(p, last["pv"], last["pa"])
        carried = _carried_full(p, self.sim.sp, last["pv"], last["pa"])
        return {"H_before": H_before, "H_after": H_after, "carried": float(carried), "completion": float(last["tcmd"] / T_end),
                "min_s": float(min([r[9] for r in res["series"]], default=1.0)), "peak_demand": float(max(dem_log, default=0.0)),
                "drag_mean_N": float(-np.mean(fx_log)) if fx_log else 0.0, "drag_peak_N": float(-np.min(fx_log)) if fx_log else 0.0,
                "vert_peak_N": float(np.max(np.abs(fz_log))) if fz_log else 0.0, "wall": time.perf_counter() - t0,
                "steps": res["steps"], "particles": int(self.n0)}

    def run_bite(self, poses, f, meta, limits=True):
        P = np.array(poses, dtype=np.float64)
        T_end = float(P[-1, 0])

        def path(t):
            t = min(t, T_end)
            px, pz, pa = (np.interp(t, P[:, 0], P[:, c]) for c in (1, 2, 3))
            ro = float(np.interp(t, P[:, 0], P[:, 5])) if P.shape[1] > 5 else 0.0
            return self.bed_pose(px, pz, pa, ro)

        p0, a0, r0_ = path(0.0)
        self.sim.set_pose(p0, a0, r0_)
        self.sim.settle(0.25)
        dem_log = []

        def limiter(fx, fz, ty, pb, ab):
            # machine frame: r = r0 + window x, y = window z - ground (window frame)
            piv = (self.r0 + pb[0] - self.M, pb[2] + self.zfloor - self.ground_win)
            lip = np.array([pb[0], pb[2]]) + lip_offset(ab)
            d = A.demand(fx, fz, ty, piv, bucket_xy=(self.r0 + lip[0] - self.M, lip[1] + self.zfloor - self.ground_win))
            dem_log.append(d["max_frac"])
            return limits and d["max_frac"] > 1.0

        Hs, Ts, poses_w, forces, carried, srel, tcmd, dem = [], [], [], [], [], [], [], []
        done = {"n": 0, "stall": 0}

        def rec(s, row):
            if done["n"] > 3:
                return True
            pv, pa = np.array(row[2:5]), row[5]
            p = s.particles()
            Hs.append(self.window_H(p, pv, pa).astype(np.float32))
            Ts.append(self.window_T(pv, pa).astype(np.float32))
            poses_w.append([pv[0] - self.M, pv[2] + self.zfloor, pa])
            forces.append(row[6:9])
            carried.append(_carried_full(p, s.sp, pv, pa))
            srel.append(row[9]); tcmd.append(row[1]); dem.append(dem_log[-1] if dem_log else 0.0)
            if row[1] >= T_end - 1e-6:
                done["n"] += 1
            # the app's relief never stops the bucket (speed >= 0.05x); end a bite stalled at the floor for 3 s
            done["stall"] = done["stall"] + 1 if row[9] <= 0.051 else 0
            return done["n"] > 3 or done["stall"] >= 30

        t0 = time.perf_counter()
        res = self.sim.drive_fast(path, T_end * 1.6 + 0.5, limiter=limiter, sample_every=0.1, on_sample=rec, speed_floor=0.05)
        H = np.array(Hs)
        # anchor exactly as the app: window frame shift so the first frame's window mean is 1.2
        zs = SURF - float(H[0].mean())
        np.savez_compressed(f, H=H + zs, T=np.array(Ts) + zs, pose=(np.array(poses_w) + np.array([0.0, zs, 0.0])).astype(np.float32),
                            F=np.array(forces, np.float32), carried=np.array(carried, np.float32), s=np.array(srel, np.float32),
                            tcmd=np.array(tcmd, np.float32), demand=np.array(dem, np.float32), mu=self.mu, dt=0.1, cell=CELL,
                            zshift=zs, margin=self.M, meta=json.dumps(meta))
        pe, ae, _ = path(1e9)
        return {"frames": len(Hs), "carried": carried[-1] if carried else 0.0, "wall": time.perf_counter() - t0,
                "completion": float(tcmd[-1] / T_end) if tcmd else 0.0, "min_s": float(min(srel)) if srel else 1.0,
                "peak_demand": float(max(dem_log)) if dem_log else 0.0, "end_pose": (pe, ae)}

    def dump_and_clear(self, pe, ae):
        def in_bucket(x):
            r = x - pe
            ca, sa = math.cos(ae), math.sin(ae)
            lx = ca * r[:, 0] + sa * r[:, 2]; lz = -sa * r[:, 0] + ca * r[:, 2]
            return (lx > -0.95) & (lx < 0.05) & (lz > -0.95) & (lz < 0.1) & (np.abs(r[:, 1]) < 0.55)
        n = self.sim.remove_where(in_bucket)
        self.sim.set_pose(np.array([pe[0], pe[1], self.size[2] - 0.2]), 0.0)
        self.sim.settle(0.3)
        return n


def _carried_full(p, sp, pv, pa):
    r = p - pv
    ca, sa = math.cos(pa), math.sin(pa)
    lx = ca * r[:, 0] + sa * r[:, 2]
    lz = -sa * r[:, 0] + ca * r[:, 2]
    inside = (lx > -0.9) & (lx < 0.0) & (lz > -0.9) & (lz < 0.0) & (np.abs(r[:, 1]) < 0.5)
    return float(inside.sum() * sp ** 3)


def rough_terrain(rng, nx, ny, amp):
    X, Y = np.meshgrid(np.arange(nx) * CELL, np.arange(ny) * CELL, indexing="ij")
    H = np.full(X.shape, SURF)
    for _ in range(4):
        kx, ky = rng.uniform(2.0, 8.0), rng.uniform(0.0, 4.0)
        H += amp / 2 * math.sin(rng.uniform(0, 6.3)) * np.cos(kx * X + ky * Y + rng.uniform(0, 6.3))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--yield-model", choices=["dp", "mc"], default="dp")
    ap.add_argument("--phi", type=float, default=None, help="MC friction angle (deg); default atan(mu)")
    ap.add_argument("--cohesion", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=1600.0)
    ap.add_argument("--preset", default=None, help="app soil preset -> MC params (overrides yield/phi/cohesion/rho; mu = tan phi)")
    ap.add_argument("--lean", action="store_true", help="replays: no per-frame readout; save surface before/after + carried only")
    ap.add_argument("--traces", nargs="*", default=[])
    ap.add_argument("--standard", action="store_true")
    ap.add_argument("--chained", type=int, default=0, help="bites per bed (synthetic rough-ground chains)")
    ap.add_argument("--beds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu-range", type=float, nargs=2, default=[0.3, 0.7])
    ap.add_argument("--limit", type=int, default=0, help="max traces to run (0 = all)")
    ap.add_argument("--select", default="", help="comma list of trace indices to run (overrides --limit)")
    ap.add_argument("--no-limits", action="store_true")
    ap.add_argument("--volume-correction", action="store_true", help="Tampubolon 2017 volume correction (no spurious bulking)")
    ap.add_argument("--particle-seed", type=int, default=0, help="offset added to the particle-jitter seed (aleatoric-variance runs)")
    ap.add_argument("--col-depth", type=float, default=None, help="per-column floor: simulate only this much soil below "
                    "the surface away from the bucket's reach (speed; validated against the full column)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    VOLCORR[0] = args.volume_correction
    YIELD.update(model=args.yield_model, phi=args.phi, c=args.cohesion, rho=args.rho)
    if args.preset:
        from .soil_presets import mc_params
        sp_ = mc_params(args.preset)
        YIELD.update(model="mc", phi=sp_["phi"], c=sp_["c"], rho=sp_["rho"])
        args.mu_range = [math.tan(math.radians(sp_["phi"]))] * 2
    M = args.margin
    nxb, nyb = int(round((3.4 + 2 * M) / CELL)), int(round((2.0 + 2 * M) / CELL))
    jobs = []
    if args.standard:
        jobs.append(("standard_task3", None))
    for b in range(args.beds if args.chained else 0):
        jobs.append((f"chain{args.seed:03d}_{b:03d}", b))
    for tf in args.traces:
        d = json.loads(Path(tf).read_text())
        tr = d["traces"]
        idx = [int(i) for i in args.select.split(",")] if args.select else list(range(len(tr)))
        if args.limit and not args.select:
            idx = idx[:args.limit]
        for i in idx:
            jobs.append((f"{Path(tf).stem}_{i:03d}", (d, tr[i])))
    for k, (name, spec) in enumerate(jobs):
        if k % args.num_workers != args.worker:
            continue
        rng = np.random.default_rng(zlib.crc32(f'{args.seed}:{name}'.encode()))
        if name == "standard_task3":
            f = out / "standard_task3.npz"
            if f.exists():
                continue
            terr = np.full((nxb, nyb), SURF)
            poses = synthetic_poses(0.6, 1.2)
            ep = Episode(terr, M, 0.38, args.dx, args.arch, args.num_workers, r0=4.0, ground_win=SURF, seed=k,
                         min_floor_win=min(p[2] for p in poses) - 1.0, poses=poses, col_depth=args.col_depth)
            r = ep.run_bite(poses, f, {"source": "standard", "depth": 0.6, "drag": 1.2}, limits=not args.no_limits)
            print(f"{name}: frames={r['frames']} carried={r['carried']:.2f} completion={r['completion']:.2f} "
                  f"peak_demand={r['peak_demand']:.2f} wall={r['wall']:.0f}s particles={ep.n0}", flush=True)
        elif name.startswith("chain"):
            if (out / f"{name}_{args.chained - 1}.npz").exists():
                continue
            mu = float(rng.uniform(*args.mu_range))
            terr = rough_terrain(rng, nxb, nyb, 0.15)
            ep = Episode(terr, M, mu, args.dx, args.arch, args.num_workers, r0=4.0, ground_win=SURF, seed=k,
                         min_floor_win=SURF - 1.1)
            for b in range(args.chained):
                Hnow = ep.window_H(ep.sim.particles(), np.array([0.0, -10.0, -10.0]), 0.0)
                local = float(np.percentile(Hnow[50:58, 14:26], 98))
                depth, drag = float(rng.uniform(0.15, 0.6)), float(rng.uniform(0.6, 1.6))
                poses = synthetic_poses(depth, drag, x_pivot0=float(rng.uniform(2.5, 2.9)), surf=local)
                r = ep.run_bite(poses, out / f"{name}_{b}.npz", {"source": "chained", "bite": b, "depth": depth, "drag": drag},
                                limits=not args.no_limits)
                ep.dump_and_clear(*r["end_pose"])
                print(f"{name} bite {b}: mu={mu:.2f} depth={depth:.2f} drag={drag:.2f} frames={r['frames']} carried={r['carried']:.2f} "
                      f"completion={r['completion']:.2f} peak_demand={r['peak_demand']:.2f} wall={r['wall']:.0f}s particles={ep.n0}", flush=True)
        else:
            d, tr = spec
            f = out / f"{name}.npz"
            if f.exists():
                continue
            mu = float(rng.uniform(*args.mu_range))
            n_full = tr["nx"], tr["ny"]
            terr = np.array(tr["terrain"], dtype=np.float64).reshape(n_full)
            m_tr = tr["margin"]
            cut = int(round((m_tr - M) / CELL))
            if cut > 0:
                terr = terr[cut:-cut, cut:-cut]
            poses = tr["poses"]
            lip_min = min(p[2] + lip_offset(p[3])[1] for p in poses)
            ep = Episode(terr, M, mu, args.dx, args.arch, args.num_workers, r0=tr["r0"], ground_win=tr["groundY"] + tr["shift"],
                         seed=k + 1000 * args.particle_seed, min_floor_win=lip_min, poses=poses, col_depth=args.col_depth)
            meta = {"source": "replay", "name": name, "bite": tr["bite"], "pass": tr["pass"],
                    "soil": tr["soil"], "neural": tr["neural"], "attitude": tr.get("attitude", "")}
            if args.lean:
                r = ep.run_bite_lean(poses, limits=not args.no_limits)
                np.savez_compressed(f, H_before=r.pop("H_before").astype(np.float32), H_after=r.pop("H_after").astype(np.float32),
                                    margin=M, cell=CELL, mu=mu, meta=json.dumps({**meta, **r, "yield": YIELD}))
                print(f"{name}: mu={mu:.2f} carried={r['carried']:.3f} completion={r['completion']:.2f} drag={r['drag_mean_N']/1e3:.1f}kN "
                      f"peak_demand={r['peak_demand']:.2f} wall={r['wall']:.0f}s particles={ep.n0}", flush=True)
                continue
            r = ep.run_bite(poses, f, meta, limits=not args.no_limits)
            print(f"{name}: mu={mu:.2f} frames={r['frames']} carried={r['carried']:.2f} completion={r['completion']:.2f} "
                  f"min_s={r['min_s']:.2f} peak_demand={r['peak_demand']:.2f} wall={r['wall']:.0f}s particles={ep.n0}", flush=True)


if __name__ == "__main__":
    main()
