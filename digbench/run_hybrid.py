"""DigBench task 4: multiscale hybrid (MPM active zone + LiveDirt heightfield far field) vs full-site MPM.

Site 8 m x 6 m (half-width 3 m by symmetry) of sand 1.2 m deep. The same bucket cycle as task 3 is run
(a) with MPM over the whole site (reference), (b) with MPM only in a box around the bucket's swept
footprint plus a margin, written back into the site heightfield and relaxed with LiveDirt's relax, and
(c) LiveDirt-style kinematic cut + relax (no MPM). Scores: terrain RMSE vs reference in the disturbed
region, carried volume, drag force, wall time.

  python -m digbench.run_hybrid --arch cuda --dx 0.04 --margins 0.25,0.5,1.0
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .run_bucket import BED, SURF, bucket_boxes, carried_volume, path
from .run_collapse import device_name, git_rev
from .solvers.heightfield import Heightfield
from .solvers.sim3d import Sim3D, connected_surface

ROOT = Path(__file__).resolve().parent
SITE = (8.0, 3.0, BED[2])
SHIFT = 2.3                     # dig starts at pivot x = 5.0 m
FOOT = (0.25 + SHIFT, 2.85 + SHIFT, 0.0, 0.5 + 0.1)   # swept bucket footprint (x0, x1, y0, y1), shell included
CELL = 0.1


def run_mpm(size, x_off, args, wall_mu, E=1.0e6, floor_z=0.0):
    """MPM over `size`; E: stiffness (lower = larger stable dt); floor_z: soil below this height is replaced by the
    sticky floor (only the column above it is simulated; heights are reported in site coordinates)."""
    t_setup = time.perf_counter()
    col = SURF - floor_z
    sim = Sim3D((size[0], size[1], size[2] - floor_z), args.dx, mu_f=args.mu_f, rho_bulk=1600.0, E=E, arch=args.arch,
                tool_speed_max=1.5, wall_mu=wall_mu, max_particles=int(1.1 * size[0] * size[1] * col / (args.dx / 2) ** 3) + 1000)
    sim.fill_box((0, 0, 0), (size[0], size[1], col))
    sim.set_tool(bucket_boxes(args.dx))
    def pth(t):
        pv, a = path(t, args.depth, SHIFT - x_off)
        return pv - np.array([0.0, 0.0, floor_z]), a
    p0, a0 = pth(0.0)
    sim.set_pose(p0, a0)
    t0 = time.perf_counter()
    sim.settle(0.3)
    t1 = time.perf_counter()
    res = sim.drive(pth, args.duration, sample_every=0.02)
    t2 = time.perf_counter()
    wall = t2 - t0
    ser = res["series"]
    p = sim.particles()
    pv, pa = ser[-1, 2:5], ser[-1, 5]
    carried = carried_volume(p, sim.sp, pv, pa)
    r = p - np.array([pv[0], 0.0, pv[2]])
    ca, sa = math.cos(pa), math.sin(pa)
    lx = ca * r[:, 0] + sa * r[:, 2]; lz = -sa * r[:, 0] + ca * r[:, 2]
    ground = ~((lx > -1.25) & (lx < 0.4) & (lz > -1.25) & (lz < 0.4) & (r[:, 1] < 0.85))
    nxy = (int(round(size[0] / CELL)), int(round(size[1] / CELL)))
    H = connected_surface(p[ground], nxy, CELL, sim.sp) + floor_z
    drag = (ser[:, 1] > 1.2) & (ser[:, 1] < 3.0)
    prof = {"setup_s": t0 - t_setup, "settle_s": t1 - t0, "drive_s": t2 - t1, "readout_s": time.perf_counter() - t2,
            "substeps": res["steps"], "dt": res["dt"], "ms_per_substep": 1e3 * (t2 - t1) / max(1, res["steps"]),
            "particles": int(sim.n_p[None]), "grid_nodes": int(np.prod(sim.n)),
            "ns_per_particle_substep": 1e9 * (t2 - t1) / max(1, res["steps"]) / max(1, int(sim.n_p[None]))}
    return {"H": H, "carried": carried, "drag_mean": float(ser[drag, 6].mean()), "drag_peak": float(ser[:, 6].max()),
            "wall_s": wall, "particles": int(sim.n_p[None]), "series": ser, "profile": prof}


def relax_region(Hsite, x0, x1, y1, ang):
    """LiveDirt relax over [x0, x1) x [0, y1) cells, mirrored across y = 0."""
    sub = Hsite[x0:x1, :y1]
    full = np.concatenate([sub[:, ::-1], sub], axis=1)
    hf = Heightfield(np.full_like(full, SURF), CELL, ang, ang)
    hf.B[:] = full
    hf.settle()
    out = Hsite.copy()
    out[x0:x1, :y1] = hf.h()[:, sub.shape[1]:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--depth", type=float, default=0.6)
    ap.add_argument("--mu-f", type=float, default=0.38)
    ap.add_argument("--duration", type=float, default=5.5)
    ap.add_argument("--margins", default="0.25,0.5,1.0")
    ap.add_argument("--wall-mu", type=float, default=None, help="zone wall friction (default = mu_f)")
    ap.add_argument("--skip-ref", action="store_true")
    ap.add_argument("--zone-E", type=float, default=1.0e6, help="zone stiffness (speed lever: larger stable dt)")
    ap.add_argument("--zone-floor-below", type=float, default=None, help="zone floor = SURF - depth - this (particle culling)")
    ap.add_argument("--tag", default="hybrid")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    wmu = args.mu_f if args.wall_mu is None else args.wall_mu
    ang = math.degrees(math.atan(args.mu_f))
    nsite = (int(SITE[0] / CELL), int(SITE[1] / CELL))
    out = {"task": "hybrid", "tag": args.tag, "device": device_name(args.arch), "git": git_rev(), "params": vars(args), "runs": []}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pth = Path(args.out) / f"hybrid-{args.tag}.json"

    cache = Path(args.out) / f"hybrid-ref-dx{args.dx:g}-d{args.depth:g}-mu{args.mu_f:g}.npz"
    if cache.exists():
        z = np.load(cache)
        ref = {"H": z["H"], "carried": float(z["carried"]), "drag_mean": float(z["drag_mean"]), "drag_peak": float(z["drag_peak"]),
               "wall_s": float(z["wall_s"]), "particles": int(z["particles"])}
    else:
        ref = run_mpm(SITE, 0.0, args, 0.0)
        out["reference_profile"] = ref["profile"]
        np.savez(cache, H=ref["H"], carried=ref["carried"], drag_mean=ref["drag_mean"], drag_peak=ref["drag_peak"],
                 wall_s=ref["wall_s"], particles=ref["particles"])
    Href = ref["H"]
    print(f"reference full-site MPM: {ref['particles']} particles, {ref['wall_s']:.0f}s, carried {ref['carried']:.3f} m3, "
          f"drag mean {ref['drag_mean']/1e3:.1f} kN", flush=True)
    changed = np.abs(Href - SURF) > 0.02
    region = np.zeros(nsite, bool)
    region[int(1.0 / CELL):int(7.5 / CELL), :int(2.5 / CELL)] = True   # scoring window around the dig
    out["reference"] = {k: ref[k] for k in ("carried", "drag_mean", "drag_peak", "wall_s", "particles")}
    out["reference"]["changed_cells"] = int(changed.sum())
    out["H_ref"] = Href.round(4).tolist()

    def score(name, H, extra):
        rm = float(np.sqrt(((H - Href)[region] ** 2).mean()))
        rmc = float(np.sqrt(((H - Href)[changed] ** 2).mean())) if changed.any() else float("nan")
        row = {"name": name, "rmse_region_m": rm, "rmse_changed_m": rmc, **extra}
        out["runs"].append(row)
        print(json.dumps(row), flush=True)
        pth.write_text(json.dumps(out))

    # (c) LiveDirt-style kinematic cut + relax
    from .run_bucket import kinematic_heightfield_baseline
    t0 = time.perf_counter()
    Hk_small, removed = kinematic_heightfield_baseline(CELL, (int(BED[0] / CELL), int(BED[1] / CELL)), args.depth)
    Hk = np.full(nsite, SURF)
    sx = int(round(SHIFT / CELL))
    Hk[sx:sx + Hk_small.shape[0], :Hk_small.shape[1]] = Hk_small
    Hk = relax_region(Hk, int(1.0 / CELL), int(7.5 / CELL), int(2.5 / CELL), ang)
    score("livedirt_kinematic_relax", Hk, {"wall_s": time.perf_counter() - t0, "carried": removed, "speedup": ref["wall_s"] / (time.perf_counter() - t0)})

    # (b) hybrid zones
    for m in [float(s) for s in args.margins.split(",")]:
        zx0 = max(0.0, FOOT[0] - m); zx1 = min(SITE[0], FOOT[1] + m); zy1 = min(SITE[1], FOOT[3] + m)
        zx0 = math.floor(zx0 / CELL) * CELL; zx1 = math.ceil(zx1 / CELL) * CELL; zy1 = math.ceil(zy1 / CELL) * CELL
        t0 = time.perf_counter()
        fz = max(0.0, SURF - args.depth - args.zone_floor_below) if args.zone_floor_below is not None else 0.0
        z = run_mpm((zx1 - zx0, zy1, SITE[2]), zx0, args, wmu, E=args.zone_E, floor_z=fz)
        H = np.full(nsite, SURF)
        i0 = int(round(zx0 / CELL))
        H[i0:i0 + z["H"].shape[0], :z["H"].shape[1]] = z["H"]
        H = relax_region(H, int(1.0 / CELL), int(7.5 / CELL), int(2.5 / CELL), ang)
        wall = time.perf_counter() - t0
        score(f"hybrid_margin_{m:g}m", H, {"wall_s": wall, "speedup": ref["wall_s"] / z["wall_s"], "particles": z["particles"],
                                           "carried": z["carried"], "carried_err": (z["carried"] - ref["carried"]) / ref["carried"],
                                           "drag_mean_err": (z["drag_mean"] - ref["drag_mean"]) / ref["drag_mean"],
                                           "drag_peak_err": (z["drag_peak"] - ref["drag_peak"]) / ref["drag_peak"],
                                           "zone": [zx0, zx1, zy1], "zone_E": args.zone_E, "zone_floor": fz, "profile": z["profile"]})
        out["H_" + f"hybrid_{m:g}"] = H.round(4).tolist()
    pth.write_text(json.dumps(out))
    print("->", pth)


if __name__ == "__main__":
    main()
