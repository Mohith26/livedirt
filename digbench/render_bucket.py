"""Render a bucket dig cycle from the 3D MPM as an animated GIF: side cutaway (particles near the bucket
centreplane), coloured by speed, with the bucket outline and the live soil-force trace.

  python -m digbench.render_bucket --arch cuda --dx 0.03 --out digbench/results/bucket-cycle.gif
"""
import argparse
import math

import numpy as np

from .run_bucket import BED, SURF, bucket_boxes, path
from .solvers.sim3d import Sim3D


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--dx", type=float, default=0.03)
    ap.add_argument("--mu-f", type=float, default=0.38)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="bucket-cycle.gif")
    args = ap.parse_args()
    sim = Sim3D(BED, args.dx, mu_f=args.mu_f, arch=args.arch, tool_speed_max=1.5,
                max_particles=int(1.15 * BED[0] * BED[1] * SURF / (args.dx / 2) ** 3) + 1000)
    sim.fill_box((0, 0, 0), (BED[0], BED[1], SURF))
    boxes = bucket_boxes(args.dx)
    sim.set_tool(boxes)
    p0, a0 = path(0.0)
    sim.set_pose(p0, a0)
    sim.settle(0.3)
    frames, rows = [], []
    vfield = sim.v

    def rec(s, row):
        n = int(s.n_p[None])
        p = s.particles()
        v = vfield.to_numpy()[:n][s.alive.to_numpy()[:n] == 1]
        sl = p[:, 1] < 0.12  # slab next to the centreplane (y = 0 is the symmetry plane)
        frames.append((p[sl][:, [0, 2]].copy(), np.linalg.norm(v[sl], axis=1), np.array(row[2:6])))
        rows.append(row)

    sim.drive(lambda t: path(t), 5.5, sample_every=1.0 / args.fps, on_sample=rec)
    ser = np.array(rows)
    fig, (ax, axf) = plt.subplots(2, 1, figsize=(8, 6.2), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#111")
    for a in (ax, axf):
        a.set_facecolor("#111"); a.tick_params(colors="#bbb"); [sp.set_color("#444") for sp in a.spines.values()]
    sc = ax.scatter([], [], s=1.2, c=[], cmap="afmhot", vmin=-0.5, vmax=1.2)  # static soil stays visible (brown)
    (outline,) = ax.plot([], [], color="#7fd4ff", lw=2)
    ax.set_xlim(0, BED[0]); ax.set_ylim(0, SURF + 1.6); ax.set_aspect("equal")
    ax.set_title(f"3D MPM dig cycle (centreplane slab), sand mu={args.mu_f}, dx={args.dx*100:g} cm", color="#ddd", fontsize=10)
    axf.plot(ser[:, 0], ser[:, 6] / 1e3, color="#555"); axf.plot(ser[:, 0], ser[:, 7] / 1e3, color="#555")
    (fx_line,) = axf.plot([], [], color="#ffb347", label="drag Fx (kN)")
    (fz_line,) = axf.plot([], [], color="#7fd4ff", label="vertical Fz (kN)")
    axf.legend(facecolor="#222", labelcolor="#ddd", fontsize=8, loc="upper left"); axf.set_xlabel("time (s)", color="#bbb")

    def outline_xy(pivot, pitch):
        pts = []
        ca, sa = math.cos(pitch), math.sin(pitch)
        for lx, lz in [(-0.9, 0.0), (0.0, 0.0), (0.0, -0.9), (-0.9, -0.9)]:
            pts.append((pivot[0] + ca * lx - sa * lz, pivot[2] + sa * lx + ca * lz))
        return np.array(pts)

    def draw(i):
        xz, sp, pose = frames[i]
        sc.set_offsets(xz); sc.set_array(sp)
        o = outline_xy(pose[:3], pose[3]); outline.set_data(o[:, 0], o[:, 1])
        fx_line.set_data(ser[: i + 1, 0], ser[: i + 1, 6] / 1e3); fz_line.set_data(ser[: i + 1, 0], ser[: i + 1, 7] / 1e3)
        return sc, outline, fx_line, fz_line

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    fig.savefig(args.out.replace(".gif", "-last.png"), dpi=110)
    print("wrote", args.out, len(frames), "frames")


if __name__ == "__main__":
    main()
