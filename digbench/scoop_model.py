"""Per-scoop learned soil (v4): one inference per bite instead of one per 0.1 s.

Sample (from mpm_server --save-dir, or gen_replay --lean): the app's terrain on the bite's bed grid (window frame,
5 cm, margin M around the 3.4 m x 2.0 m window), the bucket's pose list for the whole bite (t, pivot x, pivot z, pitch,
relief, roll), soil parameters, and MPM's result: the surface change `delta` and the carried bank volume.

Inputs (all on the bed grid):
  0  terrain - anchor (anchor = mean terrain under the swept area)
  1  swept envelope: lowest bucket point per cell over the bite, minus terrain, clipped to [-1.5, 0.5] (0.5 where untouched)
  2  untouched mask (1 where the bucket never passes over the cell)
  3  physics prior: cut + relax delta (material above the envelope removed, then slopes relaxed to the friction angle)
  4-6  soil: (phi - 30) / 10, log10(1 + c) / 3, (rho - 1800) / 200   (constant planes)
Outputs: residual delta (target delta - prior delta) per cell, and scalars (carried bank m^3, completion, peak demand).

The residual is added back to the prior and the predicted terrain is passed through the same relax step at inference,
so no prediction leaves a slope steeper than the friction angle (INC-style projection after the correction).
"""
import json
import math
from pathlib import Path

import numpy as np

CELL = 0.05


def _bucket_points():
    from .gen_replay import bucket_boxes_full, tool_points
    return tool_points(bucket_boxes_full(0.04), n=10)


_BP = None


def swept_envelope(poses, nx, ny, M, C=CELL):
    """Lowest bucket point per bed cell over the whole bite (window-frame z), NaN where the bucket never passes."""
    global _BP
    if _BP is None:
        _BP = _bucket_points()
    from .attitude import lip_offset
    env = np.full((nx, ny), np.inf)
    P = np.asarray(poses, np.float64)
    for row in P[:: max(1, len(P) // 120)].tolist() + [P[-1].tolist()]:
        px, pz, pa = row[1], row[2], row[3]
        ro = row[5] if len(row) > 5 else 0.0
        ca, sa = math.cos(pa), math.sin(pa)
        x = ca * _BP[:, 0] - sa * _BP[:, 2]
        z = sa * _BP[:, 0] + ca * _BP[:, 2]
        y = _BP[:, 1]
        if ro != 0.0:
            lx, lz = lip_offset(pa)
            cr, sr = math.cos(ro), math.sin(ro)
            yy, zz = y, z - lz
            y, z = cr * yy - sr * zz, sr * yy + cr * zz + lz
        wx, wy, wz = px + x, y, pz + z
        i = np.floor((wx + M) / C).astype(int)
        j = np.floor((wy + 1.0 + M) / C).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        np.minimum.at(env, (i[ok], j[ok]), wz[ok])
    env[~np.isfinite(env)] = np.nan
    return env


def relax(H, repose_deg, C=CELL, iters=200):
    """Mass-conserving slope relaxation: any neighbour difference steeper than tan(repose) * C moves half the excess."""
    H = H.copy()
    lim = math.tan(math.radians(repose_deg)) * C
    for _ in range(iters):
        moved = 0.0
        for ax in (0, 1):
            d = np.diff(H, axis=ax)
            ex = np.where(np.abs(d) > lim, (np.abs(d) - lim) * np.sign(d) * 0.25, 0.0)
            if not np.any(ex):
                continue
            moved += float(np.abs(ex).sum())
            if ax == 0:
                H[:-1] += ex; H[1:] -= ex
            else:
                H[:, :-1] += ex; H[:, 1:] -= ex
        if moved < 1e-6:
            break
    return H


def cut_relax_delta(H, env, repose_deg, capacity_bank=0.95, C=CELL):
    """Physics prior: soil above the swept envelope is removed (up to the bucket capacity; the rest stays), then the
    surface relaxes to the repose angle. Returns (delta, removed bank m^3)."""
    cut = np.where(np.isfinite(env), np.maximum(0.0, H - env), 0.0)
    vol = cut.sum() * C * C
    if vol > capacity_bank:
        cut *= capacity_bank / vol
        vol = capacity_bank
    H1 = relax(H - cut, repose_deg, C)
    d = H1 - H
    # natural slopes far from the bite are not the bite's doing: keep the prior within 0.6 m of the swept cells
    near = np.isfinite(env)
    for _ in range(int(0.6 / C)):
        near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
    return np.where(near, d, 0.0), vol


def soil_planes(soil, shape):
    phi = float(soil.get("phi", math.degrees(math.atan(soil.get("mu", 0.6)))))
    c = float(soil.get("c", 0.0))
    rho = float(soil.get("rho", 1600.0))
    return np.stack([np.full(shape, (phi - 30) / 10), np.full(shape, math.log10(1 + c) / 3), np.full(shape, (rho - 1800) / 200)])


def make_inputs(terrain, poses, soil, M):
    nx, ny = terrain.shape
    env = swept_envelope(poses, nx, ny, M)
    touched = np.isfinite(env)
    anchor = float(terrain[touched].mean()) if touched.any() else float(terrain.mean())
    phi = float(soil.get("phi", math.degrees(math.atan(soil.get("mu", 0.6)))))
    prior, prior_vol = cut_relax_delta(terrain, env, phi)
    e = np.where(touched, np.clip(env - terrain, -1.5, 0.5), 0.5)
    X = np.concatenate([np.stack([terrain - anchor, e, (~touched).astype(float), prior]), soil_planes(soil, terrain.shape)]).astype(np.float32)
    return X, prior.astype(np.float32), prior_vol


def load_sample(f):
    z = np.load(f, allow_pickle=False)
    soil = json.loads(str(z["soil"]))
    meta = json.loads(str(z["meta"]))
    M = float(z["margin"])
    terrain = z["terrain"].astype(np.float64)
    X, prior, prior_vol = make_inputs(terrain, z["poses"], soil, M)
    d = z["delta"].astype(np.float32)
    y_s = np.array([meta["carried"], meta["completion"], min(meta["peak_demand"], 20.0)], np.float32)
    return {"X": X, "prior": prior, "delta": d, "scalars": y_s, "prior_vol": prior_vol, "name": Path(f).stem, "soil": soil}
