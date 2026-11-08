"""Bucket attitude for the task-3 rig (shared by run_bucket, the replay generator and app/src/machine/autonomy.ts).

The rig's tool frame has its pivot at the top-back corner and the cutting edge (lip) at (-0.9, -0.9). At pitch 0 the
floor plate is horizontal, so a vertical plunge at pitch 0 pushes a flat 0.9 m x 1.0 m plate straight down (the
legacy schedule; it produced ~850 kN of vertical force at 1.0 m depth, several times the machine weight).

Physical schedule ("teeth first"): penetrate with the lip leading and the floor trailing at PEN_PITCH (the floor plate
enters edge-on at a 40 deg attack angle), drag with the attack angle easing from PEN_PITCH to DRAG_END_PITCH as the
bucket fills, then curl from DRAG_END_PITCH to CURL_END_PITCH. Pitch is CCW positive in the (x, z) plane; the bucket
drags toward -x.
"""
import math

import numpy as np

LIP = np.array([-0.9, -0.9])
PEN_PITCH = math.radians(40.0)
DRAG_END_PITCH = math.radians(10.0)
CURL_END_PITCH = math.radians(-75.0)


def lip_offset(pitch):
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([c * LIP[0] - s * LIP[1], s * LIP[0] + c * LIP[1]])


def pivot_from_lip(lip_xz, pitch):
    return np.asarray(lip_xz, float) - lip_offset(pitch)


def physical_pitch(phase, u):
    """phase in {'penetrate', 'drag', 'curl', 'lift'}; u = progress through the phase in [0, 1]."""
    u = min(1.0, max(0.0, u))
    if phase == "penetrate":
        return PEN_PITCH
    if phase == "drag":
        return PEN_PITCH + (DRAG_END_PITCH - PEN_PITCH) * u
    if phase == "curl":
        return DRAG_END_PITCH + (CURL_END_PITCH - DRAG_END_PITCH) * u
    return CURL_END_PITCH


def physical_path(depth=0.6, surf=1.2, x_lip0=1.8, drag=1.2, t_pen=1.0, t_drag=2.0, t_curl=1.5, t_lift=1.0):
    """Task-3 cycle with the physical attitude. The LIP follows the legacy lip path through penetrate and drag (start
    0.3 m above the surface, 0.2 m back while plunging to `depth`, drag `drag` toward the machine); the curl raises the
    pivot 0.3 m while rotating to CURL_END_PITCH; the lift raises it 0.7 m. Returns path(t) -> (pivot xyz, pitch), T."""
    z_top, z_cut = surf + 0.3, surf - depth

    def lip_and_pitch(t):
        if t < t_pen:
            u = t / t_pen
            return np.array([x_lip0 - 0.2 * u, z_top + (z_cut - z_top) * u]), physical_pitch("penetrate", u)
        if t < t_pen + t_drag:
            u = (t - t_pen) / t_drag
            return np.array([x_lip0 - 0.2 - drag * u, z_cut]), physical_pitch("drag", u)
        return None, None

    end_drag_pivot = pivot_from_lip([x_lip0 - 0.2 - drag, z_cut], DRAG_END_PITCH)

    def path(t):
        lip, p = lip_and_pitch(t)
        if lip is not None:
            pv = pivot_from_lip(lip, p)
            return np.array([pv[0], 0.0, pv[1]]), p
        if t < t_pen + t_drag + t_curl:
            u = (t - t_pen - t_drag) / t_curl
            return np.array([end_drag_pivot[0] - 0.1 * u, 0.0, end_drag_pivot[1] + 0.3 * u]), physical_pitch("curl", u)
        u = min(1.0, (t - t_pen - t_drag - t_curl) / t_lift)
        return np.array([end_drag_pivot[0] - 0.1, 0.0, end_drag_pivot[1] + 0.3 + 0.7 * u]), CURL_END_PITCH

    return path, t_pen + t_drag + t_curl + t_lift
