"""Deposit measurement shared by every solver, so all are scored identically."""
import numpy as np


def profile_from_particles(x, y, dx, spacing):
    """Top-surface height per column of width dx from particle positions (wall at x=0)."""
    nb = int(np.ceil(x.max() / dx)) + 2
    h = np.zeros(nb)
    idx = np.clip((x / dx).astype(int), 0, nb - 1)
    np.maximum.at(h, idx, y + 0.5 * spacing)
    return np.arange(nb) * dx + 0.5 * dx, h


def deposit(xc, h, L0, grain_d):
    """Hf = max deposit height; Lf = farthest point where thickness >= one grain diameter."""
    Hf = float(h.max())
    # contiguous from the wall: detached clumps beyond a gap do not count as runout
    below = np.nonzero(h < grain_d)[0]
    first_gap = below[0] if len(below) else len(h)
    Lf = float(xc[first_gap - 1] + 0.5 * (xc[1] - xc[0])) if first_gap > 0 else 0.0
    return {"Hf": Hf, "Lf": Lf, "dL": Lf - L0}


def score(case, dep, ref_runout, ref_height):
    a, L0 = case["a"], case["L0"]
    r_sim, h_sim = dep["dL"] / L0, dep["Hf"] / L0
    r_ref, h_ref = ref_runout(a), ref_height(a)
    return {
        "runout_sim": r_sim, "runout_ref": r_ref, "runout_relerr": (r_sim - r_ref) / r_ref,
        "height_sim": h_sim, "height_ref": h_ref, "height_relerr": (h_sim - h_ref) / h_ref,
    }
