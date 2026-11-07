"""Published reference data for DigBench tasks.

Every number here comes from a cited source. Nothing is tuned to a solver.
"""

# --- Task: quasi-2D granular column collapse ---------------------------------
# Sarlin, Morize, Sauret, Gondret, "Collapse dynamics of dry granular columns:
# from free-fall to quasi-static flow", Phys. Rev. E 104, 064904 (2021), arXiv:2112.11458.
# Glass beads d = 5 mm, rho = 2.5 g/cm^3, packing 0.64, repose angle 23.5 +/- 1.2 deg,
# rough floor of the same grains, channel 15 cm wide, gate lifted at 1.2 m/s (Series A,
# free-fall regime). Fits (their eqs. 1-2), inspired by Lajeunesse et al. (2005):
#   dL/L0 = 1.85 a          (a <~ 3)      2.67 a^(2/3)   (a >~ 3)
#   Hf/L0 = a               (a <~ 0.8)    0.93 a^(1/3)   (a >~ 0.8)
# Lf is measured one grain diameter above the floor.
# Exponents agree with Lajeunesse, Monnier, Homsy, Phys. Fluids 17, 103302 (2005)
# (rectangular channel); prefactors vary between setups, which Sarlin et al. note.
COLLAPSE = {
    "source": "Sarlin et al. 2021, Phys. Rev. E 104, 064904 (arXiv:2112.11458), Series A fits",
    "grain_d": 0.005,
    "rho_grain": 2500.0,
    "packing": 0.64,
    "repose_deg": 23.5,
    "repose_sd_deg": 1.2,
}


def collapse_runout(a: float) -> float:
    """Reference normalized runout dL/L0 for aspect ratio a = H0/L0."""
    return 1.85 * a if a <= 3 else 2.67 * a ** (2.0 / 3.0)


def collapse_height(a: float) -> float:
    """Reference normalized final height Hf/L0."""
    return a if a <= 0.8 else 0.93 * a ** (1.0 / 3.0)


# Lab geometries spanning Series A (H0 10-50 cm, L0 2.5-20 cm, a 0.5-20).
COLLAPSE_CASES = [
    {"a": 0.5, "H0": 0.10, "L0": 0.20},
    {"a": 1.0, "H0": 0.15, "L0": 0.15},
    {"a": 2.0, "H0": 0.20, "L0": 0.10},
    {"a": 3.0, "H0": 0.30, "L0": 0.10},
    {"a": 5.0, "H0": 0.375, "L0": 0.075},
    {"a": 8.0, "H0": 0.40, "L0": 0.05},
    {"a": 12.0, "H0": 0.45, "L0": 0.0375},
]
