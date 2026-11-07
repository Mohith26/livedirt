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


# --- Task 2: plate intrusion resistive force (granular RFT) ------------------
# Li, Zhang, Goldman, "A terradynamics of legged locomotion on granular media", Science 339, 1408 (2013),
# supplementary Table S1/S2/S3 (arXiv:1303.7065). Aluminium plate 3.81 x 2.54 x 0.64 cm moved at ~1 cm/s
# in a 15 cm deep air-fluidized bed; stress per unit depth alpha_{z,x}(beta, gamma) in N/cm^3 is the slope
# of F/A vs centre depth |z|. Run-to-run variation <= 0.005 N/cm^3. Fourier fit (their eq. S8):
#   alpha_z = sum_{m=-1..1, n=0..1} A_mn cos(2pi(m beta/pi + n gamma/2pi)) + B_mn sin(...)
#   alpha_x = sum C_mn cos(...) + D_mn sin(...)
# Medium used here: 3 mm glass spheres, closely packed (volume fraction 0.63, particle density 2.6 g/cm^3,
# repose 21 deg). Fit underestimates the one directly reported measurement alpha_z(0, pi/2) = 0.29 N/cm^3
# (fit 0.22; Table S3), so the fit itself carries ~25-33% error at its peak.
RFT_3MM_CP = {
    "source": "Li, Zhang, Goldman 2013, Science 339:1408, Table S2 (3 mm glass spheres, CP)",
    "A": {(0, 0): 0.045, (1, 0): 0.031},
    "B": {(1, 1): 0.046, (0, 1): 0.084, (-1, 1): 0.012},
    "C": {(1, 1): -0.031, (0, 1): 0.060, (-1, 1): 0.0},
    "D": {(1, 0): 0.015},
    "measured_alpha_z_0_90": 0.29,
    "rho_bulk": 2600.0 * 0.63,
    "mu_f": 0.38,  # tan(21 deg) repose; also the value used by Askari & Kamrin 2016 for glass beads
    "plate_mu": 0.4,  # aluminium-poppy seed value reported in the paper (glass value not reported)
}


def rft_alpha(beta, gamma, M=RFT_3MM_CP):
    """(alpha_x, alpha_z) in N/cm^3; beta, gamma in radians (Li et al. conventions)."""
    import math
    ang = lambda m, n: 2 * math.pi * (m * beta / math.pi + n * gamma / (2 * math.pi))
    az = sum(a * math.cos(ang(m, n)) for (m, n), a in M["A"].items()) + sum(b * math.sin(ang(m, n)) for (m, n), b in M["B"].items())
    ax = sum(c * math.cos(ang(m, n)) for (m, n), c in M["C"].items()) + sum(d * math.sin(ang(m, n)) for (m, n), d in M["D"].items())
    return ax, az
