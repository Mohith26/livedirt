"""App soil presets (app/src/sim/soil.ts) mapped to Mohr-Coulomb MPM parameters, using only the preset's own numbers.

  phi = reposeDeg (loose angle of repose ~ critical-state friction angle of the granular skeleton)
  c   = cohesion that makes a REF_H-deep planar face at bankDeg exactly critical by Culmann's wedge:
        c = gamma H (1 - cos(beta - phi)) / (4 sin(beta) cos(phi))     (0 when bankDeg <= phi)
  rho = density (t/m^3 -> kg/m^3)
These are ASSUMPTIONS that make the MPM soil consistent with the heightfield's slough rule; they are not lab values.
Verification of the MPM cohesion against limit analysis: run_cut.py (vertical cut, N in [2, 4]).
"""
import math

REF_H = 1.0
PRESETS = {  # id: (reposeDeg, bankDeg, density t/m^3, swell)
    "wet-clay": (30, 62, 1.9, 1.3),
    "common-earth": (34, 50, 1.8, 1.25),
    "sandy-gravel": (36, 40, 1.95, 1.12),
    "saturated-silt": (22, 32, 1.85, 1.2),
}


def culmann_c(phi_deg, beta_deg, rho, H=REF_H):
    if beta_deg <= phi_deg:
        return 0.0
    b, f = math.radians(beta_deg), math.radians(phi_deg)
    return rho * 9.81 * H * (1 - math.cos(b - f)) / (4 * math.sin(b) * math.cos(f))


def mc_params(soil_id):
    rep, bank, dens, swell = PRESETS[soil_id]
    rho = dens * 1000.0
    return {"model": "mc", "phi": float(rep), "c": round(culmann_c(rep, bank, rho), 1), "rho": rho, "swell": swell}


if __name__ == "__main__":
    for k in PRESETS:
        print(k, mc_params(k))
