# Teacher (MPM) checks, v4

The learned soil model copies the 3D MPM solver, so the solver is checked first. Every number here is from a run;
files are under `results/v4/`. Published numbers are labelled as such.

## 1. Yield surface: Mohr-Coulomb replaces the legacy Drucker-Prager cone

`Sim3D(yield_model="mc", phi_deg, cohesion)`: Mohr-Coulomb in principal Kirchhoff stresses, Lode-angle-preserving radial
return at constant pressure (zero dilatancy), hydrostatic apex at c cot(phi). The legacy cone (alpha = sqrt(2)/3 mu_f) is
Mohr-Coulomb with sin(phi) = mu_f in plane strain, so "mu 0.6" behaved like phi 37 deg there and like 26 deg in triaxial
compression. Every soil parameter in the v1-v3 surrogate data was therefore ambiguous by up to ~10 deg.

## 2. Code verification against analytical solutions

**Flat blade, plane strain (no side-wall friction) vs Sokolovski passive pressure** (Coetzee 2004 Fig 6.14 reference
values for the same corn, `run_blade.py`):

| model | grid | h = 150 / 250 / 350 mm |
|---|---|---|
| MC | dx 6 mm | +4% / -4% / -7% |
| MC | dx 4 mm | +4% / -3% / -7% |
| legacy DP cone | dx 6 mm | +20% / +9% / +7% |

Grid-converged within 1 point. The legacy cone is 15-17% too strong, as its plane-strain friction angle predicts.

**Cohesion: unsupported vertical cut** (`run_cut.py`, c swept around theory, H = 1 m). Limit analysis brackets the critical
stability number N = H_c gamma / (c tan(45 + phi/2)) between 3.83 (Chen upper bound) and 4.0 (Culmann wedge); a
correct tension-capable MC should collapse just at or below the upper bound.

| phi | grid | N where the cut stands / collapses |
|---|---|---|
| 20 deg | dx 2.5 cm | 3.83 / 4.2 |
| 20 deg | dx 1.25 cm | 3.5 / 3.83 |
| 30 deg | dx 2.5 cm | 3.5 / 3.83 |

## 3. Validation against the lab experiment (bracketed)

Same blade test vs Coetzee's measured average draft force (96.6 / 205.8 / 386.1 N; glass side walls 200 mm apart):

| side-wall friction | grid | h = 150 / 250 / 350 mm |
|---|---|---|
| none (plane strain) | dx 4 mm | -32% / -19% / -19% |
| full glass friction, 12 deg | dx 4 mm | -7% / +31% / +54% |
| [PUBLISHED] Coetzee nonpolar MPM (plane strain) | - | -25% / -11% / -9% |
| [PUBLISHED] Coetzee DEM | - | +9% / +11% / -5% |

The measurement lies between the two side-wall bounds at every depth. The full-friction case over-predicts more with
depth (the wall-friction force grows ~h^3 in the model; the measured excess over plane strain shrinks with depth), so
the lab wall contact is weaker than full Coulomb friction on both panels. The field application has no glass walls, so
the plane-strain verification (section 2) is the check that matters for digging; the lab test only bounds it.

## 4. What is still NOT validated

* No measured dig geometry (surface after a bite) exists for this solver. Shape is verified only through the analytical
  checks above.
* Soil parameters per app preset are derived from the preset's own numbers (`soil_presets.py`: phi = repose angle,
  cohesion from Culmann at the preset's bank angle for a 1 m face, preset density), not from lab tests.
