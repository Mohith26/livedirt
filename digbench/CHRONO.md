# Project Chrono baseline (external open-source solver)

Code: `digbench/solvers/chrono_baseline.py` (SCM), `digbench/solvers/chrono_crm.py` (CRM), runner `digbench/run_chrono.py`.
Results: `digbench/results/chrono-*.json`. Every number below comes from a run; items marked QUEUED have not run.

## Install (AWS L40S box, isolated)
Miniforge in `~/miniforge3`, env `chrono`: `conda create -n chrono python=3.12`, `conda install projectchrono::pychrono -c conda-forge`
(pychrono 10.0.0, py312). `import pychrono, pychrono.vehicle, pychrono.fsi` all succeed. Modules present: core, fea, fsi, irrlicht,
pardisomkl, parsers, postprocess, robot, sensor, vehicle, vsg3d. SCM = `pychrono.vehicle.SCMTerrain`; CRM = `pychrono.fsi`
(`ChFsiProblemCartesian`, `SoilProperties`), CUDA. Run from a copy in `~/chrono_work` (not the data-generation checkout), CPU runs under
`nice -n 19` with `OMP_NUM_THREADS=2`. The env has no taichi, so `run_chrono.py` stubs the taichi import to reuse `attitude.py`,
`run_bucket.bucket_boxes`, `metrics.py` and `references.py` unchanged.

## Task 3: bucket dig, SCM (CPU)
Bucket: the four `bucket_boxes` plates plus the mirrored y = -0.5 side plate as one rigid collision body, driven kinematically along
`attitude.physical_path(0.6, 1.2)` (5.5 s), dt = 1 ms. Terrain 3.4 x 2.0 m, surface at z = 1.2 (SCM z = 0). Force = `GetContactForceBody`.
Drag mean is over t in [1.2, 3.0) s as in `run_bucket`.

SCM parameters (dry sand, none fitted to MPM):
| param | value | source |
|---|---|---|
| Kphi | 1.52843e6 N/m^(n+2) | Wong dry sand (NASA/TM-20250006958 table of Bekker values) |
| Kc | 990 N/m^(n+1) | same |
| n | 1.1 | same |
| Mohr friction | atan(0.38) = 20.8 deg | the task's bulk friction (run_bucket --mu-f), not Wong's 28 deg |
| cohesion | 0 Pa | dry sand treated as cohesionless; Wong's 1.04 kPa run as a sensitivity |
| Janosi K | 0.025 m | Wong dry sand (as tabulated in secondary sources) |
| elastic_K, damping_R | 2e8 Pa/m, 3e4 Pa s/m | Chrono SCM demo values (not in original SCM) |
| bulldozing | on, erosion angle 20.8 deg, flow 1, 3 iterations, 10 propagations | angle of repose = friction angle; rest Chrono defaults |
| grid delta | 0.05 m (0.025 m for convergence) | |

Results (tag: drag mean / drag peak / vertical peak kN, wall s, RTF):
| tag | drag mean | drag peak | Fz peak | wall | RTF |
|---|---|---|---|---|---|
| scm-d5cm (main) | 35.2 | 283 | 781 | 4.9 | 1.13 |
| scm-d2.5cm | 37.5 | 280 | 771 | 17.6 | 0.31 |
| scm-d5cm-nobulldoze | 32.4 | 224 | 618 | 2.7 | 2.01 |
| scm-d5cm-c1.04kPa | 35.3 | 284 | 781 | 4.9 | 1.13 |
| scm-d5cm-kphi0.5 | 16.0 | 150 | 415 | 4.7 | 1.17 |
MPM reference runs for the same path (bucket-v3-phys-dx4/dx2-d0.6-none): drag mean 19.9 / 21.3 kN, drag peak 51 / 53 kN, Fz peak 74 / 79 kN.
SCM drag mean is 1.7x the MPM value and its peaks are 5-10x higher; the result is highly sensitive to Kphi (halving it halves the drag),
which is not constrained by the task. A coarse grid (delta 0.1 m, 1.5 s smoke run, not saved) gave 66 kN, so use delta <= 0.05 m.
Limitations: SCM is a surface (heightfield) model with vertical-ray contact. Carried volume is NOT MODELED (`carried_bank_m3 = null`).
Bekker parameters come from small plate-sinkage tests and are extrapolated here to a 0.6 m deep, 1 m wide bucket; pressure-sinkage
loads then give very large vertical forces during penetration. The heightfield output (`H_scm`, 34 x 10 cells of 0.1 m, y >= 0)
records the SCM-deformed terrain; the `terrain_volume_change_bank_m3` field is negative (SCM removes more height than bulldozing
redistributes), so no terrain RMSE against MPM is claimed.

## Task 2: plate RFT, SCM: not scored
SCM casts vertical rays from terrain nodes against the body; a plate below the surface does not load the soil above it, and an
edge-on (vertical) plate has no vertical-ray footprint. A probe (`chrono-plate-probe.json`, 6 orientations, plate 3.81 x 2.54 x 0.64 cm,
5 cm below the surface, 0.1 m/s, 5 cm travel, delta 1 cm) gives nearly orientation-independent forces
(Fx 107-156 N, Fz 593-821 N across (beta, gamma) = (0,0), (0,90), (90,0), (30,0), (-30,0), (60,90)), unlike the strongly anisotropic Li et al. map.
Not scored against `rft_alpha`; treat the RFT map as not representable by SCM.

## Task 1: column collapse
SCM: not applicable (no granular flow). CRM: QUEUED on the GPU queue (`chrono-collapse-crm-sp5mm.json`, a = 1 and 3, 5 mm spacing,
15 cm channel). Task 3 CRM: QUEUED (`chrono-bucket-crm-sp4cm.json`, 4 cm spacing, full-width bed). Fetch with
`scp 'digaws:chrono_work/res/*.json' digbench/results/` once `~/queue.log` shows both END lines. Neither CRM path has executed on a GPU yet
(construction was checked CPU-only), so CRM may still need API fixes.
