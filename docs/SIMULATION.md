# Closed-loop earthmoving simulation

The machine changes the ground, and the ground pushes back. This document describes the soil simulator behind
LiveDirt and how it is measured. Every number lives in [`digbench/RESULTS.md`](../digbench/RESULTS.md), generated
from the JSON each run writes; nothing here is typed by hand.

## Layers

| Layer | What it is | Where |
|---|---|---|
| Heightfield | LiveDirt's original real-time terrain: bank + loose layers, repose/critical-angle relaxation on a 10 cm grid | `app/src/sim/heightfield.ts`, Python port `digbench/solvers/heightfield.py` |
| 2D MPM | MLS-MPM, Drucker-Prager sand (Hencky strain), plane-strain yield surface, optional quasi-2D sidewall friction and mu(I) | `digbench/solvers/mpm_dp.py` |
| 3D MPM + tool | Same constitutive model in 3D; a rigid tool made of oriented boxes (translation + pitch) imposes its velocity on grid nodes with Coulomb friction; the momentum removed is the soil force on the tool | `digbench/solvers/sim3d.py` |
| Closed-loop drive | The tool follows a commanded trajectory, but a force limit (like a hydraulic relief valve) slows it when the soil resists harder than the machine can push. DX225LC-7 ISO arm digging force (11,500 kgf) is the reference limit | `Sim3D.drive(f_max=...)` |
| Multiscale hybrid | MPM only in an active box around the bucket (frictional box walls stand in for the soil beyond), written back to the heightfield, which relaxes the far field | `digbench/run_hybrid.py` |
| Learned surrogate | U-Net trained on 3D MPM dig cycles: terrain change per 0.1 s and soil force from (terrain, bucket raster now/next, friction). Runs in the browser through onnxruntime | `digbench/train_surrogate.py`, `app/src/sim/neuralSoil.ts` |
| Calibration | Infer soil friction from one observed slump profile (noisy, drone-like), then predict digging force | `digbench/run_calibrate.py` |

## DigBench tasks

1. **Granular column collapse** vs Sarlin et al. 2021 (Phys. Rev. E 104, 064904): runout and final height over seven aspect ratios.
2. **Plate intrusion stress maps** vs Li, Zhang, Goldman 2013 (Science 339, 1408): 36 plate angle / motion angle pairs, the measured basis of resistive force theory.
3. **Full-scale bucket cycle**: penetrate, drag, curl, lift; forces, carried volume, trench shape, grid convergence, force-limited closed loop.
4. **Multiscale hybrid** vs full-site MPM: terrain error and speedup as a function of active-zone margin.
5. **Calibration**: observe a slump, infer friction, predict bucket drag; plus a fit of friction to the real collapse data.
6. **Surrogate**: held-out MPM episodes, autoregressive rollouts, against LiveDirt's cut-and-relax given the same bucket raster.

## Rules the benchmark follows

- Material inputs come from the reference papers (friction from measured repose angles, densities, plate friction), not from fitting the benchmark. Where a parameter is fitted (task 5, real data), the fit is reported as a result.
- Every claimed improvement has an ablation in the results table (for example the 3D-cone vs plane-strain yield surface, sidewall friction on/off, grid resolution).
- A convention mapping (task 2's plate-angle sign) was checked on the cases it cannot affect before being applied.
- Runs that were later found to be scored wrongly are kept in `digbench/results/v1/` and labelled superseded.

## Reproduce

```bash
pip install taichi==1.7.4 numpy torch onnx
python -m digbench.run_collapse --arch cuda --dx 0.00125 --wall-mu 0.21 --side-W 0.15 --side-mu 0.21 --tag q2d-const-dx1.25mm
python -m digbench.run_plate --arch cuda --grid coarse --dx 0.003 --tag map-dx3mm
python -m digbench.run_bucket --arch cuda --dx 0.03 --tag v2-dx3cm
python -m digbench.run_hybrid --arch cuda --dx 0.04 --margins 0.25,0.5,1.0 --tag v2-dx4cm
python -m digbench.run_calibrate --arch cuda --tag v1
python -m digbench.gen_surrogate_data --arch cuda --episodes 100 --out data/surrogate --worker 0 --num-workers 1
python -m digbench.train_surrogate --data data/surrogate --tag v1
python -m digbench.make_report
cd app && npx tsx bench/neural-parity.ts ../digbench/results/surrogate-v1.onnx ../digbench/results/surrogate-v1-fixture.json
```

`digbench/colab_queue.sh` runs a job list in the background on a Colab GPU; `digbench/fetch_results.py` returns
results as text when browser downloads are blocked.

## Known gaps

- Task 1: final deposit height stays about 20% low even with sidewall friction; flow-rate friction (mu(I)) and back-wall friction were tested and do not close it. The fitted friction angle sits about two standard deviations above the measured repose angle, which points at the same missing physics (likely dilatancy of the dense initial packing).
- Task 2 compares against the paper's Fourier fit, which itself under-reports the one directly published measurement by about 25%.
- Task 3 has no public measured data at full scale; it is scored by convergence and conservation, with the contact model validated in task 2.
- The surrogate is trained on one bucket geometry and dry sand with friction 0.3 to 0.7.
- Surrogate v1 was trained only on single digs from a flat bed. Run inside the autonomy loop (multi-pass trench on the
  captured site, `app/bench/autonomy.ts --neural`) it left its training range: 115k clipped cell updates, over-dig and a
  610 kN peak drag prediction. The browser port itself matches PyTorch (parity bench); the gap is the data. v2 trains on
  chained bites over rough, already-dug terrain with the bucket emptied between bites.
- The force-limit model applies one limit along the bucket's velocity. Vertical penetration therefore stalls at the
  arm-force limit, although a real machine also pushes down with the boom and its weight; per-actuator limits through
  the linkage Jacobian are the next step.
