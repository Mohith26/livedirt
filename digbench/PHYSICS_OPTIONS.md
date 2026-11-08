# Options to improve the physics approach (2026-11-08)

Measured here: speed-lever sweep (`results/speedsweep/`, Task-3 bite, 0.6 m, volume-corrected, physical attitude, L40S
shared with a training job, so wall times are relative) and a re-read of existing loop and replay logs. Everything else is
published evidence, labelled as such.

## 1. The heightfield and MPM disagree on the baseline's own bites (existing data, new reading)

| same 32 bites (baseline controller, same start terrain) | soil per bite | bites that stall (completion < 0.65) |
|---|---|---|
| heightfield (deployed) | 0.11 m^3 (avg fill 10.5% of 1.06 m^3) | 0 |
| MPM, volume-corrected, mu 0.3-0.65 (`v3vc/replay`, t1 traces) | 0.40 m^3 mean (0.38 at mu >= 0.55) | 12 / 32 |
| learned model it2-fv in the loop | 0.49 m^3 per cycle (31.1 m^3 / 64) | - |

The learned model agrees with MPM on bite size; the heightfield is the outlier. The controller was tuned in a world where a
bite takes about a tenth of a bucket, so in MPM-like soil it over-digs and over-runs relief. This predicts that the trench
score in MPM itself would also be low, i.e. a large part of "3% vs 34.5%" is the controller/heightfield pairing.

## 2. Making MPM faster: measured levers

Cost model fitted from the sweep: wall/step ~= 1.6 ms fixed + 1.6 ns x particles.

| lever | speed | effect vs dx 4 cm, E 1e6 (drag 19.6 kN, carried 0.80 m^3, trench 0.25 m) | verdict |
|---|---|---|---|
| softer soil E 3e5 (dt x1.8) | 1.8x | drag -18%, carried -8% | reject |
| E 1e5 (dt x3.2) | 3.2x | drag -41%, carried -28% (dx 2 cm: drag -35%, carried -10%) | reject |
| coarser grid dx 6 cm | ~1.9x | carried -25%, trench 0.20 m | reject |
| dx 8 cm | ~2.7x | carried 0.075 m^3, trench 0.01 m (bucket plates thinner than a cell) | reject |
| fixed per-step overhead (Python-side stepping) | up to ~3x on a 0.5M-particle bite, ~1.1x on a 10.7M bed | none (same physics) | do: fuse substeps on the GPU |
| active zone only (Task 4 hybrid, measured earlier) | 2.9x (1 m margin), 5.8x (0.25 m) | region RMSE 0.083 m, carried 0%, drag +4% | do; try sleeping particles |

Replay beds hold 10.7M particles (~86 m^3) to move < 1 m^3 per bite (461 s per bite). The waste is in inactive particles.

## 3. Training differently: published evidence

- Predict one bite per inference instead of 0.1 s steps (Aoshima, Faelldin, Wadbro, Servin, arXiv 2309.12016): 97% pile-state
  accuracy, 4.5 ms, feasible over 40 sequential loadings. Our loop runs 5,366 autoregressive steps per trench (it2-fv).
- Learn a residual on top of cut+relax and re-apply relax after each step (Ajay et al. 2018; Um et al. 2020, >70% over
  supervised with long unrolled training). Caveat (INC, NeurIPS 2025): direct output corrections amplify perturbations, so
  the projection after the correction is part of the design.
- Train through whole bites: our curriculum stops at 32 steps (3.2 s) while bites last 5-16 s.
- Interleave the full solver (Kumar & Choi 2023, GNS/MPM): 20-24x faster than MPM alone with lower error than pure GNS.
  Offline only.
