# Learned soil surrogate: course correction report

Target: the closed loop (`app/bench/autonomy.ts`, sandy-gravel, 1:1 trench, captured site) against the deployed
heightfield (cut+relax) baseline: **34.5% on grade, 5.34 cm MAE** (re-run twice, byte-identical). Every number below is
from a run; files under `digbench/results/` (held-out: `results/v3models/`, loop table: `results/loop-gate.md`) and `app/bench/results/`.

## Frozen evaluation

* Held-out set (generated once, never trained on): standard task-3 dig, 10 chained bites on rough ground (seed 900),
  10 replayed autonomy bites from a held-out trench (yaw +12 deg, start 4.4 m, length 3.4 m; trace indices
  0,3,...,28). Full-width open-ground MPM beds (window + 1.5 m margin), anchored as the app.
  Two versions of the same episodes: `v3/heldout` (MPM without volume correction; iteration 1) and `v3vc/heldout`
  (Tampubolon volume correction; iteration 2). Each model is rolled out from frame 0, with the deploy floor projection
  on and off.
* Loop gate: on-grade %, MAE, over-dig, unplanned cut, volume delta, clipped/floored counts, peak drag, network share.

## Ranked causes (confirmed / refuted, by which ablation)

| # | cause | verdict | evidence |
|---|---|---|---|
| 1 | Floor projection: hides raw over-removal, and training with it teaches over-removal | CONFIRMED (largest held-out effect) | Same recipe +/- floor in training (it2-fv vs it2-fv-floor), floor off at eval: 0.076 vs 0.201 m, volume bias +0.05 vs +1.09 m^3/bite. Loop it1-d: model dh removes 237 m^3, projection adds back 240 m^3; floor off: 72 m^3 removed for a ~7 m^3 trench. |
| 2 | Loop states outside the training data (v1/v2) | CONFIRMED | v2 loop H feature p01 -2.8 / p99 1.6, beyond the v2 generator's range; with replayed autonomy traces only 0.6-3% of loop inputs fall outside training [p0.1, p99.9]. Loop it0 0.0-0.5% -> it1 up to 4.9%. |
| 3 | Different held-out sets for v1/v2 | CONFIRMED | v2's own set 0.146 m; shared frozen set: v2 0.469, v1 0.337, cut+relax 0.313, persistence 0.282 m. |
| 4 | Surface readout carved 12 cm under the bucket (v1/v2 generator) | CONFIRMED, fixed | min(T - H) = +0.12 m in every episode with the padded envelope; after the fix the bucket enters the ground (-0.05 to -1.2 m). |
| 5 | Truth/data quality: MPM spurious bulking; flat-floor plunge | CONFIRMED in the solver; small at window scale | bucket bed bulking +0.65/+0.49/+0.40 m^3 (dx 4/3/2 cm) -> +0.05/+0.02/-0.01 with the volume correction (trench depth converges 0.25/0.25/0.24 m; drag within 2%); window-level +0.06 vs -0.10 m^3/episode. Iteration 2 on corrected truth: best held-out 0.076 m (floor off) vs 0.176 m (it1). Flat plunge: 853 kN vertical, replaced by teeth-first attitude. |
| 6 | Train/eval gap to the loop: compounding errors over multi-bite, multi-pass trenches | CONFIRMED (dominant in the loop, not fixed) | it2-fv: held-out single-bite error 0.076 m (below the ~0.10 m seed-to-seed spread of the MPM truth on one replay) yet loop 2.2% on grade, 44.5 cm MAE, 8.2 m^3 unplanned cut over 64 cycles; final mean surface 7 cm above grade but only ~2% of cells within the 4.5 cm tolerance. DAgger (32 replayed loop states) did not close it. |
| 7 | Force over-prediction -> relief-valve stall | MINOR for grade | relief off: 2.4% -> 3.0% (it1-full). Occupied-cell force pooling: drag error 89% -> 42% (it1), 52% (it2-fv); loop peak demand still 5-16x rated on some cylinder. |
| 8 | Unroll 4 only / curriculum / noise (bundle) | CONFIRMED as a bundle (iteration 1) | it1-d 0.179 vs it1-full 0.198 m held-out; loop 3.9% vs 2.4%. Components not separated. |
| 9 | No volume conservation (flux head + volume loss) | CONFIRMED on held-out, not in the loop | it2-fv (flux + volume loss) 0.076 m, bias +0.05 vs it2-full 0.092 m, +0.04; loop 2.2% vs 2.5%. Iteration 1 (uncorrected truth): flux 0.206 vs free 0.198. |
| 10 | Inputs never asymmetric (mirrored training) | REFUTED | same data: symmetrised vs not, held-out 0.184 vs 0.198 m (it1), 0.109 vs 0.092 m floor off (it2); loop it1 4.4% vs 2.4%, it2 3.6% vs 2.5% (mu 0.6). |
| 11 | >0.1 s pose deltas while a model step is in flight | REFUTED (bench) | max pose gap 0.1 s in every bench run; synchronous vs async v2: 3.8% vs 3.6%. Browser not measured. |
| 12 | Ensemble uncertainty as the safety gate | REFUTED as a guard | 3-member ensemble, held-out-calibrated gate (spread 0.066 m, OOR 0.36): never fired in the loop (spread 0.015 m). A priori consistency gate (bucket capacity / net creation): 3.9% -> 4.9%, network share 98%. |
| 13 | Walls in view (window = bed) | NOT YET MEASURED | walled VC t1 data and `walls_ablation.sh` are queued on AWS (results land in `~/v3vc/models/walls`). |

Gate caveat: dry sand at the gate's mu = 0.6 (31 deg) cannot hold the 1:1 (45 deg) design walls, and LiveDirt's
sandy-gravel preset uses a 40 deg bank angle; on-grade % therefore mixes material and model accuracy.

## Loop gate per iteration (sandy-gravel, 1:1, mu 0.6 unless noted)

| iteration | model / config | on grade % | MAE cm | network share |
|---|---|---|---|---|
| baseline | heightfield cut+relax (deployed) | 34.5 | 5.34 | - |
| 0 | v1 / v2 (floor on, per-actuator relief) | 0.5 / 0.0 | 108 / 320 | 100% |
| 1 | it1-d + a priori consistency gate (best) | 4.9 | 38.7 | 98% |
| 1 | it1-d, floor off / 3-member ensemble gated | 4.7 / 3.9 | 41.7 / 56.0 | 100% |
| 2 | it2-sym (floor off) / at mu 0.7 | 3.6 / 7.2 | 40.9 / 38.5 | 100% |
| 2 | it2-full (floor off) / at mu 0.7 | 2.5 / 3.6 | 44.7 / 44.6 | 100% |
| 2 | it2-fv (floor off) / floor on | 2.2 / 0.0 | 44.5 / 117.4 | 100% |

Stopping rule: iteration 2 did not beat iteration 1's 4.9% at mu 0.6 (best iteration-2 result at mu 0.6: 3.6%), so round 3
(t2/t3 replays) was cancelled. The mu 0.7 rows are reported, not used for the decision.

## Deploy / don't-deploy

**Don't deploy.** The heightfield stays the default. The surrogate now beats cut+relax and persistence on the frozen
held-out set by a wide margin (0.076 m vs 0.278 / 0.274 m final RMSE on disturbed cells, volume-neutral), but in the loop it
is 2-7% on grade against 34.5%, and no gate configuration made it "never worse than baseline".

## What the evidence says to do next (not run)

1. Replace the loop gate's reference: run the same autonomy with MPM as the soil (closed loop, bite by bite) and score the
   surrogate against that, not against the idealized cut+relax. Two facts make this necessary: the MPM truth itself varies by
   ~0.10 m between particle seeds on one bite (vs a 4.5 cm grade tolerance), and dry sand at the gate's mu cannot hold the
   1:1 design walls. Until that run exists, "34.5% vs 3%" cannot separate an inaccurate surrogate from accurate soil that the
   controller was never designed for.
2. Multi-bite training targets: train on episodes that chain several replayed bites and passes, so the model is penalised
   for errors that compound across bites (the held-out set scores one bite at a time).
3. Keep: floor off in training and deploy, volume-corrected truth, flux head + volume loss, occupied-cell force pooling.
