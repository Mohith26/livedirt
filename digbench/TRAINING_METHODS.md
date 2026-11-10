# Training methods for the per-scoop model: what was measured (2026-11-09/10)

All on the fixed WSP split (train 197 scoops -> validation 107-108, split by whole trench run), same recipe as
train_scoop (U-Net ch 32, 100 epochs, AdamW + OneCycle, y-flip), pooled RMS over disturbed cells, 3 seeds each.
Code: `digbench/train_exp.py`, `digbench/exp_report.py`, `digbench/al_sim.py`. Results: `results/wsp/exp/`.

| Method | Validation RMS (cm) | Verdict |
|---|---|---|
| Baseline | 10.55 +- 0.30 | |
| x-mirror augmentation (physics is mirror-symmetric) | 10.97 +- 0.16 | worse (skewed test distribution: 87% of bites drag -x) |
| Sequential-fill prior (fill bucket in path order) | 10.58 +- 0.47 | no change |
| EMA weights (0.998) | 10.47 +- 0.28 | within noise |
| Flip test-time augmentation | 13.6 (base) / 10.85 (x-mirror models) | hurts unless trained symmetric |
| 3-seed ensemble | 9.82 | 7% better |
| Warm start from the 75%-data model, 30 epochs | 10.24 +- 0.38 in 20 s vs 67 s | as good, 3.3x faster, 35% less churn (3.9 vs 6.0 cm), 12 vs 30 bites regressed > 1 cm |
| Shrink-and-perturb warm start (0.5, 0.1) | 10.22 +- 0.13 | no better than plain warm start; more churn (5.0 cm) |
| Path-progress input channel (drag direction + order) | 10.27 / 10.54 vs 10.20 / 10.38 | no gain |

Active learning (which scoops to label): seed disagreement vs per-bite error, Spearman 0.85 (top-quarter disputed bites
14.2 cm vs 8.3 cm mean). Pool simulation (98 labelled + 49 added): random 11.41 cm single / 10.46-10.57 ensemble;
by disagreement 10.67 / 9.94; all 197 random 10.55 / 9.82. About 25% fewer simulator runs for the same accuracy
(one simulated draw; indicative).

New data (DAgger rounds 1-3, +224 scoops): from scratch 8.06 +- 0.01 single, 7.54 ensemble (deep cells 28.2 -> 18.8,
stalled 13.7 -> 9.7). Warm start for a doubling of data was worse (8.48 / 7.98): use warm start for small top-ups and
retrain from scratch when the data grows a lot.

Closed loop (a8 = a7 recipe on 583 scoops vs a7 on 246): per-bite held-out 8.0-8.4 vs 9.0-9.2 cm, but gate 1 (sandy
gravel) final-surface error 22.8 / 21.2 cm vs 13.6 / 12.7 (a local over-dug pit, 8% of design cells > 40 cm too deep;
elsewhere a8 is closer); gate 2 (wet clay) 15.8 vs 16.2 cm. Promotion must use the closed-loop gates, not per-bite error.

## Continuous loop (digbench/al_loop, started 2026-11-10 00:35 UTC)
Diggers run random scenarios (+-60 deg heading) with the newest ensemble in the loop and score every bite by ensemble
disagreement into a queue; three MPM labellers always take the most disputed bite; the train loop retrains every +30
labels (warm start, scratch every 3rd version), runs both gates, and writes `~/v4/loop/ledger.jsonl` and `BEST.json`.
