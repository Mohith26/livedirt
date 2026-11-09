# Can Bedrock-hosted open models do weight-space programming on the scoop model? (2026-11-09)

One session per model per task, same harness, prompts, practice tier and judge as the Opus/Astra plan. Base w0.
Editors: GLM-5 (zai.glm-5), Kimi K2.5 (moonshotai.kimi-k2.5), DeepSeek V3.2 (deepseek.v3.2), via Bedrock Converse.
Costs use deliberately high placeholder rates ($2/$8 per M tokens), not the price list.

## E1 read (10 feature-map passes). Key: e1_key.py (intervention, never shown to the editor), results/wsp/open/e1_key.json

| Q | Key (measured) | GLM-5 | Kimi K2.5 | DeepSeek V3.2 |
|---|---|---|---|---|
| 1 completion driver | ch 2 (untouched mask), negative; effect 0.24 vs next 0.12 | ch 1, neg: wrong | ch 1, neg: wrong | ch 5, neg: wrong |
| 2 soil use, e1 filters | encoder (2x skip path); 19, 22, 18 | decoder; 9, 23, 14: wrong | encoder; 14, 19, 5: half | encoder; 0, 1, 17: half |
| 3 stall channels in e4 | ablation 243, 184, 73, 215, 39; AUC 83, 213, 91, 43, 158 | 9, 56, 35, 20, 49: 0 overlap | 10, 13, 2, 4, 0: 0 overlap | 74, 68, 108, 9, 48: 0 overlap |
| 4 cut vs outside writers | cut 23, 24, 26, 17, 25, 21; outside 7, 3, 28, 9, 4, 8 | cut 1, 17, 23 ok; outside 24, 25, 27 wrong: half | mixed: wrong | sorted by out.weight sign: wrong |
| score (pass >= 3) | | ~1 of 4 | ~0.5 of 4 | ~0.5 of 4 |
| cost | | $2.81 | $5.13 | $1.55 |

Answers leaned on physics priors ("deeper cut -> stall") and single weight norms, not on tracing or comparing bites.

## E3 fix stalls (budget 10), judged on validation (accept: all -2 mm, no region worse by 2 mm)

| | all (m) | stalled | slump | accepted | what it did | cost |
|---|---|---|---|---|---|---|
| base w0 | 0.0981 | 0.1311 | 0.0846 | | | |
| GLM-5 | 0.0974 | 0.1319 | 0.0852 | no (gain 0.7 mm < 2 mm; stalled worse) | scaled the completion head 1.2x and similar; not a stall gate | $4.46 |
| Kimi K2.5 | 0.0993 | 0.1327 | 0.0868 | no (slump worse) | edits made everything slightly worse | $5.23 (hit cap) |
| DeepSeek V3.2 | 0.0981 | 0.1311 | 0.0846 | no | described a fix, changed no weights | $1.61 |

Verdict: none of the three passes E1 or E3; all are below the plan's "< 2 of 4" stop line for reading. Not tested:
budgets 25/50 (the paper's gain needed 50), second sessions. Useful as a weak-editor baseline for the frontier runs.

## Claude Opus 5.5 (Anthropic API, same harness, one session each), 2026-11-09

E1 read ($0.38, 14 steps, 0 of 10 probe passes used: feature_maps needs a bite name and nothing in the E1 task lists them;
both guessed names failed; answers are static weight analysis only)

| Q | Key | Opus | Score |
|---|---|---|---|
| 1 | ch 2, negative | ch 1 (+), runner-up ch 3; ch 2 ranked 4th | wrong |
| 2 | encoder; 19, 22, 18 | encoder; 17, 28, 24 (17 is key rank 7) | half |
| 3 | ablation 243, 184, 73, 215, 39; AUC 83, 213, 91, 43, 158 | 160, 98, 82, 95, 61 (its "will complete" list 215, 243, 225, 223, 113 holds 3 of the key's top-8 by ablation size) | wrong (found influential channels, labelled direction opposite to the key's framing) |
| 4 | cut 23, 24, 26, 17, 25, 21; outside 7, 3, 28, 9, 4, 8 | cut 24, 26, 17, 23 (+25); outside 28, 19, 27, 3, 7 | right on cut (5/5), 3 of 5 outside: ~1 |
| total | | | ~1.5 of 4 (open models 0.5-1); pass needs 3 |

E3 fix stalls ($1.94, 18 steps, 9 of 10 passes): diagnosed the cause (stall estimate only in the head, never reaches the
per-cell decoder), built a 4-layer gated side path (passive-wedge stall depth gate in e1.0 -> e1.3 -> d1.0 -> d1.3 -> out),
passed signals through GroupNorm by keeping hijacked groups below eps with zero-sum channels, verified the circuit (d1
activation 1274.6 vs 1275 designed), measured it on practice: v1 31.6 cm, v2 25.0 cm, takeover with correction off 18.1 cm
vs 11.8 base. Found the cost is taking over whole GroupNorm groups (1/8 of 4 layers), reverted, submitted no change.
Judge: unchanged (0.0981), not accepted. Harness gaps exposed: no bite-name listing for E1; no spare width for side paths.

## Claude Fable 5.1 (Anthropic API, same harness, one session each), 2026-11-09

E1 read ($1.09, 16 steps, 0 probe passes, same bite-name gap): Q1 ch 3 (-): wrong. Q2 encoder; 17, 24, 28: half.
Q3 82, 98, 160, 61, 245: wrong (same channels as Opus's list). Q4 cut 29, 23, 17, 16, 12, 24 (3 of 6 match), outside
28, 19, 3, 14, 18, 22 (2 match): half. Total ~1 of 4 (Opus ~1.5).

E3 fix ($7.01, 31 steps, 10 of 10 passes): same diagnosis as Opus, independently (no unit combines depth with the soil
planes; prior + residual passes the full-path cut). Built a gated stall branch in one GroupNorm group per layer using
antisymmetric (s, -s) pairs so the group mean is 0 and GroupNorm becomes a fixed gain; gate cancels the prior past the
earthmoving-equation stall depth, recalibrated from the clay bite (F_s ~36 kN/m). Probes confirmed the circuit computed
as designed. A disconnected-path control showed the takeover alone raised completed-bite error 0.108 -> 0.18-0.19 m;
three variants ended at 0.19-0.31 m overall vs 0.118. Reverted; judge unchanged (0.0981), not accepted.
Both frontier models reached the same blocker: the network has no spare channels for a side path.

## Round 2 (2026-11-09): free `list_bites` probe; E3 on a function-preserving widened model (one spare GroupNorm group
per block, max |output diff| 9.5e-7 m vs w0); Anthropic max_tokens 64k (16k let Fable's thinking consume whole turns)

E1 read: Q3 now right for both (Opus 83, 51, 158, 228, 151; Fable 83, 158, 228, 151, 68: 3 of 5 in the key's stall-AUC /
ablation lists, found by comparing stalled vs completed bites of the same soil; Fable also named 243, the key's top
ablation channel, as the anti-stall channel). Q1 still wrong for both (ch 1 / ch 3). Q2: Opus encoder, 18 (half);
Fable decoder (wrong). Q4: half for both (channels 0-4 observed directly, rest inferred). Totals: Opus ~2 of 4 ($1.54),
Fable ~1.75 of 4 ($3.49). Pass needs 3.

E3 fix on the wide model, judged on validation:
- Opus ($5.12): built a 9-layer branch entirely in spare channels (d^2 force term with soil-shifted break points, width
  sum, stall threshold, one-way spread, prior cancel). Practice: completed 13.1 / stalled 16.4 cm vs 8.3 / 11.9 base at
  the calibrated threshold; best variant 11.4 / 15.7. Found that envelope depth does not separate stalls (a completed clay
  bite reached 0.53 m, a stalled one stopped at 0.34-0.44 m). Reverted; judge unchanged, not accepted.
- Fable ($14.62 rerun; first attempt $4.05 broken by the 16k cap): found the stall is fill-driven (planned volume ~4 m^3
  vs 0.95 m^3 capacity; deepest part cut first, bucket stops in the shallow tail). Built a global stall flag through the
  GroupNorm mean that separated stalled from completed practice bites perfectly (completed-bite error bit-identical), but
  the locator (where on the path to stop) failed: stalled error 0.133 -> 0.745 m. Reverted; not accepted.
Takeaways: the widening removed the channel-takeover cost (completed bites untouched in Fable's build); the remaining
problem is locating the stop point inside a local CNN. Both now point at global quantities (fill volume, path length).


## Answer-key correction, Q1 (2026-11-09, audit)
The original Q1 key (replace each input by its dataset mean) favoured binary channels and named ch 2. Three scale-fair
measures on 187 practice + validation bites all name ch 1 (swept envelope), positive sign: input x gradient 0.18 (next
0.08), swap-with-another-bite 0.21 (next 0.12), signed 1-std shift +0.059. Rescored E1 totals (pass = 3 of 4):
GLM-5 ~1.5 (ch 1 right, sign wrong), Kimi ~1, DeepSeek ~0.5; Opus r1 ~2.5, Fable r1 ~1; Opus r2 ~3 (PASS), Fable r2 ~1.75.

## Round 3 (2026-11-09): audit fixes, stop-point input, Opus 5.5 at budgets 25 and 50 (xhigh thinking)
Audit fixes before the run: stop-point calculator (train-fit: V_th 2.75 m^3 removable soil or F_th 43 kN/m; validation
90.7% stall accuracy, median stop error 1.0% of path vs 31% for "never stops") as input channel 7 of a function-preserving
8-input widened copy (max output diff 9.5e-7 m); CONTEXT rewritten from verified facts (drag is -x on 69/79 practice
bites, prior caps volume by uniform scaling, w0 practice numbers) plus an auto-generated description of the exact copy;
probes compacted to fit the 12k tool limit (list_bites 11.8k -> 5.0k chars), new path_profile and compare_groups, zone
errors (deep_planned / shallow_planned / beyond_stop) in error_report; empty-answer runs flagged BROKEN; controls made
input-aware; E1 Q1 key corrected (see above).
Key audit finding: the stall premise in all earlier E3 prompts was wrong. On stalled bites the calculator flags, only
4-5% of squared error is past the predicted stop (the teacher still cuts 0.2-0.3 m there); 60-67% is in the dug zone and
28-37% outside the path. The worst cells are planned >0.6 m deep: RMS 27-30 cm with ~0 mean bias, on stalled AND completed
bites (a shape/placement error). The E3 task was rewritten to state this.

- Opus b25 ($5.70, 21/25 passes): traced the deep cut to d1 ch 17 (out weight -0.129, activations 5-10 -> 0.6-1.1 m);
  two spare-channel limiters on it were worse (all 11.92 / 11.83 vs 11.78 cm practice). Submitted original.
- Opus b50 ($5.35, 27/50 passes): diagnosis: the network has no input for drag direction; +x bites (13%) need the
  opposite end deep, so the net averages two groups (large RMS, ~0 bias). Wiring ch 7 into e1.0 made it much worse
  (stalled 18.3 vs 13.4). Submitted original. Neither session ran out of budget.
- Check of the b50 diagnosis by retraining (same recipe, train -> validation, 2 seeds): adding a "path progress" input
  (first-touch time fraction, encodes direction and order) did not help: all 10.27 / 10.54 cm vs 10.20 / 10.38 without;
  +x bites 11.7 / 14.1 vs 12.5 / 13.8 (seed noise is as large as any difference). So the remaining error is not fixed by
  giving the network that information either; the measured lever so far is more data (learning curve 13.0 -> 10.9 -> ~9 cm
  at 48 / 72 / 246 scoops).
