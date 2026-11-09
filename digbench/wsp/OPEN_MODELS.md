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
