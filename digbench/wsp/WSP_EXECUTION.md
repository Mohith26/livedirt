# Weight-space programming on the scoop model: execution plan

Source: Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming" (MIT CSAIL, Nov 2 2026,
rikiyat.github.io/blog/weight-space-programming). Harness: `weightspace` 0.3.0 (installed on the AWS box).
Everything in "Ready" was run on 2026-11-09; everything in "Experiments" is ready to launch but has not been run.

## 1. Their approach, in parts

| # | Part | What they did | Lesson for us |
|---|---|---|---|
| A | Read, no tools | Weights as text (17k-24k weights); the LLM names the game / shape from first- and last-layer structure | Our net has 2.0M weights: too big for text mode |
| B | Read with a Python sandbox, no forward passes | 7.2M-weight transformer middle layers; found every copying head by multiplying weight matrices | Use the sandbox on our encoder/decoder; score answers against activation measurements the LLM never sees |
| C | Read with a restricted probe | Weights alone could not reveal the sycophant; attention patterns only (no outputs) could | Give a feature-map-only probe; use it when weights alone fail |
| D | Fix from one recorded failure | Car-on-ramp policy, 0% -> 100% success; added a small separate path through a few units per layer that detects the situation and switches behaviour, 3.5% of weights | Our stall failure is the same shape: a situation the net handles wrong, fixable with a gated side path |
| E | Write from scratch | MinAtar and Go nets programmed layer by layer from the rules ("first layer computes where the ball lands") | Our physics (stall point, settle rule) is a short program; write it as weights |
| F | Edit with tools and a forward-pass budget | Python kernel over weights, practice problems, 10/25/50 forward passes; edits interleaved; keep a change only if it wins | Same loop, on a practice tier; budget sweep 10/25/50 |
| G | Remove a failure direction | Qwen looping traced to one MLP direction in 2 layers; subtracted from 64 columns (0.016%): MATH-500 7.9% -> 48.1% | Look for directions that carry our systematic errors (volume bias, pits) |
| H | Limits | Gemma (no loops) could not be improved: edits remove obstacles, do not add abilities; weights alone missed the sycophant | Use edits for gating / bias failures; missing knowledge still needs data |
| I | Replication lessons (independent harness) | Opus 5.5 edit 8.4% -> 40.0% but below the paper's 52.4%; a strong simple baseline (thinking off) beat the edits; practice-set gains overstated test gains (50.8% vs 30.5%) | Every edit must beat a gradient control on the same tensors and a random control, and is scored on a tier the editor never saw |
| J | Ideas they list | Auditing without evaluation, one-shot initialisation, zero-data steering, circuit grafting across models | Soil audit from weights; graft the stall circuit into the browser model |

## 2. Ready (built and checked 2026-11-09)

Code: `digbench/wsp/` (commits e0b1e61, and the splits fix after it).
* `splits.py`: four tiers by whole trench run. Current: train 197 bites (8 runs), practice 79 (4), validation 108 (5),
  gate 115 (4 runs: both gate trenches). File `~/v4/wsp/splits.json`.
* Base model `~/v4/wsp/w0.pt` (trained on the train tier only). Validation RMS 8.5 cm (prior 15.1 cm, zero 17.9 cm);
  practice tier: all 11.8 cm, cut cells 20.9, slump zone 12.9, far 7.6, stalled 13.4, completed 10.8, carried error
  0.079 m^3.
* `probes.py`: the only ways the editor can run the net (each costs one forward pass, practice tier only):
  `error_report`, `worst_bites`, `error_map`, `feature_maps` (activations only, no outputs: part C).
* `judge.py`: scores a candidate on validation vs the base; accept only if the overall error falls by more than 2 mm
  and no region or stall class gets worse by more than 2 mm. Checked: base vs itself -> identical, rejected; a
  hand-made 20% output shrink -> 8.5 -> 11.0 cm, rejected.
* `controls.py`: gradient control (Adam on the same tensors, practice tier, same number of passes) and random control
  (same tensors, same edit norm). Checked on the test edit (random -> 11.9 cm, gradient -> 10.6 cm).
* `run_session.py`: one weightspace session (read / fix / write / improve) with the context file and a task prompt,
  saves transcript, sparse patch, patched weights, cost.
* `modules.py`: `StopNet`, the zero-initialised target for the write experiment.
* `tasks/`: `CONTEXT.md` (architecture, channels, physics, measured errors, rules) and prompts E1-E6.

## 3. Experiments (in order)

Common command shape (on the box, `export PYTHONPATH=~/livedirt`):
```
python -m digbench.wsp.run_session --task digbench/wsp/tasks/<E>.md --mode <mode> --base ~/v4/wsp/w0.pt \
    --llm <provider:model> --budget <N> --max-cost <usd> --out ~/v4/wsp/runs/<E>-<try>
python -m digbench.wsp.controls --base ~/v4/wsp/w0.pt --edited ~/v4/wsp/runs/<E>-<try>/patched.pt --passes <N> --out ~/v4/wsp/runs/<E>-<try>/controls
for m in patched controls/gradient controls/random; do python -m digbench.wsp.judge --base ~/v4/wsp/w0.pt --candidate ~/v4/wsp/runs/<E>-<try>/$m.pt --out ~/v4/wsp/runs/<E>-<try>/judge-$(basename $m).json; done
```

| Exp | Paper part | Mode, budget | Question | Pass condition | Then |
|---|---|---|---|---|---|
| E1 read | A, B, C | read, 0 then 10 feature-map passes | Can the editor read OUR CNN: completion-driving channel, soil-carrying filters, stall channels in e4, cut vs outside writers in d1 | >= 3 of 4 answers match activation patching (run after, editor never sees it) | If < 2 of 4: skip pure reading, rely on probes in E3/E4 |
| E2 write | E | write StopNet from zero; 0, then 10 passes | Program the stall point from depth + soil | Validation: stall classification >= 91%, median stop error <= 3% of path (our hand-coded rule) | The module becomes an input channel for E3, or the graft source for E6 |
| E3 fix stalls | D, F | fix, budgets 10 / 25 / 50 | Gated side path that stops the predicted change where the bucket stalls | Judge accepts; stalled error falls >= 15%; beats gradient and random controls; reproduces in a second session | Biggest expected gain (stalls = 58% of error) |
| E4 directions | G, F | improve, budgets 10 / 25 | Find and remove directions carrying volume bias, near-end pits, under-predicted slumps | Judge accepts and beats controls; per-error claim verified on validation | Stack with E3 if both pass |
| E5 audit | C, J | read two fine-tunes (sandy gravel vs wet clay), weights then feature maps | Can soil be identified from weights alone | Correct in >= 2 of 2 pairs with layer explanation | Use as a pre-deployment check |
| E6 graft | J | fix on the merged / browser model | Move the E2/E3 stall circuit into another network without retraining | Stalled error falls in the target as in the source, completed unchanged | Only after E2 or E3 pass |
| E7 closed loop | all | eval only | Does the accepted stack move the full-trench result | Gate 1 and gate 2 final-surface error vs the MPM teacher below a7's 13 / 16 cm, both trenches | Report; no tuning on the gate |

Each accepted edit is replicated in a second independent session (the replication found 1 of 3 repairs helping). A
budget sweep (10 / 25 / 50 passes) is run for E3 because the paper's gain appeared only at 50.

## 4. What "better" must mean (fixed now, before any run)

* Validation is the score; practice numbers are reported but never used to accept (practice gains overstated test gains
  in the replication).
* An edit counts as weight-space programming only if it beats both controls on validation. If the gradient control on
  the same tensors does as well, the result is "edits are as good as fine-tuning these tensors", reported as such.
* The gate trenches are run once, at E7, on the final accepted stack.

## 5. Budget and stop rules

* Editor sessions: `--max-cost` per session (default $25). Plan ceiling: E1 2 sessions, E2 2, E3 6 (3 budgets x 2),
  E4 2, E5 2, E6 1 = 15 sessions, at most $375, likely $100-200. LLM API spend is separate from AWS credits.
* AWS: all experiments run on CPU except E7 (CPU too: learned backends). Box-hours ~10-20.
* Stop rule: if E1 shows no reading ability and E3 at budget 25 is not accepted in 2 sessions, stop the editing track
  and keep the compiled-physics track (E2 hand-coded, already 91%).

## 6. Decisions needed before launch

1. Editing model and key: e.g. `anthropic:claude-opus-5-5` or `openai:gpt-6-astra` (the paper's model), key set on
   the box through a hidden prompt (never pasted in chat).
2. API spend ceiling (suggested $200 for the full plan).
3. Alternative with no API spend: the Aside agent drives the same tools by hand (weightspace's `WeightSpace` API for
   reads, edits, checkpoints and patches; the same probes and judge). Same protocol, one editor, slower.
