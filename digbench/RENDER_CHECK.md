# DigBench item 7: rendering half (does LiveDirt deformation damage splat regions it should not touch?)

**Status: GPU render job QUEUED, not yet run.** No PSNR/SSIM/LPIPS numbers exist yet, so none are claimed here. The job is the last line of `digaws:~/jobs.txt` (behind about 10 data jobs); it writes `~/render_check/out/render-check.json` and a log `~/render_check/run.log`.
Fetch + commit when finished: `rsync digaws:~/render_check/out/render-check.json digbench/results/render-check.json`.

## Method (digbench/render_check.py, three stages)
1. `export`: COLMAP sparse model (pycolmap, 200 registered frames, 82,411 points, SIMPLE_RADIAL f=1174 px, 1600x900) + the 25 held-out frames (every 8th, f_0001, f_0009, ... = Brush `--eval-split-every 8`, same set as `pipeline/train/evaluate.py`). Splat = Brush 7000-step PLY (688,514 splats, SH3), found in the git-ignored `pipeline/.work/train/out/`.
2. `prep`: RANSAC ground plane through sparse points; trench placed where it is clear of sparse points standing over it and fully in frame in the most held-out views; the dig itself is produced by the REAL `app/src/sim/heightfield.ts` (run under node): grade-controlled passes along a 3 m x 1 m x 0.6 m design trench, wet-clay soil, then `settle()`. Result: 1.80 m3 cut (= 3x1x0.6), min dH = -0.60 m, 0.21 m3 of wall slough, 32% of cells on grade at tol 3 cm.
3. `render` (gsplat, GPU): render original and deformed splats from every held-out camera in which the trench volume is in front of the camera and covers >=0.05% of the frame; score the UNCHANGED region (outside the projected trench volume dilated 24 px, and again 64 px) for original-vs-GT and deformed-vs-GT; also max abs pixel difference between the two renders there; changed region reported separately.

Deformation is a port of `app/src/render/deform.ts` L78-L101 (bind window 0.12-0.45 m from terrainSystem.ts L62, `center.y += bind*dH` L87, opacity hand-off `1 - bind*smoothstep(0.05,0.16,changed)` L100-101). Not ported: colour re-tint / wetness / overlays / global grade (L88-99, 102-111; palette is scene metadata we lack, grade is identity by default) and the CPU generated-soil layer that the opacity fade hands over to, so the changed region shows holes rather than what the app shows.

## Assumptions (all of them)
- **No metric scale exists.** Assumed 1 COLMAP unit = 5.615 m, chosen so the 90th-percentile radius of ground-plane sparse points (4.364 units) equals 24.5 m (siteRadius of the app's *other* scene in scene.config.json). Implies camera height ~13.1 m (range 10.5-14.4), plausible for aerial footage but unverified. Sensitivity runs at x0.5 and x2 scale (trench becomes 2.0x / 0.5x its relative size).
- Original terrain H0 = flat fitted plane. Sparse ground points near the trench deviate from it by 0.111 m rms (2,350 pts), so "ground" splats are not exactly on H0; this changes which splats get bound.
- Object code = 0 everywhere (no segmentation); the trench was chosen with no sparse points 0.15-2.5 m above its footprint (+0.7 m margin).
- Pinhole rendering; frame lens distortion (k1=0.0416) ignored, as it is not modelled by the splat training either (to be confirmed by the sanity check: job also reports full-frame PSNR vs GT and vs Brush's own eval render).
- Trench placed at ground-frame (-5.25, -0.25) m, azimuth 90 deg; fully in frame in 7/25 held-out views at search time (the render step also admits partially framed views).

## CPU-side facts already measured (not render metrics)
Splats actually modified by the ported deform (of 688,514): 10,709 (scale x0.5), 2,692 (x1), 697 (x2); max centre shift 0.60 m; opacity changed on 8,791 / 2,251 / 589.

## What blocks validating the dug region (true regardless of the numbers)
1. No post-dig footage: nothing photographs the trench, so changed-region PSNR/SSIM/LPIPS only measure departure from the pre-dig photo, never correctness.
2. A post-dig capture would need matched camera poses (same SfM frame, ideally re-registered into this COLMAP model) and a known dig geometry (surveyed trench or scan) to compare against.
3. No metric scale or surveyed ground truth for this site; trench size is a fraction-of-scene assumption.
4. Splat has no object/ground segmentation and no terrain heightmap, so H0 is a plane and `onObject` is unmodelled.
5. The app's colour re-tint palette and generated soil layer are not reproduced offline, so what a user sees in the dug region is not rendered here.
6. Held-out views see the trench small (about 13 m camera height); few views, no oblique views into the trench walls or floor.

## Results (run 2026-11-08 on the L40S; `results/render-check.json`)

cuDNN failed to load in the render venv (`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`); the run disables cuDNN
(`torch.backends.cudnn.enabled = False`), which only affects LPIPS speed.

* Renderer sanity: our gsplat renders of the 23 held-out frames match Brush's own evaluation renders at 58.9 dB PSNR
  (same splats, same cameras), and score 23.50 dB / SSIM 0.722 / LPIPS 0.330 against the photos (Brush reported 23.53 /
  0.723 / 0.330).
* Locality of the deformation (scale 1.0, 23 views): outside a 64-px dilation of the dig mask the deformed render equals
  the original to max |pixel diff| 0.006 (PSNR identical to 1e-6); outside a 24-px dilation, max diff 0.032 and 0.007% of
  pixels change by more than 1/255. Same picture at scale x0.5 and x2 (max diff 0.008 / 0.015 beyond 64 px). 2,692
  splats moved, 623 of them outside the trench footprint (the deform port's falloff), none above the bind height.
* Inside the dug region the render departs from the pre-dig photo (PSNR 25.3 -> 12.9 dB, LPIPS 0.21 -> 0.41 at scale 1.0).
  This measures change, not correctness: no post-dig photo or survey of this site exists, so whether the dug region
  LOOKS right is NOT validated.
