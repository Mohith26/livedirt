# M2b: train our own splat from the raw source video

Goal: replace the downloaded reconstruction with one we trained ourselves, and report held-out image quality
(PSNR / SSIM / LPIPS) with the standard 3DGS protocol (every 8th frame held out).

## Status
Partial. Footage, frames, environment, trainer and the evaluation script are in place. SfM was mid-run
(sequential matching) when this worker handed off. **No metrics have been measured yet**; `results.json`
records only what actually ran.

## Run (resumable, each stage skips if its output exists)
```bash
pipeline/train/run_all.sh            # STEPS=7000 by default; WORK defaults to pipeline/.work/train
```
Stages:
1. Download Pexels video 29568320 (David Pickup, Pexels license), 1920x1080 30 fps, 33.9 s, 133 MB.
2. Extract 200 evenly spaced frames at 1600 px wide with the static ffmpeg 7.1 shipped by `imageio-ffmpeg`
   (no system ffmpeg here).
3. `sfm.py`: pycolmap 3.13 on ALL frames (train and test registered together so test poses exist):
   SIFT (max 8192 features, max size 1600, one shared SIMPLE_RADIAL camera), sequential matching
   (overlap 12, no loop detection), incremental mapping; writes `sparse/best` + `sfm_result.json`.
4. Lay out a COLMAP dataset (`ds/images`, `ds/sparse/0`).
5. Train with Brush v0.3.0 (Metal/wgpu, native arm64 binary) for 7000 steps, `--eval-split-every 8`,
   saving eval renders to disk and exporting the PLY.
6. `evaluate.py`: recompute PSNR, SSIM (scikit-image) and LPIPS (AlexNet, `lpips` package, CPU torch)
   from the saved render/GT pairs, independent of Brush's own logging; writes `eval_metrics.json` and
   side-by-side images into `heldout/`.

## Environment notes (this machine)
- Hardware: Apple Silicon arm64, no CUDA. Torch MPS reported unavailable in this sandbox; LPIPS runs on CPU.
- The sandbox's bundled `python3` cannot create venvs (`ensurepip` fails); use `/usr/bin/python3` (3.9.6).
- msplat is not on PyPI (`pip download msplat`: no matching distribution); Brush was used instead.
- gsplat on CPU was not attempted (too slow).

## Files
- `run_all.sh`: end-to-end pipeline. `sfm.py`: SfM. `evaluate.py`: held-out metrics + side-by-sides.
- `results.json`: measured facts only (null where not yet run).

## Known issue
Importing `pycolmap` and `torch` in the same Python process aborts with OMP Error #15 (two libomp copies).
`sfm.py` and `evaluate.py` run as separate processes, so the pipeline is unaffected; don't merge them.
