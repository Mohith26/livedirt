#!/usr/bin/env bash
# LiveDirt M2b: train our own 3DGS of the muddy excavator site from the raw source video and evaluate held-out quality.
# Resumable: each stage is skipped if its output already exists. Apple Silicon arm64 (macOS); no CUDA.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${WORK:-$HERE/../.work/train}"; mkdir -p "$W"; cd "$W"
STEPS="${STEPS:-7000}"
# 0. env: system python (the sandbox's bundled python cannot create venvs)
[ -x venv/bin/python ] || { /usr/bin/python3 -m venv venv && venv/bin/pip install -q --upgrade pip && venv/bin/pip install -q imageio-ffmpeg pycolmap numpy pillow scikit-image torch torchvision lpips; }
FF=$(venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
# 1. source footage: Pexels video 29568320 by David Pickup (Pexels license), 1920x1080 30fps, 33.9 s
[ -f src_1920.mp4 ] || curl -sfL -A "Mozilla/5.0" -o src_1920.mp4 https://videos.pexels.com/video-files/29568320/12727230_1920_1080_30fps.mp4
# 2. 200 evenly spaced frames at 1600 px wide
if [ ! -d frames ]; then mkdir frames; "$FF" -loglevel error -i src_1920.mp4 -vf "fps=200/33.9,scale=1600:-2" -q:v 2 frames/f_%04d.jpg; fi
# 3. SfM over all frames (train + test registered jointly so held-out poses exist)
[ -f sparse/best/images.bin ] || venv/bin/python -u "$HERE/sfm.py" .
# 4. COLMAP dataset layout for Brush: <ds>/images + <ds>/sparse/0
mkdir -p ds/sparse; [ -e ds/images ] || ln -s ../frames ds/images; [ -e ds/sparse/0 ] || ln -s ../../sparse/best ds/sparse/0
# 5. train on 7/8 of frames, every 8th frame held out (standard 3DGS protocol), evaluate + save renders
[ -x brush-app-aarch64-apple-darwin/brush_app ] || { curl -sfL -o brush.tar.xz https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-aarch64-apple-darwin.tar.xz && tar xf brush.tar.xz; }
mkdir -p out
/usr/bin/time -p brush-app-aarch64-apple-darwin/brush_app ds --total-steps "$STEPS" --eval-split-every 8 --eval-every "$STEPS" --eval-save-to-disk \
  --max-resolution 1600 --export-every "$STEPS" --export-path out --export-name "muddy_site_{iter}.ply" 2>&1 | tee train.log
# 6. independent metrics from the saved eval image pairs
venv/bin/python "$HERE/evaluate.py" out "$HERE/eval_metrics.json" "$HERE/heldout"
