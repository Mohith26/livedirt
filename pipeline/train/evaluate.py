"""Independent held-out evaluation of Brush eval renders against the ground-truth frames.
Brush (--eval-split-every 8 --eval-save-to-disk) writes rendered + GT images per eval step. This script
recomputes PSNR / SSIM (scikit-image) and LPIPS (lpips, AlexNet backbone, as in the 3DGS paper's reporting)
from those image pairs and writes a JSON summary plus side-by-side comparison images.
Usage: python evaluate.py <eval_dir> <out_json> <sbs_dir> [gt_frames_dir]
Brush v0.3.0 saves only the renders (out/eval_<step>/<frame>.png); ground truth is then the source frame of the same name."""
import sys, json, pathlib, numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
eval_dir, out_json, sbs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
sbs.mkdir(parents=True, exist_ok=True)
pairs = []
for r in sorted(eval_dir.rglob("*")):
    if not r.is_file() or r.suffix.lower() not in (".png", ".jpg"): continue
    n = r.name.lower()
    if "render" in n or "pred" in n:
        cands = [r.with_name(r.name.replace(k, v)) for k, v in (("render", "gt"), ("Render", "GT"), ("pred", "gt"), ("render", "ground_truth"))]
        g = next((c for c in cands if c.exists()), None)
        if g: pairs.append((r, g))
gt_dir = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else None
if not pairs and gt_dir is not None:
    steps = sorted([d for d in eval_dir.glob("eval_*") if d.is_dir()], key=lambda d: int(d.name.split("_")[1]))
    src = steps[-1] if steps else eval_dir
    for r in sorted(src.glob("*.png")):
        g = next((c for c in (gt_dir / (r.stem + ext) for ext in (".jpg", ".png", ".jpeg")) if c.exists()), None)
        if g: pairs.append((r, g))
if not pairs: sys.exit(f"no render/gt pairs found under {eval_dir}")
try:
    import torch, lpips
    lp = lpips.LPIPS(net="alex", verbose=False)
except Exception as e:  # pragma: no cover
    lp = None; print("LPIPS unavailable:", e)
rows = []
for i, (r, g) in enumerate(pairs):
    a = np.asarray(Image.open(r).convert("RGB")).astype(np.float32) / 255
    b = np.asarray(Image.open(g).convert("RGB")).astype(np.float32) / 255
    if a.shape != b.shape: a = np.asarray(Image.open(r).convert("RGB").resize(b.shape[1::-1])).astype(np.float32) / 255
    row = {"render": str(r), "gt": str(g), "psnr": float(psnr(b, a, data_range=1)), "ssim": float(ssim(b, a, channel_axis=2, data_range=1))}
    if lp is not None:
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None] * 2 - 1
        with torch.no_grad(): row["lpips_alex"] = float(lp(t(a), t(b)).item())
    rows.append(row)
    if i < 6:
        w, h = b.shape[1], b.shape[0]; im = Image.new("RGB", (w * 2 + 8, h), (20, 20, 20))
        im.paste(Image.open(g).convert("RGB"), (0, 0)); im.paste(Image.open(r).convert("RGB").resize((w, h)), (w + 8, 0))
        im.save(sbs / f"heldout_{i:02d}_real_vs_render.jpg", quality=88)
mean = lambda k: float(np.mean([x[k] for x in rows])) if rows and k in rows[0] else None
res = {"n_pairs": len(rows), "psnr": mean("psnr"), "ssim": mean("ssim"), "lpips_alex": mean("lpips_alex"), "per_image": rows}
json.dump(res, open(out_json, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_image"}, indent=1))
