#!/usr/bin/env bash
# Rebuild the LiveDirt scene products from the public source capture.
#   1. download the SOG-compressed 3DGS scene (CC BY 4.0, "Muddy Excavator Site" by dok11 on SuperSplat)
#   2. decode to PLY (SH0) with @playcanvas/splat-transform
#   3. level (RANSAC ground plane -> Y up), scale to metres, crop, strip sub-surface noise   (js/build-scene.mjs)
#   4. extract terrain products: heightfield, obstacle mask, soil palette, albedo         (js/terrain.mjs)
#   5. split into inner dig zone (r < 13 m, full detail) + outer ring (adaptive decimation), re-encode as SOG
# Requires node >= 20. Outputs to app/public/scene/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/.work}"
OUT="$HERE/../app/public/scene"
mkdir -p "$WORK/sog" "$OUT"
BASE=https://d28zzqy0iyovbz.cloudfront.net/ded12920/v1
for f in meta.json means_l.webp means_u.webp quats.webp scales.webp sh0.webp shN_centroids.webp shN_labels.webp; do
  [ -f "$WORK/sog/$f" ] || curl -sfL -o "$WORK/sog/$f" "$BASE/$f"
done
cd "$WORK"
npx -y @playcanvas/splat-transform@3 -q -w sog/meta.json -H 0 full.ply
node "$HERE/js/build-scene.mjs" full.ply "$OUT" "$WORK"
node "$HERE/js/split.mjs" clean.ply 13 inner.ply outer.ply
npx -y @playcanvas/splat-transform@3 -q -w -g cpu inner.ply "$OUT/inner.sog"
npx -y @playcanvas/splat-transform@3 -q -w -g cpu outer.ply --decimate-adaptive 55% outer_d.ply
npx -y @playcanvas/splat-transform@3 -q -w -g cpu outer_d.ply "$OUT/outer.sog"
echo "scene written to $OUT (machine.json pose is hand-measured: see js/view-raised.mjs and docs/ARCHITECTURE.md)"
