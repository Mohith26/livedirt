#!/bin/bash
# Per-scoop data campaign: random trench scenarios dug by the adaptive controller with MPM as the soil; the MPM server
# (--save-dir) records every bite. One worker = one server port; run several workers against several servers.
#   bash digbench/scoop_campaign.sh <port> <first-seed> <n-runs> [save-root]
# Scenario ranges (all from the app's own presets and plausible trench specs):
#   soil preset: wet-clay | common-earth | sandy-gravel | saturated-silt
#   side slope (H:V): the preset's stable range (1 .. 2), depth 0.6-1.4 m, bottom width 0.6-1.2 m, length 2.5-4.5 m,
#   start 3.6-5.0 m from the swing axis, trench heading -40..40 deg from the house yaw (different ground on the capture)
PORT=${1:-8766}; SEED0=${2:-100}; N=${3:-10}; ROOT=${4:-$HOME/v4/scoops}
export PATH=$HOME/node/bin:$PATH
cd "$HOME/livedirt/app" || exit 1
for ((k = 0; k < N; k++)); do
  seed=$((SEED0 + k))
  read soil slope depth bw len start yaw < <(python3 - "$seed" <<'EOF'
import random, sys
r = random.Random(int(sys.argv[1]))
soil = r.choice(["wet-clay", "common-earth", "sandy-gravel", "saturated-silt"])
slope = {"wet-clay": r.choice([0.75, 1.0]), "common-earth": r.choice([1.0, 1.5]), "sandy-gravel": r.choice([1.5, 2.0]), "saturated-silt": 2.0}[soil]
print(soil, slope, round(r.uniform(0.6, 1.4), 2), round(r.uniform(0.6, 1.2), 2), round(r.uniform(2.5, 4.5), 2), round(r.uniform(3.6, 5.0), 2), round(r.uniform(-40, 40), 1))
EOF
)
  run="c${seed}-${soil}"
  echo "=== run $run slope $slope depth $depth bw $bw len $len start $start yaw $yaw $(date -u +%T)"
  npx tsx bench/autonomy.ts "$soil" "$slope" --controller adaptive --soil-backend mpm --mpm-url "http://127.0.0.1:$PORT" \
    --mpm-req "{\"soil\":{\"preset\":\"$soil\"},\"margin\":1.0,\"seed\":$seed,\"run\":\"$run\"}" \
    --depth "$depth" --bottom "$bw" --length "$len" --start "$start" --yaw-offset "$yaw" --tag "camp-$run" 2>&1 | grep -E "^\[mpm\]|wrote|Error" | tail -40
done
echo "=== campaign done $(date -u +%T)"
