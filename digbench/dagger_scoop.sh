#!/bin/bash
# DAgger for the per-scoop model: dig random scenarios with the LEARNED soil in the loop (states the model itself
# visits, e.g. its own over-dug pits), record every bite as a trace, then have the MPM teacher label each bite from that
# exact state (mpm_server --save-dir). Scenarios are drawn like scoop_campaign.sh with fresh seeds; the evaluation
# scenario (v4 gate) is never used, so the gate stays held out.
#   bash digbench/dagger_scoop.sh <learned-port> <mpm-port> <first-seed> <n> <tag>
LP=$1; MP=$2; SEED0=$3; N=$4; TAG=$5
export PATH=$HOME/node/bin:$PATH
OUT=$HOME/v4/dagger/$TAG; mkdir -p "$OUT"
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
  run="${TAG}-${seed}-${soil}"
  echo "=== learned run $run $(date -u +%T)"
  npx tsx bench/autonomy.ts "$soil" "$slope" --controller adaptive --soil-backend mpm --mpm-url "http://127.0.0.1:$LP" \
    --mpm-req "{\"soil\":{\"preset\":\"$soil\"},\"margin\":1.0}" --depth "$depth" --bottom "$bw" --length "$len" --start "$start" \
    --yaw-offset "$yaw" --tag "dagger-$run" --trace "$OUT/$run.json" > "$OUT/$run.log" 2>&1
  echo "=== MPM labels for $run $(date -u +%T)"
  python3 - "$OUT/$run.json" "$MP" "$soil" "$run" <<'EOF'
import json, sys, urllib.request
traces = json.load(open(sys.argv[1]))["traces"]
for i, tr in enumerate(traces):
    body = json.dumps({"trace": tr, "soil": {"preset": sys.argv[3]}, "margin": 1.0, "seed": i, "run": sys.argv[4]}).encode()
    rq = urllib.request.Request(f"http://127.0.0.1:{sys.argv[2]}/bite", data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(rq, timeout=3600).read())
        print(f"label {i + 1}/{len(traces)} carried {r['carried']:.3f} completion {r['completion']:.2f}", flush=True)
    except Exception as e:
        print("label failed", i, e, flush=True)
EOF
done
echo "=== dagger done $(date -u +%T)"
