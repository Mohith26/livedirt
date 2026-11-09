#!/bin/bash
# Start the LiveDirt simulation bridge for Unreal (keep this terminal open), then open LiveDirtUE/LiveDirtUE.uproject.
cd "$(dirname "$0")/../app" || exit 1
[ -d node_modules ] || npm install
echo "control page: http://127.0.0.1:8790/"
exec npx tsx bench/unreal-bridge.ts --port 8790
