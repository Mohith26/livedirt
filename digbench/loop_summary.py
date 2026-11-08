"""Summarise closed-loop gate runs (app/bench/autonomy.ts JSON) into one markdown table.

  python -m digbench.loop_summary app/bench/results/autonomy-sandy-gravel-slope1*.json > digbench/results/loop-gate.md
"""
import json
import sys
from pathlib import Path


def row(p):
    d = json.loads(Path(p).read_text())
    f, v, n = d["final"], d.get("volume", {}), d.get("neural") or {}
    tag = Path(p).stem.replace("autonomy-sandy-gravel-slope1", "") or "-"
    model = ",".join(n["model"]) if isinstance(n.get("model"), list) else (n.get("model") or "heightfield (cut+relax)")
    share = n.get("networkShare")
    return (f"| `{tag.strip('-') or 'baseline'}` | {model} | {f['onGradePct']:.1f} | {f['meanAbsErrorCm']:.2f} | {f['overDigM3']:.3f} | "
            f"{f['unplannedCutM3']:.3f} | {v.get('deltaM3', float('nan')):+.2f} | {n.get('clippedCells', 0)} | {n.get('flooredCells', 0)} | "
            f"{n.get('peakDragKN', '-')} | {'-' if share is None else f'{share:.0%}'} | {d.get('verified')} |")


def main():
    files = sys.argv[1:]
    print("| run | model | on grade % | MAE cm | over-dig m^3 | unplanned cut m^3 | volume delta m^3 | clipped | floored | peak drag kN | network share | verified |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in files:
        try:
            print(row(p))
        except Exception as e:  # keep the table going if one file is malformed
            print(f"| `{Path(p).name}` | error: {e} |")


if __name__ == "__main__":
    main()
