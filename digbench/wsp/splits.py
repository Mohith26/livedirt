"""Fixed data tiers for weight-space programming (WSP) on the per-scoop soil model.

  train      what the base model w0 is trained on (gradient training only)
  practice   what the editing LLM may run forward passes on (never trained on)
  validation what the judge scores candidate edits on (the LLM never sees it)
  gate       closed-loop trenches + their bites (gate1 sandy gravel, gate2 wet clay); used once per accepted edit stack
Assignment is by RUN (a whole trench), deterministic from the run name, so no tier shares a trench with another.
  python -m digbench.wsp.splits --out ~/v4/wsp/splits.json
"""
import argparse, glob, json, zlib
from pathlib import Path

DIRS = ["~/v4/scoops/campaign", "~/v4/scoops/loop-adaptive"]


def run_of(f):
    return Path(f).stem.rsplit("_", 1)[0]


def assign(run):
    if run.startswith("gate"):
        return "gate"
    h = zlib.crc32(run.encode()) % 100
    return "practice" if h < 15 else "validation" if h < 32 else "train"


def build():
    files = sorted(f for d in DIRS for f in glob.glob(str(Path(d).expanduser() / "*.npz")))
    out = {"train": [], "practice": [], "validation": [], "gate": []}
    for f in files:
        out[assign(run_of(f))].append(f)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    s = build()
    Path(a.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).expanduser().write_text(json.dumps(s, indent=1))
    for k, v in s.items():
        print(k, len(v), "bites,", len({run_of(f) for f in v}), "runs")
