"""Score every bite of a learned-model trace by ensemble disagreement and enqueue it for MPM labelling.
  python -m digbench.al_loop.score_trace <trace.json> <soil-preset> <run> <models comma list>"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

from digbench.scoop_model import make_inputs
from digbench.soil_presets import mc_params
from digbench.train_scoop import UNet, pad8

Q = Path("~/v4/loop/queue").expanduser(); Q.mkdir(parents=True, exist_ok=True)


def main():
    tf, preset, run, models = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
    traces = json.load(open(tf))["traces"]; M = 1.0
    soil = {**mc_params(preset)}; soil["mu"] = float(soil.get("mu", np.tan(np.radians(soil.get("phi", 30.0)))))
    nets = []
    for m in models:
        n = UNet(7, 32); n.load_state_dict(torch.load(m, map_location="cpu")); n.eval(); nets.append(n)
    for i, tr in enumerate(traces):
        try:
            terr = np.array(tr["terrain"], float).reshape(tr["nx"], tr["ny"]); cut = int(round((tr["margin"] - M) / 0.05))
            if cut > 0:
                terr = terr[cut:-cut, cut:-cut]
            X, prior, _ = make_inputs(terr, tr["poses"], soil, M)
            Xp, (nx, ny) = pad8(X[None])
            with torch.no_grad():
                R = np.stack([n(torch.tensor(Xp))[0][0, :nx, :ny].numpy() for n in nets])
            mean = prior + R.mean(0); msk = np.abs(mean) > 0.01
            score = float(R.std(0)[msk].mean()) if msk.any() else 0.0
        except Exception as e:  # keep the bite, unscored, at the bottom
            print("score failed", i, e, flush=True); score = 0.0
        item = {"trace": tr, "soil": {"preset": preset}, "margin": 1.0, "seed": i, "run": run, "score": score, "t": time.time()}
        name = f"{int(min(score, 9.999999) * 1e6):07d}_{run}_{i:03d}.json"
        tmp = Q / ("." + name); tmp.write_text(json.dumps(item)); os.replace(tmp, Q / name)
        print(f"queued {name}", flush=True)


if __name__ == "__main__":
    main()
