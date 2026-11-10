"""Re-rank the labelling queue with the newest ensemble (DAgger wants the current policy's uncertainty; old scores
from an older ensemble are stale). Items whose score cannot be recomputed keep theirs.
  python -m digbench.al_loop.rescore <models comma list>"""
import json, os, sys
from pathlib import Path

import numpy as np
import torch

from digbench.scoop_model import make_inputs
from digbench.soil_presets import mc_params
from digbench.train_scoop import UNet, pad8

Q = Path("~/v4/loop/queue").expanduser()


def main():
    nets = []
    for m in sys.argv[1].split(","):
        n = UNet(7, 32); n.load_state_dict(torch.load(m, map_location="cpu")); n.eval(); nets.append(n)
    soils = {}; done = 0
    for p in sorted(Q.glob("*.json")):
        try:
            it = json.loads(p.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        tr = it["trace"]; pre = it["soil"]["preset"]
        if pre not in soils:
            s = dict(mc_params(pre)); s["mu"] = float(s.get("mu", np.tan(np.radians(s.get("phi", 30.0))))); soils[pre] = s
        try:
            terr = np.array(tr["terrain"], float).reshape(tr["nx"], tr["ny"]); cut = int(round((tr["margin"] - 1.0) / 0.05))
            if cut > 0:
                terr = terr[cut:-cut, cut:-cut]
            X, prior, _ = make_inputs(terr, tr["poses"], soils[pre], 1.0); Xp, (nx, ny) = pad8(X[None])
            with torch.no_grad():
                R = np.stack([n(torch.tensor(Xp))[0][0, :nx, :ny].numpy() for n in nets])
            msk = np.abs(prior + R.mean(0)) > 0.01; score = float(R.std(0)[msk].mean()) if msk.any() else 0.0
        except Exception:
            continue
        it["score"] = score; name = f"{int(min(score, 9.999999) * 1e6):07d}_" + p.name.split("_", 1)[1]
        tmp = Q / ("." + name); tmp.write_text(json.dumps(it))
        try:
            os.remove(p)  # if a labeller claimed it meanwhile, drop the rescored copy
        except FileNotFoundError:
            os.remove(tmp); continue
        os.replace(tmp, Q / name); done += 1
    print(f"rescored {done} queued bites", flush=True)


if __name__ == "__main__":
    main()
