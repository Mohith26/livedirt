"""Pool-based active-learning simulation with labels we already have: does choosing which scoops to label by ensemble
disagreement beat random choice? Seed set = random half of the train tier; candidates = the other half. Add a quarter of
the tier either by disagreement of a 3-seed ensemble trained on the seed set, or at random (2 draws); retrain 3 seeds
each from scratch; score on validation (pooled RMS, single models and 3-model ensemble)."""
import json, subprocess, sys
from pathlib import Path

import numpy as np
import torch

from digbench.train_exp import SPLITS, infer, load, pooled
from digbench.train_scoop import UNet

O = Path("~/v4/wsp/al").expanduser(); O.mkdir(parents=True, exist_ok=True)
PY = sys.executable


def train(tag, names, seed):
    f = O / f"{tag}-names.json"; f.write_text(json.dumps(sorted(names)))
    if not (O / f"{tag}-s{seed}.json").exists():
        subprocess.run([PY, "-m", "digbench.train_exp", "--tag", f"{tag}-s{seed}", "--out", str(O), "--seed", str(seed),
                        "--subset-file", str(f)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads((O / f"{tag}-s{seed}.json").read_text())


def main():
    sp = json.loads(SPLITS.read_text()); pool = load(sp["train"], "cutrelax"); val = load(sp["validation"], "cutrelax")
    names = [s["name"] for s in pool]; rng = np.random.default_rng(7); perm = rng.permutation(len(names))
    seed_set = [names[i] for i in perm[: len(names) // 2]]; cand = [names[i] for i in perm[len(names) // 2:]]
    k = len(names) // 4; dev = "cuda" if torch.cuda.is_available() else "cpu"
    for sd in range(3):
        train("seed", seed_set, sd)
    nets = []
    for sd in range(3):
        n = UNet(7, 32).to(dev); n.load_state_dict(torch.load(O / f"seed-s{sd}.pt", map_location=dev)); n.eval(); nets.append(n)
    byname = {s["name"]: s for s in pool}; dis = {}
    for c in cand:
        s = byname[c]; ps = np.stack([infer(n, s, dev) for n in nets]); m = np.abs(ps.mean(0)) > 0.01
        dis[c] = float(ps.std(0)[m].mean()) if m.any() else 0.0
    picks = {"al": sorted(cand, key=lambda c: -dis[c])[:k],
             "rand1": list(np.random.default_rng(11).choice(cand, k, replace=False)),
             "rand2": list(np.random.default_rng(12).choice(cand, k, replace=False))}
    res = {"n_pool": len(names), "n_seed": len(seed_set), "n_added": k}
    for arm, add in picks.items():
        R = [train(arm, seed_set + add, sd) for sd in range(3)]
        preds = [np.load(O / f"{arm}-s{sd}-net-pred.npz") for sd in range(3)]
        ens = [np.mean([p[s["name"]].astype(np.float32) for p in preds], 0) for s in val]
        res[arm] = {"single_all": [r["net"]["all"] for r in R], "ensemble": pooled(ens, val),
                    "added_stalled": int(sum(byname[c]["completion"] < 0.9 for c in add))}
    R = [train("seedonly", seed_set, sd) for sd in range(3)]
    res["seed_only"] = {"single_all": [r["net"]["all"] for r in R]}
    (O / "al_result.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
