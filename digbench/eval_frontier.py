"""Accuracy vs speed on the task-3 dig cycle: MPM (reference), LiveDirt cut+relax, learned surrogate.

Truth = the 3D MPM standard episode at dx 4 cm (the surrogate's training resolution), terrain sampled every 0.1 s.
  python -m digbench.eval_frontier --episode /content/std/standard_task3.npz --model digbench/results/surrogate-v1.pt \
      --mpm-wall <seconds for the MPM episode> --out digbench/results
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .train_surrogate import build_model, features, livedirt_rollout, unmirror


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bucket-json", default="digbench/results/bucket-v2-dx4cm.json", help="MPM wall time source")
    ap.add_argument("--out", default="digbench/results")
    args = ap.parse_args()
    d = np.load(args.episode)
    ep = {"H": d["H"].astype(np.float32), "T": d["T"].astype(np.float32), "F": d["F"], "mu": float(d["mu"])}
    n = len(ep["H"]); sim_s = (n - 1) * 0.1
    truth = ep["H"]
    dist = np.abs(truth[-1] - truth[0]) > 0.02
    rows = {}

    def score(name, roll, wall):
        rows[name] = {"final_rmse_disturbed_m": float(np.sqrt(((roll[-1] - truth[-1]) ** 2)[dist].mean())),
                      "rollout_rmse_m": float(np.sqrt(((roll - truth) ** 2).mean())),
                      "wall_s": wall, "realtime_factor": sim_s / wall}

    mpm = json.loads(Path(args.bucket_json).read_text())["summary"] if Path(args.bucket_json).exists() else None
    if mpm:
        rows["mpm_dx4cm"] = {"final_rmse_disturbed_m": 0.0, "rollout_rmse_m": 0.0, "wall_s": mpm["wall_s"], "realtime_factor": mpm["rtf"]}
    t0 = time.perf_counter(); lv = livedirt_rollout(ep); score("livedirt_cut_relax_python", lv, time.perf_counter() - t0)
    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        model = build_model(torch); model.load_state_dict(torch.load(args.model, map_location=dev)); model.to(dev).eval()
        tens = lambda a: torch.tensor(a, device=dev)
        with torch.no_grad():
            for warm in (True, False):
                H = tens(ep["H"][0][None]); mu = tens(np.array([ep["mu"]], np.float32)); pred = [ep["H"][0]]; forces = []
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                for t in range(n - 1):
                    dh, f = model(features(H, tens(ep["T"][t][None]), tens(ep["T"][t + 1][None]), mu, torch))
                    H = H + unmirror(dh[:, 0], H.shape[-1])
                    pred.append(H[0].cpu().numpy()); forces.append(f[0].cpu().numpy())
                if dev == "cuda": torch.cuda.synchronize()
                wall = time.perf_counter() - t0
        score(f"surrogate_{dev}", np.array(pred), wall)
        Fx = ep["F"][1:, 0]; drag = np.abs(Fx) > 1e3
        rows[f"surrogate_{dev}"]["drag_force_rel_err"] = float(np.abs(np.array(forces)[drag, 0] - Fx[drag]).mean() / np.abs(Fx[drag]).mean())
    out = {"task": "frontier_task3", "episode": args.episode, "sim_seconds": sim_s, "rows": rows}
    Path(args.out, "frontier-task3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
