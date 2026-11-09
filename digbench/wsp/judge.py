"""The judge: score a candidate model (full .pt state dict, or base + weightspace patch .npz) on the VALIDATION tier
against the base model, and decide accept / reject.

Accept iff: overall RMS error improves by more than the noise margin (default 0.002 m, ~10% of the teacher's own
2.3 cm bite-to-bite spread), no region or stall class gets worse by more than the margin, and carried error does not
get worse by more than 0.01 m^3.
  python -m digbench.wsp.judge --base ~/v4/wsp/w0.pt --candidate runs/e3/patched.pt [--tier validation]
"""
import argparse, json
import numpy as np
import torch

from digbench.wsp.common import evaluate, load_model, tier


def load_candidate(base, cand):
    net = load_model(base)
    if cand.endswith(".npz"):  # weightspace sparse patch: {name: new full tensor} or {name+"__idx"/"__val"}
        z = np.load(cand); sd = net.state_dict()
        for k in z.files:
            if k.endswith("__idx"):
                name = k[:-5]; t = sd[name].view(-1); t[torch.tensor(z[k])] = torch.tensor(z[name + "__val"], dtype=t.dtype)
            elif not k.endswith("__val") and k in sd:
                sd[k].copy_(torch.tensor(z[k]))
        net.load_state_dict(sd)
    else:
        net = load_model(cand)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--candidate", required=True)
    ap.add_argument("--tier", default="validation"); ap.add_argument("--margin", type=float, default=0.002)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = tier(a.tier)
    b = evaluate(load_model(a.base), S); c = evaluate(load_candidate(a.base, a.candidate), S)
    worse = [k for k in ("cut", "slump", "far", "stalled", "completed") if c[k] > b[k] + a.margin]
    accept = (c["all"] < b["all"] - a.margin) and not worse and (c["carried_mae"] <= b["carried_mae"] + 0.01)
    res = {"tier": a.tier, "base": b, "candidate": c, "worse_regions": worse, "accept": bool(accept)}
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}) for k, v in res.items()}, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
