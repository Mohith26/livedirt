"""Controls every accepted edit must beat (otherwise the edit is not evidence for weight-space programming):
  gradient   fine-tune the SAME tensors the edit touched (or all, with --all) on the practice tier with Adam, for the same
             number of practice evaluations the session used (one epoch over practice = one forward pass)
  random     a random perturbation of the same tensors with the same Frobenius norm as the edit
  python -m digbench.wsp.controls --base w0.pt --edited runs/E3-a/patched.pt --passes 10 --out runs/E3-a/controls
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from digbench.scoop_model import CELL
from digbench.train_scoop import batchify
from digbench.wsp.common import load_model, tier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--edited", required=True)
    ap.add_argument("--passes", type=int, default=10); ap.add_argument("--all", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    base, ed = load_model(a.base), load_model(a.edited)
    sb, se = base.state_dict(), ed.state_dict()
    touched = [k for k in sb if not torch.equal(sb[k], se[k])]
    print("tensors touched by the edit:", touched)
    # random control
    rnd = load_model(a.base); sr = rnd.state_dict(); g = torch.Generator().manual_seed(0)
    for k in touched:
        d = se[k] - sb[k]; noise = torch.randn(d.shape, generator=g); sr[k] += noise * (d.norm() / (noise.norm() + 1e-12))
    rnd.load_state_dict(sr); torch.save(rnd.state_dict(), out / "random.pt")
    # gradient control
    gm = load_model(a.base); names = set(sb) if a.all else set(touched)
    params = [p for n, p in gm.named_parameters() if n in names]
    for p in gm.parameters():
        p.requires_grad_(any(p is q for q in params))
    opt = torch.optim.Adam(params, lr=3e-4); S = tier("practice"); gm.train()
    for _ in range(a.passes):
        for i in range(0, len(S), 8):
            X, P, D, Y, (nx, ny) = batchify(S[i:i + 8]); r, ys = gm(X); pred = P + r
            w = 1.0 + 9.0 * ((D.abs() > 0.01) | (P.abs() > 0.01)).float()
            loss = (w * (pred - D) ** 2)[:, :nx, :ny].mean() + ((pred[:, :nx, :ny].sum((1, 2)) - D[:, :nx, :ny].sum((1, 2))) * CELL * CELL).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    gm.eval(); torch.save(gm.state_dict(), out / "gradient.pt")
    (out / "touched.json").write_text(json.dumps(touched))


if __name__ == "__main__":
    main()
