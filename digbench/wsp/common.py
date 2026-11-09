"""Shared pieces: load the scoop model, load tier samples (cached), evaluate per bite with region and stall breakdown."""
import json, math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from digbench.scoop_model import load_sample, swept_envelope
from digbench.train_scoop import UNet, batchify

SPLITS = Path("~/v4/wsp/splits.json").expanduser()


def load_model(path, ch=32):
    sd = torch.load(path, map_location="cpu")
    from digbench.wsp.wide import WideUNet, is_wide
    cin = int(sd["e1.0.weight"].shape[1])
    net = WideUNet(cin, ch) if is_wide(sd, ch) else UNet(cin, ch)
    net.load_state_dict(sd); net.eval()
    return net


@lru_cache(maxsize=4)
def tier(name):
    files = json.loads(SPLITS.read_text())[name]
    out = []
    for f in files:
        s = load_sample(f); z = np.load(f)
        T = z["terrain"].astype(float)
        env = swept_envelope(z["poses"], *T.shape, float(z["margin"]))
        cut = np.isfinite(env) & (env < T)
        near = cut.copy()
        for _ in range(10):
            near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
        s["regions"] = {"cut": cut, "slump": near & ~cut, "far": ~near}
        s["completion"] = json.loads(str(z["meta"]))["completion"]
        s["file"] = f
        P_ = z["poses"]; s["drag"] = "-x" if P_[-1, 1] < P_[0, 1] else "+x"
        try:
            from digbench.wsp.stop_point import for_file
            s["stop"], s["completion_pred"] = for_file(f)
        except FileNotFoundError:  # stop params not fitted yet
            s["stop"], s["completion_pred"] = np.zeros_like(s["prior"]), 1.0
        out.append(s)
    return out


def n_inputs(net):
    return int(net.e1[0].in_channels)


def with_inputs(net, S):
    """Samples with X matching the net: 7 base channels, plus channel 7 = beyond-predicted-stop for 8-input nets."""
    k = n_inputs(net)
    if k == 7:
        return S
    return [{**s, "X": np.concatenate([s["X"], s["stop"][None]]).astype(np.float32)} for s in S]


def batch(net, S, flip=False):
    return batchify(with_inputs(net, S), flip=flip)


@torch.no_grad()
def predict(net, s):
    X, P, D, Y, (nx, ny) = batch(net, [s])
    r, ys = net(X)
    return (P + r)[0, :nx, :ny].numpy(), ys[0].numpy()


def evaluate(net, samples, per_bite=False):
    """RMS surface error over disturbed cells, split by region and by stalled / completed bites; carried error."""
    acc = {"all": [0.0, 0], "cut": [0.0, 0], "slump": [0.0, 0], "far": [0.0, 0], "stalled": [0.0, 0], "completed": [0.0, 0],
           "deep_planned": [0.0, 0], "shallow_planned": [0.0, 0], "beyond_stop": [0.0, 0]}
    car, rows = [], []
    for s in samples:
        pred, ys = predict(net, s); d = s["delta"]
        m = (np.abs(d) > 0.01) | (np.abs(pred) > 0.01); e2 = (pred - d) ** 2
        acc["all"][0] += e2[m].sum(); acc["all"][1] += m.sum()
        for k, rm in s["regions"].items():
            acc[k][0] += e2[m & rm].sum(); acc[k][1] += (m & rm).sum()
        sw = s["X"][2] < 0.5  # zones (reported, not used by the judge): planned depth > 0.6 m, shallower swept, beyond stop
        for k, zm in (("deep_planned", sw & (s["X"][1] < -0.6)), ("shallow_planned", sw & (s["X"][1] >= -0.6)), ("beyond_stop", s["stop"] > 0.5)):
            acc[k][0] += e2[m & zm].sum(); acc[k][1] += (m & zm).sum()
        k = "stalled" if s["completion"] < 0.9 else "completed"
        acc[k][0] += e2[m].sum(); acc[k][1] += m.sum()
        car.append(abs(float(ys[0]) - float(s["scalars"][0])))
        if per_bite:
            rows.append({"name": s["name"], "rmse": float(np.sqrt(e2[m].mean())) if m.any() else 0.0, "completion": s["completion"]})
    out = {k: float(math.sqrt(v[0] / max(v[1], 1))) for k, v in acc.items()}
    out["carried_mae"] = float(np.mean(car)); out["n_bites"] = len(samples)
    if per_bite:
        out["bites"] = rows
    return out
