"""Budgeted probes for weightspace sessions: what the editing LLM is allowed to observe. Each call costs one forward
pass of the session budget. All probes run on the PRACTICE tier only; the LLM never sees validation or gate data.

  error_report      overall / region / stalled-vs-completed RMS error and carried error on the practice tier
  worst_bites       the k practice bites with the largest error, with completion and soil
  error_map         one practice bite: terrain, true change, predicted change, error (downsampled numbers)
  feature_maps      activations of a named layer on one practice bite, NO outputs (the Exp-7-style restricted view)
"""
import numpy as np
import torch
import weightspace as wsp

from digbench.wsp.common import evaluate, predict, tier

MODEL = {"net": None}  # set by the session runner to the live, edited model


def _ds(a, k=4):
    a = np.asarray(a, float); return np.round(a[::k, ::k], 2).tolist()


@wsp.probe(cost=0)
def list_bites():
    """Free (no forward pass, no model outputs): names of all practice bites with soil preset, completion (<0.9 = the
    machine stalled) and grid size, so other probes can be pointed at specific bites."""
    return [{"name": s["name"], "soil": s["soil"].get("preset", s["soil"].get("phi")), "completion": round(float(s["completion"]), 3),
             "grid": list(s["delta"].shape)} for s in tier("practice")]


@wsp.probe(cost=1)
def error_report():
    """RMS surface error (m) on the practice tier: all, by region (cut / slump within 0.5 m / far), stalled vs completed
    bites, and carried-volume error (m^3)."""
    return evaluate(MODEL["net"], tier("practice"))


@wsp.probe(cost=1)
def worst_bites(k: int = 8):
    """The k practice bites with the largest RMS error: name, rmse, completion (<0.9 = the machine stalled), soil."""
    r = evaluate(MODEL["net"], tier("practice"), per_bite=True)["bites"]
    soil = {s["name"]: s["soil"].get("preset", s["soil"].get("phi")) for s in tier("practice")}
    r = sorted(r, key=lambda x: -x["rmse"])[:k]
    for x in r:
        x["soil"] = soil[x["name"]]
    return r


@wsp.probe(cost=1)
def error_map(name: str):
    """For one practice bite (by name): terrain, true surface change, predicted change and error, every 4th cell (m)."""
    s = next(s for s in tier("practice") if s["name"] == name)
    pred, ys = predict(MODEL["net"], s)
    return {"terrain_rel": _ds(s["X"][0]), "true_change": _ds(s["delta"]), "pred_change": _ds(pred),
            "error": _ds(pred - s["delta"]), "carried_true": float(s["scalars"][0]), "carried_pred": float(ys[0]),
            "completion_true": s["completion"], "completion_pred": float(ys[1])}


@wsp.probe(cost=1)
def feature_maps(layer: str, name: str, channels: int = 8):
    """Activations of a named module (e.g. 'e4', 'd1') on one practice bite, per channel mean / max and a coarse map of
    the first `channels` channels. Outputs of the network are NOT returned."""
    s = next(s for s in tier("practice") if s["name"] == name)
    from digbench.train_scoop import batchify
    X, *_ = batchify([s]); store = {}
    mod = dict(MODEL["net"].named_modules())[layer]
    h = mod.register_forward_hook(lambda m, i, o: store.__setitem__("a", o.detach()))
    with torch.no_grad():
        MODEL["net"](X)
    h.remove()
    a = store["a"][0].numpy()
    return {"shape": list(a.shape), "mean": np.round(a.mean((1, 2)), 3).tolist(), "max": np.round(a.max((1, 2)), 3).tolist(),
            "maps": [_ds(a[c], max(1, a.shape[1] // 16)) for c in range(min(channels, a.shape[0]))]}
