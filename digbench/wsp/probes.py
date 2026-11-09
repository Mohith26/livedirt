"""Budgeted probes for weightspace sessions: what the editing LLM is allowed to observe. Every probe that runs the network
costs one forward pass of the session budget; list_bites is free. All probes use the PRACTICE tier only; the LLM never
sees validation or gate data. Outputs are sized to stay under the harness's 12,000-character tool limit.

  list_bites        free: practice bite names, soil, true completion, calculator's predicted completion, drag direction
  error_report      RMS error overall / by region / stalled vs completed, carried error
  worst_bites       the k worst practice bites
  path_profile      one bite, along the drag axis, averaged across the bucket width: true vs predicted change, prior,
                    envelope, beyond-stop channel (the compact way to see where a stalled bite stops)
  error_map         one bite, 2-D maps (chosen fields, coarse stride)
  feature_maps      one bite, one layer: mean/max of every channel, maps of chosen channels; no outputs shown
  compare_groups    one layer: per-channel mean/max averaged over stalled vs completed practice bites, ranked by gap
"""
import numpy as np
import torch
import weightspace as wsp

from digbench.wsp.common import batch, evaluate, predict, tier

MODEL = {"net": None}  # set by the session runner to the live, edited model
LIMIT = 10500


def _ds(a, k):
    a = np.asarray(a, float); return np.round(a[::k, ::k], 2).tolist()


def _fit(build, strides=(3, 4, 5, 6, 8, 10, 12, 16)):
    """Pick the finest stride whose output fits the tool limit."""
    for k in strides:
        out = build(k)
        if len(str(out)) <= LIMIT:
            out["stride"] = k
            return out
    out = build(strides[-1]); out["stride"] = strides[-1]; return out


def _bite(name):
    for s in tier("practice"):
        if s["name"] == name:
            return s
    raise ValueError(f"unknown practice bite {name!r}; call list_bites")


@wsp.probe(cost=0)
def list_bites():
    """Free (no forward pass, no model outputs): every practice bite with soil (phi deg, cohesion Pa, density), true
    completion (<0.9 = the machine stalled), the stop calculator's predicted completion (what input channel 7 encodes,
    when present), drag direction along x, and grid size."""
    rows = []
    for s in tier("practice"):
        so = s["soil"]
        rows.append([s["name"], round(float(so.get("phi", 0)), 1), round(float(so.get("c", 0))), round(float(so.get("rho", 0))),
                     round(float(s["completion"]), 3), round(float(s["completion_pred"]), 3), s["drag"]])
    return {"columns": ["name", "phi", "c", "rho", "completion", "completion_pred", "drag"], "grid": "108 x 80 for all", "rows": rows}


@wsp.probe(cost=1)
def error_report():
    """RMS surface error (m) on the practice tier: all, by region (cut / slump within 0.5 m / far), stalled vs completed
    bites, by planned depth (deep_planned: envelope more than 0.6 m below terrain; shallow_planned), beyond_stop (cells
    channel 7 marks), and carried-volume error (m^3)."""
    return evaluate(MODEL["net"], tier("practice"))


@wsp.probe(cost=1)
def worst_bites(k: int = 8):
    """The k practice bites with the largest RMS error: name, rmse, completion (<0.9 = the machine stalled), soil."""
    r = evaluate(MODEL["net"], tier("practice"), per_bite=True)["bites"]
    soil = {s["name"]: round(float(s["soil"].get("phi", 0)), 1) for s in tier("practice")}
    r = sorted(r, key=lambda x: -x["rmse"])[:k]
    for x in r:
        x["phi"] = soil[x["name"]]; x["rmse"] = round(x["rmse"], 4)
    return r


@wsp.probe(cost=1)
def path_profile(name: str, step: int = 2):
    """One practice bite, row by row along x (every `step` rows), averaged over the bucket width (y = 31..49):
    true change, predicted change, prior (ch 3), envelope (ch 1), beyond-stop (ch 7 if present) and terrain (ch 0).
    Also the true / predicted completion and drag direction. The quickest way to see where a stalled bite stopped."""
    s = _bite(name); pred, ys = predict(MODEL["net"], s); sl = slice(31, 50)
    cols = {"true": s["delta"][:, sl].mean(1), "pred": pred[:, sl].mean(1), "prior": s["X"][3][:, sl].mean(1),
            "env": s["X"][1][:, sl].mean(1), "beyond": s["stop"][:, sl].mean(1), "terrain": s["X"][0][:, sl].mean(1)}
    rows = [[x] + [round(float(v[x]), 2) for v in cols.values()] for x in range(0, s["delta"].shape[0], max(1, step))]
    return {"columns": ["x"] + list(cols), "rows": rows, "completion_true": round(float(s["completion"]), 3),
            "completion_pred_calculator": round(float(s["completion_pred"]), 3), "completion_pred_head": round(float(ys[1]), 3),
            "drag": s["drag"]}


@wsp.probe(cost=1)
def error_map(name: str, fields: str = "true_change,pred_change,error"):
    """One practice bite as 2-D maps (m), coarse stride chosen to fit. fields: any of terrain_rel, true_change,
    pred_change, error, prior, beyond_stop (comma separated)."""
    s = _bite(name); pred, ys = predict(MODEL["net"], s)
    F = {"terrain_rel": s["X"][0], "true_change": s["delta"], "pred_change": pred, "error": pred - s["delta"],
         "prior": s["X"][3], "beyond_stop": s["stop"]}
    want = [f.strip() for f in fields.split(",") if f.strip() in F] or ["error"]
    return _fit(lambda k: {**{f: _ds(F[f], k) for f in want}, "carried_true": round(float(s["scalars"][0]), 3),
                           "carried_pred": round(float(ys[0]), 3), "completion_true": round(float(s["completion"]), 3),
                           "completion_pred_head": round(float(ys[1]), 3)})


def _acts(layer, S):
    net = MODEL["net"]; mod = dict(net.named_modules())[layer]; store = []
    h = mod.register_forward_hook(lambda m, i, o: store.append(o.detach()[0].numpy()))
    with torch.no_grad():
        for s in S:
            X, *_ = batch(net, [s]); net(X)
    h.remove()
    return store


@wsp.probe(cost=1)
def feature_maps(layer: str, name: str, channels: str = ""):
    """Activations of a named module (e.g. 'e4', 'd1', 'e1.3') on one practice bite: mean and max of EVERY channel, plus
    coarse maps of the listed channels (comma separated, e.g. '1,17,33'; default first 4). Outputs NOT returned."""
    a = _acts(layer, [_bite(name)])[0]
    ch = [int(c) for c in channels.split(",") if c.strip()] or list(range(min(4, a.shape[0])))
    base = {"shape": list(a.shape), "mean": np.round(a.mean((1, 2)), 3).tolist(), "max": np.round(a.max((1, 2)), 3).tolist()}
    return _fit(lambda k: {**base, "maps": {c: _ds(a[c], k) for c in ch if 0 <= c < a.shape[0]}},
                strides=tuple(max(1, a.shape[1] // d) for d in (24, 16, 12, 8, 6, 4)))


@wsp.probe(cost=1)
def compare_groups(layer: str, top: int = 24):
    """One layer over ALL practice bites: per channel, mean of the spatial mean and of the spatial max on stalled vs
    completed bites, ranked by the standardised gap in the max. Shows which channels carry stall information."""
    S = tier("practice"); A = _acts(layer, S); st = np.array([s["completion"] < 0.9 for s in S])
    mx = np.stack([a.max((1, 2)) for a in A]); mn = np.stack([a.mean((1, 2)) for a in A])
    gap = (mx[st].mean(0) - mx[~st].mean(0)) / (mx.std(0) + 1e-6)
    order = np.argsort(-np.abs(gap))[:top]
    return {"n_stalled": int(st.sum()), "n_completed": int((~st).sum()),
            "channels": [{"ch": int(c), "gap_std": round(float(gap[c]), 2), "max_stalled": round(float(mx[st, c].mean()), 3),
                          "max_completed": round(float(mx[~st, c].mean()), 3), "mean_stalled": round(float(mn[st, c].mean()), 3),
                          "mean_completed": round(float(mn[~st, c].mean()), 3)} for c in order]}


ALL = [list_bites, error_report, worst_bites, path_profile, error_map, feature_maps, compare_groups]
