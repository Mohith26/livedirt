"""Answer key for E1 (READ) by intervention on the network, computed after the sessions and never shown to the editor.

Q1  input channel with the largest effect on the completion head (ys[1]). PRIMARY (scale-fair): swap channel c with the
    same channel of another bite and measure mean |change|; confirmed by input x gradient. Sign = summed input gradient.
    (The first version used replace-by-dataset-mean, which inflates binary channels such as the untouched mask; it named
    ch 2. Corrected 2026-11-09: all three scale-fair measures name ch 1, positive.)
Q2  soil channels 4-6: swap each bite's soil planes for another soil's and (a) rank e1 filters by normalised activation
    change, (b) split the output change into the part that flows through the e1->d1 skip ("decoder") vs the part through
    e2..e4 ("encoder"), by patching only one path at a time.
Q3  e4 channels for "the bucket will stall": mean-ablate each e4 channel, measure mean |change| of the completion head;
    also report AUC of the channel's spatial max for stalled (completion < 0.9) bites.
Q4  d1 -> out writers: contribution w_c * d1_c; ratio of mean |contribution| on cut cells (channel 2 == 0) vs outside.

  python -m digbench.wsp.e1_key --base ~/v4/wsp/w0.pt --out ~/v4/wsp/e1_key.json
"""
import argparse, json

import numpy as np
import torch
import torch.nn.functional as Fn

from digbench.train_scoop import batchify
from digbench.wsp.common import load_model, tier


def fwd(net, x, e1_skip=None, e1_deep=None, e4_override=None):
    e1 = net.e1(x)
    e1d = e1 if e1_deep is None else e1_deep
    e1s = e1 if e1_skip is None else e1_skip
    e2 = net.e2(Fn.max_pool2d(e1d, 2)); e3 = net.e3(Fn.max_pool2d(e2, 2)); e4 = net.e4(Fn.max_pool2d(e3, 2))
    if e4_override is not None:
        e4 = e4_override(e4)
    g = torch.cat([e4.mean((2, 3)), e4.amax((2, 3))], 1)
    d3 = net.d3(torch.cat([Fn.interpolate(e4, size=e3.shape[2:]), e3], 1))
    d2 = net.d2(torch.cat([Fn.interpolate(d3, size=e2.shape[2:]), e2], 1))
    d1 = net.d1(torch.cat([Fn.interpolate(d2, size=e1.shape[2:]), e1s], 1))
    return net.out(d1)[:, 0], net.head(g), e1, e4, d1


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    net = load_model(a.base); torch.manual_seed(0); rng = np.random.default_rng(0)
    S = tier("practice") + tier("validation")
    Xs = [batchify([s])[0] for s in S]
    chmean = torch.stack([x[0].mean((1, 2)) for x in Xs]).mean(0)
    stalled = np.array([s["completion"] < 0.9 for s in S])

    # Q1
    eff = np.zeros(7); grad = np.zeros(7)
    for x in Xs:
        base_c = fwd(net, x)[1][0, 1]
        for c in range(7):
            x2 = x.clone(); x2[0, c] = chmean[c]
            eff[c] += abs(float(fwd(net, x2)[1][0, 1] - base_c))
        with torch.enable_grad():
            xg = x.clone().requires_grad_(True); fwd(net, xg)[1][0, 1].backward()
            grad += xg.grad[0].sum((1, 2)).numpy()
    eff /= len(Xs)
    swap = np.zeros(7)
    for i, x in enumerate(Xs):
        base_c = fwd(net, x)[1][0, 1]; j = int(rng.integers(len(Xs)))
        if Xs[j].shape != x.shape:
            continue
        for c in range(7):
            x2 = x.clone(); x2[0, c] = Xs[j][0, c]
            swap[c] += abs(float(fwd(net, x2)[1][0, 1] - base_c))
    swap /= len(Xs)
    q1 = {"swap_effect": swap.round(4).tolist(), "mean_ablation_effect_biased": eff.round(4).tolist(),
          "grad_sign": np.sign(grad).astype(int).tolist(), "top": int(swap.argmax()), "top_sign": "+" if grad[swap.argmax()] > 0 else "-"}

    # Q2
    soils = np.stack([x[0, 4:7].mean((1, 2)).numpy() for x in Xs])
    e1chg = np.zeros(net.e1[0].out_channels if hasattr(net.e1, "__getitem__") else 32); skip_e = deep_e = 0.0; head_e = 0.0
    for i, x in enumerate(Xs):
        others = [j for j in range(len(Xs)) if np.abs(soils[j] - soils[i]).max() > 0.05]
        if not others:
            continue
        j = others[rng.integers(len(others))]
        x2 = x.clone(); x2[0, 4:7] = torch.tensor(soils[j], dtype=x.dtype)[:, None, None]
        o, h, e1, *_ = fwd(net, x); _, _, e1b, *_ = fwd(net, x2)
        sd = e1.std((2, 3))[0] + 1e-6
        e1chg += ((e1b - e1).abs().mean((2, 3))[0] / sd).numpy()
        o_s = fwd(net, x, e1_skip=e1b)[0]; o_d, h_d, *_ = fwd(net, x, e1_deep=e1b)
        skip_e += float((o_s - o).abs().mean()); deep_e += float((o_d - o).abs().mean()); head_e += float((h_d - h).abs().mean())
    q2 = {"e1_rank": np.argsort(-e1chg).tolist()[:8], "e1_change": np.round(np.sort(e1chg)[::-1][:8] / len(Xs), 3).tolist(),
          "surface_change_via_skip_decoder": round(skip_e / len(Xs), 5), "surface_change_via_encoder": round(deep_e / len(Xs), 5),
          "head_change_via_encoder": round(head_e / len(Xs), 5),
          "mostly": "encoder" if deep_e > skip_e else "decoder"}

    # Q3
    n4 = net.e4[0].out_channels if hasattr(net.e4, "__getitem__") else 256
    e4m = None; e4max = []
    for x in Xs:
        e4 = fwd(net, x)[3]; e4max.append(e4.amax((2, 3))[0].numpy())
        e4m = e4.mean((2, 3))[0] if e4m is None else e4m + e4.mean((2, 3))[0]
    e4m /= len(Xs); e4max = np.stack(e4max); n4 = e4max.shape[1]
    abl = np.zeros(n4)
    for x in Xs:
        base_c = fwd(net, x)[1][0, 1]
        for c in range(n4):
            def ov(e4, c=c):
                e4 = e4.clone(); e4[:, c] = e4m[c]; return e4
            abl[c] += abs(float(fwd(net, x, e4_override=ov)[1][0, 1] - base_c))
    abl /= len(Xs)
    aucs = np.array([auc(e4max[:, c], stalled) for c in range(n4)])
    sep = np.abs(aucs - 0.5)
    q3 = {"top_by_ablation": np.argsort(-abl).tolist()[:10], "ablation_effect": np.round(np.sort(abl)[::-1][:10], 4).tolist(),
          "top_by_stall_auc": np.argsort(-sep).tolist()[:10], "auc": np.round(aucs[np.argsort(-sep)[:10]], 3).tolist(),
          "n_stalled": int(stalled.sum()), "n": len(S)}

    # Q4
    w = net.out.weight[0, :, 0, 0]; inside = np.zeros(w.numel()); outside = np.zeros(w.numel()); ni = no = 0
    for x in Xs:
        d1 = fwd(net, x)[4][0]; contrib = (w[:, None, None] * d1).abs()
        cutm = x[0, 2] < 0.5
        if cutm.any() and (~cutm).any():
            inside += contrib[:, cutm].sum(1).numpy(); outside += contrib[:, ~cutm].sum(1).numpy()
            ni += int(cutm.sum()); no += int((~cutm).sum())
    mi, mo = inside / max(ni, 1), outside / max(no, 1); ratio = mi / (mo + 1e-9)
    strong = (mi + mo) > np.quantile(mi + mo, 0.25)  # ignore near-silent channels
    order = np.argsort(-ratio)
    q4 = {"cut_writers": [int(c) for c in order if strong[c]][:6], "outside_writers": [int(c) for c in order[::-1] if strong[c]][:6],
          "ratio_cut_over_outside": {int(c): round(float(ratio[c]), 2) for c in order if strong[c]}}

    key = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "n_bites": len(S), "base": a.base}
    open(a.out, "w").write(json.dumps(key, indent=1)); print(json.dumps(key, indent=1)[:4000])


if __name__ == "__main__":
    main()
