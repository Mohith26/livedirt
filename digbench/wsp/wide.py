"""Function-preserving widening of the scoop U-Net: every conv block gets ONE extra GroupNorm group of all-zero channels
(e1/d1 +4, e2/d2 +8, e3/d3 +16, e4 +32), appended after the original channels of every tensor. The spare channels have
zero incoming and outgoing weights, so they output exactly 0 and the network's predictions are unchanged; an editor can
build a side path there without taking over channels the network already uses (the blocker Opus 5.5 and Fable 5.1 both
hit in E3, 2026-11-09).

  python -m digbench.wsp.wide --base ~/v4/wsp/w0.pt --out ~/v4/wsp/w0-wide.pt
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as Fn

EXTRA = 1  # extra GroupNorm groups per block


def wconv(ci, co, groups):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(groups, co), nn.SiLU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(groups, co), nn.SiLU())


def widths(ch=32, extra=EXTRA):
    w = lambda c: c + extra * (c // 8)
    return {"e1": w(ch), "e2": w(2 * ch), "e3": w(4 * ch), "e4": w(8 * ch), "d3": w(4 * ch), "d2": w(2 * ch), "d1": w(ch)}


class WideUNet(nn.Module):
    """Same forward as train_scoop.UNet; widths from widths()."""

    def __init__(self, cin=7, ch=32, extra=EXTRA):
        super().__init__()
        W = widths(ch, extra); g = 8 + extra
        self.e1, self.e2 = wconv(cin, W["e1"], g), wconv(W["e1"], W["e2"], g)
        self.e3, self.e4 = wconv(W["e2"], W["e3"], g), wconv(W["e3"], W["e4"], g)
        self.d3 = wconv(W["e4"] + W["e3"], W["d3"], g)
        self.d2 = wconv(W["d3"] + W["e2"], W["d2"], g)
        self.d1 = wconv(W["d2"] + W["e1"], W["d1"], g)
        self.out = nn.Conv2d(W["d1"], 1, 1)
        self.head = nn.Sequential(nn.Linear(2 * W["e4"], 128), nn.SiLU(), nn.Linear(128, 3))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(Fn.max_pool2d(e1, 2))
        e3 = self.e3(Fn.max_pool2d(e2, 2))
        e4 = self.e4(Fn.max_pool2d(e3, 2))
        g = torch.cat([e4.mean((2, 3)), e4.amax((2, 3))], 1)
        d3 = self.d3(torch.cat([Fn.interpolate(e4, size=e3.shape[2:]), e3], 1))
        d2 = self.d2(torch.cat([Fn.interpolate(d3, size=e2.shape[2:]), e2], 1))
        d1 = self.d1(torch.cat([Fn.interpolate(d2, size=e1.shape[2:]), e1], 1))
        return self.out(d1)[:, 0], self.head(g)


def is_wide(sd, ch=32):
    return sd["e1.0.weight"].shape[0] != ch


@torch.no_grad()
def widen(net, ch=32, extra=EXTRA):
    """Copy a trained UNet into a WideUNet; spare channels all zero."""
    old = {"e1": ch, "e2": 2 * ch, "e3": 4 * ch, "e4": 8 * ch, "d3": 4 * ch, "d2": 2 * ch, "d1": ch}
    W = widths(ch, extra)
    # input layout of each block, as a list of (n_old, n_new) segments in concat order
    ins = {"e1": [(7, 7)], "e2": [(old["e1"], W["e1"])], "e3": [(old["e2"], W["e2"])], "e4": [(old["e3"], W["e3"])],
           "d3": [(old["e4"], W["e4"]), (old["e3"], W["e3"])], "d2": [(old["d3"], W["d3"]), (old["e2"], W["e2"])],
           "d1": [(old["d2"], W["d2"]), (old["e1"], W["e1"])]}
    wn = WideUNet(7, ch, extra)
    for p in wn.parameters():
        p.zero_()
    sd_o, sd_w = net.state_dict(), wn.state_dict()

    def in_index(segs):  # positions of old input channels inside the new concatenated input
        idx, off = [], 0
        for no, nn_ in segs:
            idx += list(range(off, off + no)); off += nn_
        return torch.tensor(idx)

    for b in old:
        co = old[b]; ii = in_index(ins[b]); mid = torch.arange(co)
        sd_w[f"{b}.0.weight"][:co][:, ii] = sd_o[f"{b}.0.weight"]
        sd_w[f"{b}.0.bias"][:co] = sd_o[f"{b}.0.bias"]
        sd_w[f"{b}.3.weight"][:co][:, mid] = sd_o[f"{b}.3.weight"]
        sd_w[f"{b}.3.bias"][:co] = sd_o[f"{b}.3.bias"]
        for k in ("1", "4"):  # GroupNorm affine: originals copied, spare gamma=1 beta=0 (inputs are 0 -> output 0)
            sd_w[f"{b}.{k}.weight"][:] = 1.0
            sd_w[f"{b}.{k}.weight"][:co] = sd_o[f"{b}.{k}.weight"]
            sd_w[f"{b}.{k}.bias"][:co] = sd_o[f"{b}.{k}.bias"]
    sd_w["out.weight"][:, :ch] = sd_o["out.weight"]; sd_w["out.bias"][:] = sd_o["out.bias"]
    e4o, e4w = old["e4"], W["e4"]
    hi = torch.cat([torch.arange(e4o), e4w + torch.arange(e4o)])  # [mean-pool | max-pool] halves
    sd_w["head.0.weight"][:, hi] = sd_o["head.0.weight"]; sd_w["head.0.bias"][:] = sd_o["head.0.bias"]
    sd_w["head.2.weight"][:] = sd_o["head.2.weight"]; sd_w["head.2.bias"][:] = sd_o["head.2.bias"]
    wn.load_state_dict(sd_w); wn.eval()
    return wn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from digbench.wsp.common import evaluate, load_model, predict, tier
    net = load_model(a.base); wn = widen(net)
    torch.save(wn.state_dict(), a.out)
    wn2 = load_model(a.out)
    S = tier("validation") + tier("practice")
    dmax = max(float(abs(predict(net, s)[0] - predict(wn2, s)[0]).max()) for s in S)
    hmax = max(float(abs(predict(net, s)[1] - predict(wn2, s)[1]).max()) for s in S)
    n_o, n_w = sum(p.numel() for p in net.parameters()), sum(p.numel() for p in wn2.parameters())
    print(f"params {n_o:,} -> {n_w:,}; max |surface diff| {dmax:.2e} m, max |head diff| {hmax:.2e} over {len(S)} bites")
    print("validation narrow", {k: round(v, 5) for k, v in evaluate(net, tier('validation')).items() if isinstance(v, float)})
    print("validation wide  ", {k: round(v, 5) for k, v in evaluate(wn2, tier('validation')).items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
