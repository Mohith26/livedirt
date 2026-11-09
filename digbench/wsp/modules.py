"""Target architectures for WRITE sessions (weights start at zero; the editing LLM programs them).

StopNet (E2): per bite, the fraction of the planned path completed before the machine stalls.
  conv_a  Conv3x3 7->16 + ReLU        per-cell features (cut depth = terrain - envelope, soil terms)
  conv_b  Conv1x1 16->16 + ReLU       piecewise-linear pieces of d^2, c*d (ReLU bases)
  band    mean over the bucket's lateral band (rows 32..48 of y), fixed, not a parameter
  force   Linear 16 -> 1 per x         resistance per unit drag length at each x
  accum   Linear 108 -> 108 (no bias)  cumulative / peak along x (e.g. a lower-triangular matrix)
  gate    Linear 108 -> 108 + ReLU     exceedance of the machine limit at each x
  out     Linear 108 -> 1 + sigmoid    completion fraction
"""
import torch
import torch.nn as nn


class StopNet(nn.Module):
    def __init__(self, nx=108):
        super().__init__()
        self.conv_a = nn.Conv2d(7, 16, 3, padding=1); self.conv_b = nn.Conv2d(16, 16, 1)
        self.force = nn.Linear(16, 1); self.accum = nn.Linear(nx, nx, bias=False)
        self.gate = nn.Linear(nx, nx); self.out = nn.Linear(nx, 1)
        for p in self.parameters():
            nn.init.zeros_(p)

    def forward(self, x):  # x: (B, 7, nx, ny)
        h = torch.relu(self.conv_b(torch.relu(self.conv_a(x))))
        h = h[:, :, :, 32:49].mean(-1)            # (B, 16, nx)
        f = self.force(h.transpose(1, 2))[..., 0]  # (B, nx)
        g = torch.relu(self.gate(self.accum(f)))
        return torch.sigmoid(self.out(g))[:, 0]
