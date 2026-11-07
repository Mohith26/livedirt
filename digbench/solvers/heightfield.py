"""Python port of LiveDirt's heightfield relaxation (app/src/sim/heightfield.ts, relax()), unmodified logic.

Two layers: bank B (undisturbed) and loose L. Loose material flows to 8 neighbours where the slope exceeds
the angle of repose; bank faces steeper than the critical (bank) angle slough into loose material (x swell),
unless they were already that steep in the initial scan H0 (+2 cm). Shares 0.16 (axial) / 0.09 (diagonal),
alternating sweep direction each iteration, settle when max move < 2e-4 m.
"""
import math

import numpy as np


class Heightfield:
    def __init__(self, H0, cell, repose_deg, bank_deg, swell=1.0):
        self.H0 = np.array(H0, dtype=np.float64)
        self.B = self.H0.copy()
        self.L = np.zeros_like(self.B)
        self.cell = cell
        self.tanL = math.tan(math.radians(repose_deg)) * cell
        self.tanB = math.tan(math.radians(bank_deg)) * cell
        self.swell = swell
        self.sloughed = 0.0

    def h(self):
        return self.B + self.L

    def relax(self, iterations=2):
        B, L, H0 = self.B, self.L, self.H0
        nx, ny = B.shape
        nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        dist = [1.0] * 4 + [math.sqrt(2)] * 4
        share = [0.16] * 4 + [0.09] * 4
        max_move = 0.0
        for it in range(iterations):
            xs = range(1, nx - 1) if it % 2 == 0 else range(nx - 2, 0, -1)
            ys = range(1, ny - 1) if it % 2 == 0 else range(ny - 2, 0, -1)
            for j in ys:
                for i in xs:
                    for q, (di, dj) in enumerate(nb):
                        a, b = i + di, j + dj
                        hn = B[a, b] + L[a, b]
                        faceB = B[i, j] - hn
                        allowed = max(self.tanB * dist[q], H0[i, j] - H0[a, b] + 0.02)
                        if faceB > allowed:
                            c = (faceB - allowed) * share[q]
                            B[i, j] -= c
                            L[i, j] += c * self.swell
                            self.sloughed += c * self.cell ** 2
                            max_move = max(max_move, c)
                        lk = L[i, j]
                        if lk <= 0:
                            continue
                        d = B[i, j] + lk - hn
                        excess = d - self.tanL * dist[q]
                        if excess > 1e-4:
                            qv = min(lk, excess * share[q])
                            L[i, j] -= qv
                            L[a, b] += qv
                            max_move = max(max_move, qv)
        return max_move

    def settle(self, max_steps=4000):
        for _ in range(max_steps):
            if self.relax(2) <= 2e-4:
                break
