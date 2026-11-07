import { describe, it, expect } from 'vitest';
import { Heightfield } from '../src/sim/heightfield';
import { soilById } from '../src/sim/soil';

const flat = (N = 80, cell = 0.1, h = 0) => new Heightfield({ N, cell, half: (N * cell) / 2, H0: new Float32Array(N * N).fill(h) });

describe('Heightfield dig/dump bookkeeping', () => {
  it('conserves material: loose on field + carried == swell * (bank cut + sloughed)', () => {
    const hf = flat();
    hf.soil = soilById('common-earth');
    for (let s = 0; s < 30; s++) hf.dig(-0.5 + s * 0.03, 0, 0.5, 0.05);
    hf.dump(2, 2, hf.ledger.carried * 0.6, 0.8);
    hf.settle(600);
    const lhs = hf.looseOnField() + hf.ledger.carried;
    const rhs = hf.soil.swell * (hf.ledger.bankCut + hf.ledger.sloughed);
    expect(Math.abs(lhs - rhs) / rhs).toBeLessThan(1e-4);
    // bank change equals -(bank cut + sloughed)
    expect(Math.abs(hf.bankChange() + hf.ledger.bankCut + hf.ledger.sloughed)).toBeLessThan(1e-4);
  });

  it('spoil relaxes to (about) the angle of repose', () => {
    const hf = flat(120);
    hf.soil = soilById('sandy-gravel'); // repose 36 deg
    hf.ledger.carried = 3; // pretend we carried 3 m^3
    hf.dump(0, 0, 3, 0.4);
    hf.settle(3000);
    // max slope between 4-neighbours should not exceed tan(repose) by more than a small tolerance
    let maxSlope = 0;
    const N = hf.N;
    for (let j = 1; j < N - 1; j++) for (let i = 1; i < N - 1; i++) {
      const k = j * N + i; if (hf.L[k] <= 1e-4) continue;
      for (const n of [k + 1, k - 1, k + N, k - N]) maxSlope = Math.max(maxSlope, (hf.h(k) - hf.h(n)) / hf.cell);
    }
    const deg = (Math.atan(maxSlope) * 180) / Math.PI;
    expect(deg).toBeLessThan(36 + 3);
    expect(deg).toBeGreaterThan(25); // and it stays a pile, not a puddle
  });

  it('never digs inside masked cells and never below max dig depth', () => {
    const N = 60; const mask = new Uint8Array(N * N);
    for (let j = 25; j < 35; j++) for (let i = 25; i < 35; i++) mask[j * N + i] = 1;
    const hf = new Heightfield({ N, cell: 0.1, half: 3, H0: new Float32Array(N * N).fill(1), mask });
    hf.maxDigDepth = 0.5;
    for (let s = 0; s < 100; s++) hf.dig(0, 0, 1.5, 0.2);
    for (let k = 0; k < N * N; k++) {
      if (mask[k]) expect(hf.h(k)).toBe(1);
      expect(hf.h(k)).toBeGreaterThanOrEqual(1 - 0.5 - 1e-6);
    }
  });

  it('grade assist stops the cut at design grade; stats report on-grade cells', () => {
    const hf = flat(100, 0.1, 0);
    hf.soil = soilById('wet-clay');
    hf.setDesign({ p0: [-2, 0], p1: [2, 0], bottomWidth: 0.8, depth: 0.6, sideSlope: 0 });
    for (let pass = 0; pass < 40; pass++) for (let x = -2; x <= 2; x += 0.1) hf.dig(x, 0, 0.5, 0.1, true);
    const st = hf.gradeStats(0.03);
    expect(st.cells).toBeGreaterThan(0);
    expect(st.overDig).toBeLessThan(1e-6);
    expect(st.onGradePct).toBeGreaterThan(80);
  });

  it('machine-control mode never cuts outside the design footprint', () => {
    const hf = flat(100, 0.1, 0);
    hf.setDesign({ p0: [-1, 0], p1: [1, 0], bottomWidth: 0.6, depth: 0.5, sideSlope: 1 });
    for (let pass = 0; pass < 30; pass++) for (let x = -2.5; x <= 2.5; x += 0.1) for (let zz = -1.5; zz <= 1.5; zz += 0.25) hf.dig(x, zz, 0.6, 0.1, true, true);
    expect(hf.gradeStats().unplannedCut).toBeLessThan(1e-9);
    expect(hf.gradeStats().overDig).toBeLessThan(1e-9);
    expect(hf.gradeStats().onGradePct).toBeGreaterThan(95);
  });

  it('existing steep faces from the scan do not collapse; new over-steep cuts do', () => {
    const N = 60; const H0 = new Float32Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) H0[j * N + i] = i < 30 ? 1 : 0; // a 1 m vertical step that existed in the scan
    const hf = new Heightfield({ N, cell: 0.1, half: 3, H0 });
    hf.soil = soilById('sandy-gravel');
    hf.dig(1.5, 1.5, 0.3, 0.001); // tiny disturbance far from the step to activate relaxation
    hf.settle(200);
    expect(hf.ledger.sloughed).toBe(0);
    // now cut a deep vertical hole in flat ground: should slough
    hf.dig(-1.5, -1.5, 0.4, 1.0);
    hf.settle(800);
    expect(hf.ledger.sloughed).toBeGreaterThan(0);
  });

  it('undo restores state and ledger', () => {
    const hf = flat();
    hf.beginStroke(); hf.dig(0, 0, 0.5, 0.3);
    expect(hf.ledger.bankCut).toBeGreaterThan(0);
    hf.undo();
    expect(hf.ledger.bankCut).toBe(0);
    expect(hf.bankChange()).toBe(0);
  });
});
