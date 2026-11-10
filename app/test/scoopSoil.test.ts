// @ts-nocheck - node-only fixture test (fs, onnxruntime-node); run by vitest
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { makeInputs, sweptEnvelope, ScoopSoil, type BucketGeom } from '../src/sim/scoopSoil';

const fx = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/scoop_fixture.json'), 'utf8'));
// v3: Python reference output of the a16 ensemble (5 members, mean residual + scalars) on the same fixture inputs
const fx16 = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/scoop_fixture_a16.json'), 'utf8'));
const geom: BucketGeom = { TP: fx.TP, LIP: fx.LIP };
const nx = fx.nx, ny = fx.ny;

const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => { let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k])); return m; };

describe('per-scoop learned soil (port of digbench/scoop_model.py)', () => {
  it('swept envelope matches Python', () => {
    const env = sweptEnvelope(fx.poses, nx, ny, fx.M, geom);
    let worst = 0;
    for (let k = 0; k < env.length; k++) {
      const py = fx.env[k] === -999 ? NaN : fx.env[k];
      expect(Number.isNaN(env[k])).toBe(Number.isNaN(py));
      if (!Number.isNaN(py)) worst = Math.max(worst, Math.abs(env[k] - py));
    }
    expect(worst).toBeLessThan(1e-4);
  });
  it('inputs and prior match Python', () => {
    const { X, priorVol } = makeInputs(Float64Array.from(fx.terrain), fx.poses, fx.soil, nx, ny, fx.M, geom);
    expect(maxAbs(X, fx.X)).toBeLessThan(1e-3);
    expect(Math.abs(priorVol - fx.prior_vol)).toBeLessThan(1e-3);
  });
  it('full bite, a16 ensemble of 5 (ONNX + relax projection) matches Python', async () => {
    const ort = await import('onnxruntime-node');
    const sessions = await Promise.all([0, 1, 2, 3, 4].map((i) => ort.InferenceSession.create(path.join(__dirname, `../public/models/scoop-a16-s${i}.onnx`))));
    const s = ScoopSoil.fromParts(ort as never, sessions as never, geom);
    expect(s.members).toBe(5);
    const n = nx * ny;
    const tr = { margin: fx.M, cell: 0.05, nx, ny, terrain: fx.terrain, poses: fx.poses, bite: 0, pass: 1, t0: 0, soil: '', neural: false, r0: 0, groundY: 0, shift: 0, attitude: 'physical' };
    const r = await s.bite(tr, fx.soil);
    expect(r.delta.length).toBe(n);
    expect(Math.abs(r.carried - fx16.carried)).toBeLessThan(2e-3);
    // cells straddling the 1 cm display cutoff may flip; everything else within 2 mm
    let bad = 0;
    for (let k = 0; k < n; k++) if (Math.abs(r.delta[k] - fx16.delta[k]) > 0.002) bad++;
    expect(bad / n).toBeLessThan(0.002);
  }, 60000);
});
