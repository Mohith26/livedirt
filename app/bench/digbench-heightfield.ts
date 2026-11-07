/**
 * DigBench task 1 (column collapse) run on LiveDirt's production heightfield (src/sim/heightfield.ts),
 * unmodified. The column is loose material in a 1-cell-per-row quasi-2D channel: masked side walls,
 * masked back wall at x = 0, flat floor. Repose angle = the measured glass-bead value (23.5 deg),
 * the same material input given to the MPM solver.
 *
 *   npx tsx bench/digbench-heightfield.ts [cell=0.0025]
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { Heightfield } from '../src/sim/heightfield';
import type { SoilType } from '../src/sim/soil';

const CASES = [
  { a: 0.5, H0: 0.1, L0: 0.2 },
  { a: 1.0, H0: 0.15, L0: 0.15 },
  { a: 2.0, H0: 0.2, L0: 0.1 },
  { a: 3.0, H0: 0.3, L0: 0.1 },
  { a: 5.0, H0: 0.375, L0: 0.075 },
  { a: 8.0, H0: 0.4, L0: 0.05 },
  { a: 12.0, H0: 0.45, L0: 0.0375 },
];
const runoutRef = (a: number) => (a <= 3 ? 1.85 * a : 2.67 * a ** (2 / 3));
const heightRef = (a: number) => (a <= 0.8 ? a : 0.93 * a ** (1 / 3));
const GRAIN_D = 0.005;
const cell = Number(process.argv[2] ?? 0.0025);
const soil: SoilType = { id: 'glass-beads', name: 'glass beads (Sarlin 2021)', swell: 1, reposeDeg: 23.5, bankDeg: 89 } as SoilType;

const rows: any[] = [];
for (const c of CASES) {
  const len = c.L0 + 3.2 * c.H0 + 0.05;
  const N = Math.ceil(len / cell) + 4;
  const half = (N * cell) / 2;
  const H0 = new Float32Array(N * N);
  const mask = new Uint8Array(N * N);
  const row0 = Math.floor(N / 2) - 1; // channel rows row0..row0+2 (3 cells wide)
  const wallI = 2;
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const k = j * N + i;
      if (j < row0 || j > row0 + 2 || i < wallI || i === N - 1) mask[k] = 1;
    }
  const hf = new Heightfield({ N, cell, half, H0, mask, soil });
  const nCol = Math.round(c.L0 / cell);
  for (let j = row0; j <= row0 + 2; j++) for (let i = wallI; i < wallI + nCol; i++) hf.L[j * N + i] = c.H0;
  (hf as any).active = { i0: 0, j0: 0, i1: N - 1, j1: N - 1 };
  const vol0 = hf.L.reduce((s, v) => s + v, 0);
  const t0 = performance.now();
  let steps = 0;
  while (hf.isSettling() && steps < 2_000_000) {
    hf.relax(2);
    steps++;
  }
  const wall = (performance.now() - t0) / 1000;
  const vol1 = hf.L.reduce((s, v) => s + v, 0);
  const j = row0 + 1;
  const h: number[] = [];
  for (let i = wallI; i < N - 1; i++) h.push(hf.L[j * N + i]);
  const Hf = Math.max(...h);
  let n = 0;
  while (n < h.length && h[n] >= GRAIN_D) n++;
  const Lf = n * cell;
  const r = (Lf - c.L0) / c.L0, hh = Hf / c.L0;
  const row = {
    ...c, Hf, Lf, dL: Lf - c.L0,
    runout_sim: r, runout_ref: runoutRef(c.a), runout_relerr: (r - runoutRef(c.a)) / runoutRef(c.a),
    height_sim: hh, height_ref: heightRef(c.a), height_relerr: (hh - heightRef(c.a)) / heightRef(c.a),
    relax_calls: steps, wall_s: wall, settled: !hf.isSettling(), mass_rel_change: (vol1 - vol0) / vol0,
    profile: { x: h.map((_, i) => +((i + 0.5) * cell).toFixed(5)).filter((_, i) => i % 2 === 0), h: h.map((v) => +v.toFixed(5)).filter((_, i) => i % 2 === 0) },
  };
  rows.push(row);
  console.log(`a=${c.a}: runout ${r.toFixed(2)} vs ${row.runout_ref.toFixed(2)} (${(row.runout_relerr * 100).toFixed(1)}%)  height ${hh.toFixed(2)} vs ${row.height_ref.toFixed(2)} (${(row.height_relerr * 100).toFixed(1)}%)  relax=${steps} wall=${wall.toFixed(2)}s mass ${(row.mass_rel_change * 100).toExponential(1)}%`);
}
const mae = (k: string) => rows.reduce((s, r) => s + Math.abs(r[k]), 0) / rows.length;
const out = {
  task: 'column_collapse', solver: 'livedirt-heightfield', tag: `heightfield-${cell}`,
  reference: 'Sarlin et al. 2021, Phys. Rev. E 104, 064904 (arXiv:2112.11458), Series A fits',
  params: { cell, reposeDeg: 23.5, channelCells: 3, neighbors: 8 },
  summary: { runout_mae_rel: mae('runout_relerr'), height_mae_rel: mae('height_relerr') },
  cases: rows,
};
mkdirSync('../digbench/results', { recursive: true });
writeFileSync(`../digbench/results/collapse-heightfield.json`, JSON.stringify(out, null, 1));
console.log(`summary: runout MAE ${(out.summary.runout_mae_rel * 100).toFixed(1)}%, height MAE ${(out.summary.height_mae_rel * 100).toFixed(1)}%`);
