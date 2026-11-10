/**
 * LiveDirt v3 learned soil: one inference per SCOOP (digbench/scoop_model.py, trained on MPM-in-the-loop bites).
 * v3: an ensemble (mean of the members' residual and scalars, as digbench/mpm_server.py does for a comma list of models).
 *
 * Input (bite's bed grid, 5 cm, window frame of the DigTrace, margin M around the 3.4 m x 2 m window):
 *   0 terrain - anchor, 1 swept envelope - terrain (clipped), 2 untouched mask, 3 cut+relax prior,
 *   4-6 soil planes (phi, cohesion, density).
 * Output: residual over the prior + scalars (carried bank m^3, completion, peak demand). The corrected surface is
 * projected through the same relax step, so no slope steeper than phi survives.
 * Port of digbench/scoop_model.py + mpm_server.run_learned; checked against a Python fixture in test/scoopSoil.test.ts.
 */
import type { DigTrace } from '../machine/autonomy';

export const CELL = 0.05;

export interface ScoopSoilParams { phi: number; c: number; rho: number; mu?: number }
export interface BucketGeom { TP: number[][]; LIP: [number, number] }
export interface ScoopResult { delta: number[]; nx: number; ny: number; cell: number; margin: number; carried: number; completion: number; peak_demand: number; removed_m3: number; added_m3: number; wall: number }

/** app soil preset -> MC parameters (digbench/soil_presets.py): phi = repose angle, cohesion from Culmann at the bank angle for a 1 m face */
export function presetParams(soil: { reposeDeg: number; bankDeg: number; density: number }): ScoopSoilParams {
  const rho = soil.density * 1000;
  let c = 0;
  if (soil.bankDeg > soil.reposeDeg) {
    const b = (soil.bankDeg * Math.PI) / 180, f = (soil.reposeDeg * Math.PI) / 180;
    c = (rho * 9.81 * 1.0 * (1 - Math.cos(b - f))) / (4 * Math.sin(b) * Math.cos(f));
  }
  return { phi: soil.reposeDeg, c: Math.round(c * 10) / 10, rho };
}

/** lowest bucket point per bed cell over the bite (window-frame z); NaN where the bucket never passes */
export function sweptEnvelope(poses: number[][], nx: number, ny: number, M: number, g: BucketGeom): Float64Array {
  const env = new Float64Array(nx * ny).fill(Infinity);
  const step = Math.max(1, Math.floor(poses.length / 120));
  const rows: number[][] = [];
  for (let k = 0; k < poses.length; k += step) rows.push(poses[k]);
  rows.push(poses[poses.length - 1]);
  const TP = g.TP;
  for (const row of rows) {
    const px = row[1], pz = row[2], pa = row[3], ro = row.length > 5 ? row[5] : 0;
    const ca = Math.cos(pa), sa = Math.sin(pa);
    let lz = 0, cr = 1, sr = 0;
    if (ro !== 0) { lz = sa * g.LIP[0] + ca * g.LIP[1]; cr = Math.cos(ro); sr = Math.sin(ro); }
    for (let q = 0; q < TP.length; q++) {
      const tx = TP[q][0], ty = TP[q][1], tz = TP[q][2];
      const x = ca * tx - sa * tz;
      let z = sa * tx + ca * tz, y = ty;
      if (ro !== 0) { const yy = y, zz = z - lz; y = cr * yy - sr * zz; z = sr * yy + cr * zz + lz; }
      const i = Math.floor((px + x + M) / CELL), j = Math.floor((y + 1.0 + M) / CELL);
      if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
      const k = i * ny + j, wz = pz + z;
      if (wz < env[k]) env[k] = wz;
    }
  }
  for (let k = 0; k < env.length; k++) if (!Number.isFinite(env[k])) env[k] = NaN;
  return env;
}

/** mass-conserving slope relaxation (numpy port): neighbour differences steeper than tan(repose) * cell move a quarter of the excess */
export function relax(H: Float64Array, nx: number, ny: number, reposeDeg: number, iters = 200): Float64Array {
  const h = Float64Array.from(H);
  const lim = Math.tan((reposeDeg * Math.PI) / 180) * CELL;
  const exA = new Float64Array((nx - 1) * ny), exB = new Float64Array(nx * (ny - 1));
  for (let it = 0; it < iters; it++) {
    let moved = 0;
    // axis 0 (i)
    let any = false;
    for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny; j++) {
      const d = h[(i + 1) * ny + j] - h[i * ny + j], a = Math.abs(d);
      const e = a > lim ? (a - lim) * Math.sign(d) * 0.25 : 0;
      exA[i * ny + j] = e; if (e !== 0) any = true;
    }
    if (any) {
      for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny; j++) { const e = exA[i * ny + j]; if (e) { h[i * ny + j] += e; h[(i + 1) * ny + j] -= e; moved += Math.abs(e); } }
    }
    any = false;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny - 1; j++) {
      const d = h[i * ny + j + 1] - h[i * ny + j], a = Math.abs(d);
      const e = a > lim ? (a - lim) * Math.sign(d) * 0.25 : 0;
      exB[i * (ny - 1) + j] = e; if (e !== 0) any = true;
    }
    if (any) {
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny - 1; j++) { const e = exB[i * (ny - 1) + j]; if (e) { h[i * ny + j] += e; h[i * ny + j + 1] -= e; moved += Math.abs(e); } }
    }
    if (moved < 1e-6) break;
  }
  return h;
}

/** physics prior: soil above the envelope is cut (bucket capacity cap), surface relaxed; kept within 0.6 m of swept cells */
export function cutRelaxDelta(H: Float64Array, env: Float64Array, nx: number, ny: number, reposeDeg: number, cap = 0.95): { d: Float64Array; vol: number } {
  const n = nx * ny, cut = new Float64Array(n);
  let vol = 0;
  for (let k = 0; k < n; k++) { if (!Number.isNaN(env[k])) cut[k] = Math.max(0, H[k] - env[k]); vol += cut[k]; }
  vol *= CELL * CELL;
  if (vol > cap) { const s = cap / vol; for (let k = 0; k < n; k++) cut[k] *= s; vol = cap; }
  const H0 = new Float64Array(n);
  for (let k = 0; k < n; k++) H0[k] = H[k] - cut[k];
  const H1 = relax(H0, nx, ny, reposeDeg);
  // dilation with wrap-around, exactly like np.roll in the Python reference
  let near = new Uint8Array(n);
  for (let k = 0; k < n; k++) near[k] = Number.isNaN(env[k]) ? 0 : 1;
  for (let s = 0; s < Math.floor(0.6 / CELL); s++) {
    const nn = new Uint8Array(n);
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const k = i * ny + j;
      nn[k] = near[k] | near[((i + 1) % nx) * ny + j] | near[((i - 1 + nx) % nx) * ny + j] | near[i * ny + ((j + 1) % ny)] | near[i * ny + ((j - 1 + ny) % ny)];
    }
    near = nn;
  }
  const d = new Float64Array(n);
  for (let k = 0; k < n; k++) d[k] = near[k] ? H1[k] - H[k] : 0;
  return { d, vol };
}

export function makeInputs(terrain: Float64Array, poses: number[][], soil: ScoopSoilParams, nx: number, ny: number, M: number, g: BucketGeom) {
  const n = nx * ny;
  const env = sweptEnvelope(poses, nx, ny, M, g);
  let s = 0, cnt = 0, sAll = 0;
  for (let k = 0; k < n; k++) { sAll += terrain[k]; if (!Number.isNaN(env[k])) { s += terrain[k]; cnt++; } }
  const anchor = cnt ? s / cnt : sAll / n;
  const { d: prior, vol } = cutRelaxDelta(terrain, env, nx, ny, soil.phi);
  const X = new Float32Array(7 * n);
  const planes = [(soil.phi - 30) / 10, Math.log10(1 + soil.c) / 3, (soil.rho - 1800) / 200];
  for (let k = 0; k < n; k++) {
    const touched = !Number.isNaN(env[k]);
    X[k] = terrain[k] - anchor;
    X[n + k] = touched ? Math.min(0.5, Math.max(-1.5, env[k] - terrain[k])) : 0.5;
    X[2 * n + k] = touched ? 0 : 1;
    X[3 * n + k] = prior[k];
    for (let p = 0; p < 3; p++) X[(4 + p) * n + k] = planes[p];
  }
  return { X, prior, priorVol: vol, env };
}

/** edge-replicating pad of [C, nx, ny] to multiples of 8 (np.pad mode='edge') */
export function pad8(X: Float32Array, C: number, nx: number, ny: number) {
  const px = nx + ((8 - (nx % 8)) % 8), py = ny + ((8 - (ny % 8)) % 8);
  const out = new Float32Array(C * px * py);
  for (let c = 0; c < C; c++) for (let i = 0; i < px; i++) for (let j = 0; j < py; j++)
    out[(c * px + i) * py + j] = X[(c * nx + Math.min(i, nx - 1)) * ny + Math.min(j, ny - 1)];
  return { data: out, px, py };
}

type Ort = { Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown; InferenceSession: { create(u: string | Uint8Array, o?: unknown): Promise<{ run(f: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>> }> } };

type Session = { run(f: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>> };

export class ScoopSoil {
  private sessions: Session[];
  constructor(private ort: Ort, session: Session | Session[], readonly geom: BucketGeom) { this.sessions = Array.isArray(session) ? session : [session]; }
  /** modelUrl: one ONNX file, or a list = ensemble (members averaged) */
  static async load(ort: Ort, modelUrl: string | string[], bucketUrl: string, opts?: unknown) {
    const urls = Array.isArray(modelUrl) ? modelUrl : [modelUrl];
    const [sessions, geom] = await Promise.all([Promise.all(urls.map((u) => ort.InferenceSession.create(u, opts))), fetch(bucketUrl).then((r) => r.json())]);
    return new ScoopSoil(ort, sessions, geom as BucketGeom);
  }
  static fromParts(ort: Ort, session: Session | Session[], geom: BucketGeom) { return new ScoopSoil(ort, session, geom); }
  get members() { return this.sessions.length; }

  /** run one bite: trace terrain (window frame) + poses -> surface delta on the trace grid + carried volume */
  async bite(tr: DigTrace, soil: ScoopSoilParams, minDelta = 0.01): Promise<ScoopResult> {
    const t0 = performance.now();
    const nx = tr.nx, ny = tr.ny, M = tr.margin, n = nx * ny;
    const T = Float64Array.from(tr.terrain);
    const { X, prior } = makeInputs(T, tr.poses, soil, nx, ny, M, this.geom);
    const { data, px, py } = pad8(X, 7, nx, ny);
    const r = new Float64Array(px * py), ys = new Float64Array(3);
    for (const sess of this.sessions) {  // sequential: each member is ~8 MB of weights; wasm runs one at a time anyway
      const out = await sess.run({ x: new this.ort.Tensor('float32', data, [1, 7, px, py]) });
      const rr = out.residual.data, yy = out.scalars.data;
      for (let k = 0; k < r.length; k++) r[k] += rr[k] / this.sessions.length;
      for (let k = 0; k < 3; k++) ys[k] += yy[k] / this.sessions.length;
    }
    const H1 = new Float64Array(n);
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) { const k = i * ny + j; H1[k] = T[k] + prior[k] + r[i * py + j]; }
    const H2 = relax(H1, nx, ny, soil.phi);
    const delta = new Array<number>(n);
    let removed = 0, added = 0;
    for (let k = 0; k < n; k++) {
      let d = H2[k] - T[k];
      if (Math.abs(d) < minDelta) d = 0;
      delta[k] = d;
      if (d < 0) removed -= d; else added += d;
    }
    return { delta, nx, ny, cell: CELL, margin: M, carried: Math.max(0, ys[0]), completion: ys[1], peak_demand: ys[2], removed_m3: removed * CELL * CELL, added_m3: added * CELL * CELL, wall: (performance.now() - t0) / 1000 };
  }
}
