/**
 * Neural soil: the DigBench task-6 surrogate (U-Net trained on 3D MPM dig cycles) running on a moving window of
 * LiveDirt's heightfield. Replaces the kinematic cut (dig cones + relax) inside the window while the bucket digs.
 *
 * Window frame (matches digbench/gen_surrogate_data.py): x along the radial direction from the machine (bucket
 * drags toward -x), y lateral (bucket centreline at y = 0), z up. 68 x 40 cells at 5 cm = 3.4 m x 2.0 m.
 * The model's training bed was 1.2 m deep with its surface at z = 1.2; window heights are shifted so the ground at
 * anchoring sits at 1.2 m.
 *
 * Inputs per step (channels, see train_surrogate.features): (H - 1.2) * 2, bucket penetration now / next
 * (T - H clamped to [-1, 1], 1 where no bucket), bucket occupancy now / next, (mu - 0.5) * 4.
 * Outputs: dh (1 x 1 x NX x NY, metres over 0.1 s), force (1 x 3: Fx, Fz, Ty in N, soil on bucket, full width).
 */
import type { Heightfield } from './heightfield';

export const NX = 68;
export const NY = 40;
export const CELL = 0.05;
export const SURF = 1.2;
export const STEP_S = 0.1;

/** bucket shell boxes in the tool frame (pivot at origin): [centre x, y, z], [half x, y, z]; matches run_bucket.bucket_boxes */
const T = 0.05;
const BOXES: [number[], number[]][] = [
  [[T, 0, -0.45], [T, 0.5 + 2 * T, 0.45 + 2 * T]],
  [[-0.45, 0, -0.9 - T], [0.45, 0.5 + 2 * T, T]],
  [[-0.45, 0.5 + T, -0.45], [0.45, T, 0.45]],
  [[-0.45, -0.5 - T, -0.45], [0.45, T, 0.45]],
  [[-0.45, 0, T], [0.45, 0.5 + 2 * T, T]],
];

let toolPts: Float32Array | null = null;
function toolPoints(n = 14): Float32Array {
  if (toolPts) return toolPts;
  const out: number[] = [];
  for (const [c, h] of BOXES)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        for (let k = 0; k < n; k++) {
          const u = -1 + (2 * i) / (n - 1), v = -1 + (2 * j) / (n - 1), w = -1 + (2 * k) / (n - 1);
          out.push(c[0] + h[0] * u, c[1] + h[1] * v, c[2] + h[2] * w);
        }
  toolPts = Float32Array.from(out);
  return toolPts;
}

/** lowest bucket-solid elevation per window cell (NaN where the bucket is not over the cell) */
export function rasterizeBucket(pivotX: number, pivotZ: number, pitch: number, out: Float32Array = new Float32Array(NX * NY)): Float32Array {
  out.fill(Infinity);
  const P = toolPoints();
  const ca = Math.cos(pitch), sa = Math.sin(pitch);
  for (let q = 0; q < P.length; q += 3) {
    const lx = P[q], ly = P[q + 1], lz = P[q + 2];
    const wx = pivotX + ca * lx - sa * lz, wz = pivotZ + sa * lx + ca * lz;
    const i = Math.floor(wx / CELL), j = Math.floor((ly + 1.0) / CELL);
    if (i < 0 || i >= NX || j < 0 || j >= NY) continue;
    const k = i * NY + j;
    if (wz < out[k]) out[k] = wz;
  }
  for (let k = 0; k < out.length; k++) if (!Number.isFinite(out[k])) out[k] = NaN;
  return out;
}

export function buildFeatures(H: Float32Array, T0: Float32Array, T1: Float32Array, mu: number, out: Float32Array = new Float32Array(6 * NX * NY)): Float32Array {
  const n = NX * NY;
  for (let k = 0; k < n; k++) {
    const h = H[k];
    out[k] = (h - SURF) * 2;
    const a = T0[k], b = T1[k];
    const fa = Number.isFinite(a), fb = Number.isFinite(b);
    out[n + k] = fa ? Math.max(-1, Math.min(1, a - h)) : 1;
    out[2 * n + k] = fa ? 1 : 0;
    out[3 * n + k] = fb ? Math.max(-1, Math.min(1, b - h)) : 1;
    out[4 * n + k] = fb ? 1 : 0;
    out[5 * n + k] = (mu - 0.5) * 4;
  }
  return out;
}

export interface NeuralModel {
  /** run one 0.1 s step: features (6 x NX x NY) -> dh (NX x NY, m), force [Fx, Fz, Ty]; ensembles also return the
   * per-cell standard deviation of dh across members */
  step(features: Float32Array): Promise<{ dh: Float32Array; force: [number, number, number]; dhStd?: Float32Array }>;
  members?: number;
}

/** onnxruntime (web or node) wrapper; the exported graph takes x: [1, 6, NX, NY] */
export async function loadNeuralModel(ort: any, url: string, opts: Record<string, unknown> = {}): Promise<NeuralModel> {
  const session = await ort.InferenceSession.create(url, opts);
  return {
    members: 1,
    async step(features: Float32Array) {
      const res = await session.run({ x: new ort.Tensor('float32', features, [1, 6, NX, NY]) });
      return { dh: res.dh.data as Float32Array, force: Array.from(res.force.data as Float32Array).slice(0, 3) as [number, number, number] };
    },
  };
}

/** ensemble of independently trained members: mean dh / force and per-cell dh spread (uncertainty for gating) */
export async function loadEnsemble(ort: any, urls: string[], opts: Record<string, unknown> = {}): Promise<NeuralModel> {
  const ms = await Promise.all(urls.map((u) => loadNeuralModel(ort, u, opts)));
  return {
    members: ms.length,
    async step(features: Float32Array) {
      const outs: { dh: Float32Array; force: [number, number, number] }[] = [];
      for (const m of ms) outs.push(await m.step(features)); // sequential: onnxruntime sessions are not re-entrant
      const n = NX * NY, k = outs.length;
      const dh = new Float32Array(n), sd = new Float32Array(n);
      for (const o of outs) for (let c = 0; c < n; c++) dh[c] += o.dh[c] / k;
      for (const o of outs) for (let c = 0; c < n; c++) { const d = o.dh[c] - dh[c]; sd[c] += (d * d) / k; }
      for (let c = 0; c < n; c++) sd[c] = Math.sqrt(sd[c]);
      const force = [0, 1, 2].map((i) => outs.reduce((a, o) => a + o.force[i], 0) / k) as [number, number, number];
      return { dh, force, dhStd: sd };
    },
  };
}

/**
 * Uncertainty gate. A step is handed to cut+relax instead of the network when (a) more than maxOOR of the bucket-near
 * cells have an input channel outside the training range [lo, hi] (per channel; from the trainer's percentiles), or
 * (b) the ensemble's mean dh spread over bucket-near cells exceeds maxStd (m per 0.1 s).
 */
export interface NeuralGate {
  ranges?: [number, number][]; maxOOR: number; maxStd: number;
  /** physical consistency (a priori, not tuned): fall back when this window would have removed more than maxCarried
   * m^3 (set per step from the bucket's remaining heaped capacity x 1.2) or created more than maxCreate m^3 */
  maxCarried?: number; maxCreate?: number;
}

/** loop-side samples of each input channel (for comparing deployment inputs to training percentiles) */
export class FeatureStats {
  samples: number[][] = [[], [], [], [], [], []];
  oorSteps = 0; capSteps = 0; steps = 0; stdSum = 0; stdMax = 0;
  private seed = 7;
  add(f: Float32Array) {
    const n = NX * NY;
    this.steps++;
    for (let s = 0; s < 48; s++) {
      this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
      const k = this.seed % n;
      for (let c = 0; c < 6; c++) if (c === 0 || c === 5 || f[(c >= 3 ? 4 : 2) * n + k] > 0.5 || c === 2 || c === 4) this.samples[c].push(f[c * n + k]);
    }
  }
  summary(ranges?: [number, number][]) {
    const q = (a: number[], p: number) => { if (!a.length) return NaN; const b = a.slice().sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * b.length))]; };
    const names = ['H', 'pen0', 'occ0', 'pen1', 'occ1', 'mu'];
    return names.map((nm, c) => {
      const a = this.samples[c];
      const r = ranges?.[c];
      return { channel: nm, n: a.length, p01: q(a, 0.01), p50: q(a, 0.5), p99: q(a, 0.99), min: q(a, 0), max: q(a, 1),
        fracOutsideTrain: r ? a.filter((v) => v < r[0] || v > r[1]).length / Math.max(1, a.length) : null };
    });
  }
}

/**
 * Moving window bound to a LiveDirt heightfield. Anchor it at the start of a dig cycle (origin = radial position of
 * the window's x = 0, swing = machine swing), then call step() every 0.1 s with the bucket pose in window coordinates.
 */
export class NeuralSoilWindow {
  H = new Float32Array(NX * NY);
  private H0 = new Float32Array(NX * NY);
  private shift = 0;
  private T0: Float32Array = new Float32Array(NX * NY);
  carried = 0; // bank m^3 removed from the window terrain (positive into the bucket)
  lastForce: [number, number, number] = [0, 0, 0];
  clipped = 0; // count of clipped/non-finite cell updates (out-of-distribution indicator)
  floored = 0; // count of cell updates raised to the swept floor
  floorAddedM3 = 0; // volume created by the floor projection
  modelDvM3 = 0; // net volume change from the model's dh (after clipping, before the projection)
  stats: FeatureStats | null = null;
  /** physical projection: no cell may end below min(its anchored height, the lowest bucket point that passed over it) */
  floorProject = true;
  private floor = new Float32Array(NX * NY);
  constructor(private hf: Heightfield, private model: NeuralModel, public mu: number,
              private cx: number, private cz: number, private swing: number, private r0: number) {}

  /** world (x, z) of window cell centre (i, j) */
  private world(i: number, j: number): [number, number] {
    const r = this.r0 + (i + 0.5) * CELL, lat = -1.0 + (j + 0.5) * CELL;
    const ux = Math.cos(this.swing), uz = Math.sin(this.swing); // radial
    const vx = -uz, vz = ux; // lateral
    return [this.cx + ux * r + vx * lat, this.cz + uz * r + vz * lat];
  }

  anchor(pivotX: number, pivotZ: number, pitch: number) {
    let sum = 0;
    for (let i = 0; i < NX; i++)
      for (let j = 0; j < NY; j++) {
        const [x, z] = this.world(i, j);
        const h = this.hf.heightAt(x, z);
        this.H0[i * NY + j] = h;
        sum += h;
      }
    this.shift = SURF - sum / (NX * NY);
    for (let k = 0; k < this.H.length; k++) this.H[k] = this.H0[k] + this.shift;
    rasterizeBucket(pivotX, pivotZ + this.shift, pitch, this.T0);
    this.floor.set(this.H);
    for (let k = 0; k < this.floor.length; k++) if (Number.isFinite(this.T0[k]) && this.T0[k] < this.floor[k]) this.floor[k] = this.T0[k];
    this.carried = 0;
  }

  /** advance 0.1 s; pivot/pitch are the bucket pose at the END of the step (window coords, metres, radians).
   * With a gate, returns gated=true WITHOUT changing the window when the step should go to cut+relax. */
  async step(pivotX: number, pivotZ: number, pitch: number, gate?: NeuralGate) {
    const T1 = rasterizeBucket(pivotX, pivotZ + this.shift, pitch);
    const feats = buildFeatures(this.H, this.T0, T1, this.mu);
    this.stats?.add(feats);
    const n = NX * NY;
    let near: Uint8Array | null = null;
    if (gate) {
      near = new Uint8Array(n);
      for (let k = 0; k < n; k++) if (feats[2 * n + k] > 0.5 || feats[4 * n + k] > 0.5) {
        const i = Math.floor(k / NY), j = k % NY;
        for (let di = -2; di <= 2; di++) for (let dj = -2; dj <= 2; dj++) { const a = i + di, b = j + dj; if (a >= 0 && a < NX && b >= 0 && b < NY) near[a * NY + b] = 1; }
      }
      if (gate.ranges) {
        let cnt = 0, bad = 0;
        for (let k = 0; k < n; k++) {
          if (!near[k]) continue;
          cnt++;
          for (const c of [0, 1, 3]) { const v = feats[c * n + k], r = gate.ranges[c]; if (v < r[0] || v > r[1]) { bad++; break; } }
        }
        if (cnt && bad / cnt > gate.maxOOR) { if (this.stats) this.stats.oorSteps++; return { force: this.lastForce, carried: this.carried, gated: true, reason: 'oor' }; }
      }
    }
    const { dh, force: raw, dhStd } = await this.model.step(feats);
    if (gate && dhStd && near) {
      let sum = 0, cnt = 0;
      for (let k = 0; k < n; k++) if (near[k]) { sum += dhStd[k]; cnt++; }
      const m = cnt ? sum / cnt : 0;
      if (this.stats) { this.stats.stdSum += m; this.stats.stdMax = Math.max(this.stats.stdMax, m); }
      if (m > gate.maxStd) return { force: this.lastForce, carried: this.carried, gated: true, reason: 'std' };
    }
    if (gate && (gate.maxCarried !== undefined || gate.maxCreate !== undefined)) {
      let dvp = 0;
      for (let k = 0; k < n; k++) { const d = dh[k]; if (Number.isFinite(d)) dvp += Math.max(-0.3, Math.min(0.3, d)); }
      const carriedNext = this.carried - dvp * CELL * CELL;
      if (gate.maxCarried !== undefined && carriedNext > gate.maxCarried) { if (this.stats) this.stats.capSteps++; return { force: this.lastForce, carried: this.carried, gated: true, reason: 'cap' }; }
      if (gate.maxCreate !== undefined && -carriedNext > gate.maxCreate) { if (this.stats) this.stats.capSteps++; return { force: this.lastForce, carried: this.carried, gated: true, reason: 'create' }; }
    }
    for (let k = 0; k < T1.length; k++) if (Number.isFinite(T1[k]) && T1[k] < this.floor[k]) this.floor[k] = T1[k];
    const force = raw.map((f) => (Number.isFinite(f) ? f : 0)) as [number, number, number];
    // guards: the model is only trusted inside its training range; non-finite or extreme steps are clipped and counted
    let dv = 0;
    for (let k = 0; k < this.H.length; k++) {
      let d = dh[k];
      if (!Number.isFinite(d)) { d = 0; this.clipped++; }
      else if (d > 0.3 || d < -0.3) { d = Math.max(-0.3, Math.min(0.3, d)); this.clipped++; }
      let h = Math.max(SURF - 2.0, Math.min(SURF + 1.5, this.H[k] + d));
      this.modelDvM3 += (h - this.H[k]) * CELL * CELL;
      if (this.floorProject && h < this.floor[k]) { const h2 = Math.max(h, Math.min(this.floor[k], this.H[k])); this.floorAddedM3 += (h2 - h) * CELL * CELL; h = h2; this.floored++; }
      dv += h - this.H[k];
      this.H[k] = h;
    }
    this.carried -= dv * CELL * CELL;
    this.T0 = T1;
    this.lastForce = force;
    return { force, carried: this.carried, gated: false, reason: '' };
  }

  /** write the window terrain back into the heightfield (loose layer absorbs changes; bank only if cut below it) */
  commit() {
    for (let i = 0; i < NX; i++)
      for (let j = 0; j < NY; j++) {
        const [x, z] = this.world(i, j);
        const [ci, cj] = this.hf.cellOf(x, z);
        if (!this.hf.inGrid(ci, cj)) continue;
        const k = cj * this.hf.N + ci;
        const target = this.H[i * NY + j] - this.shift;
        const cur = this.hf.B[k] + this.hf.L[k];
        const d = target - cur;
        if (d >= 0) this.hf.L[k] += d;
        else {
          const fromL = Math.min(this.hf.L[k], -d);
          this.hf.L[k] -= fromL;
          this.hf.B[k] -= -d - fromL;
        }
      }
    this.hf.markAllDirty();
  }
}
