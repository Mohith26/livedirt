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
  /** run one 0.1 s step: features (6 x NX x NY) -> dh (NX x NY, m), force [Fx, Fz, Ty] */
  step(features: Float32Array): Promise<{ dh: Float32Array; force: [number, number, number] }>;
}

/** onnxruntime (web or node) wrapper; the exported graph takes x: [1, 6, NX, NY] */
export async function loadNeuralModel(ort: any, url: string, opts: Record<string, unknown> = {}): Promise<NeuralModel> {
  const session = await ort.InferenceSession.create(url, opts);
  return {
    async step(features: Float32Array) {
      const res = await session.run({ x: new ort.Tensor('float32', features, [1, 6, NX, NY]) });
      return { dh: res.dh.data as Float32Array, force: Array.from(res.force.data as Float32Array).slice(0, 3) as [number, number, number] };
    },
  };
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
    this.carried = 0;
  }

  /** advance 0.1 s; pivot/pitch are the bucket pose at the END of the step (window coords, metres, radians) */
  async step(pivotX: number, pivotZ: number, pitch: number) {
    const T1 = rasterizeBucket(pivotX, pivotZ + this.shift, pitch);
    const feats = buildFeatures(this.H, this.T0, T1, this.mu);
    const { dh, force } = await this.model.step(feats);
    let dv = 0;
    for (let k = 0; k < this.H.length; k++) {
      this.H[k] += dh[k];
      dv += dh[k];
    }
    this.carried -= dv * CELL * CELL;
    this.T0 = T1;
    this.lastForce = force;
    return { force, carried: this.carried };
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
