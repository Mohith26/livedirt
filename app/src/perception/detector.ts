import * as ort from 'onnxruntime-web/wasm';

/**
 * YOLOX (Megvii, Apache-2.0) COCO detector running in the browser via onnxruntime-web (WASM backend).
 * The released ONNX models take a 0-255 BGR CHW image letterboxed with value 114 and output
 * [1, A, 85] = (dx, dy, log w, log h, obj, 80 class scores) per anchor with sigmoids applied but grid
 * decoding left to the caller (strides 8/16/32).
 */
export interface Detection { x: number; y: number; w: number; h: number; score: number; cls: number }

export const COCO_PERSON = 0, COCO_CAR = 2, COCO_TRUCK = 7;

export class YoloxDetector {
  private session: ort.InferenceSession | null = null;
  readonly size: number;
  lastMs = 0;
  private grids: { gx: Float32Array; gy: Float32Array; stride: Float32Array } | null = null;

  constructor(size = 416) { this.size = size; }

  get ready() { return !!this.session; }

  async load(modelUrl: string, wasmBase = '/ort/') {
    ort.env.wasm.wasmPaths = wasmBase;
    ort.env.wasm.numThreads = (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    this.session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    // precompute anchor grids
    const strides = [8, 16, 32];
    let n = 0;
    for (const s of strides) n += (this.size / s) ** 2;
    const gx = new Float32Array(n), gy = new Float32Array(n), st = new Float32Array(n);
    let i = 0;
    for (const s of strides) {
      const g = this.size / s;
      for (let y = 0; y < g; y++) for (let x = 0; x < g; x++) { gx[i] = x; gy[i] = y; st[i] = s; i++; }
    }
    this.grids = { gx, gy, stride: st };
  }

  /** run on an RGBA image (top row first). Returns detections in source pixel coordinates. */
  async detect(rgba: Uint8Array, w: number, h: number, classes: number[] = [COCO_PERSON], minScore = 0.25): Promise<Detection[]> {
    if (!this.session || !this.grids) throw new Error('detector not loaded');
    const t0 = performance.now();
    const S = this.size;
    const r = Math.min(S / w, S / h);
    const nw = Math.round(w * r), nh = Math.round(h * r);
    const input = new Float32Array(3 * S * S).fill(114);
    // nearest-neighbour resize into the letterbox (top-left aligned, as in the reference preprocess)
    for (let y = 0; y < nh; y++) {
      const sy = Math.min(h - 1, Math.floor(y / r));
      for (let x = 0; x < nw; x++) {
        const sx = Math.min(w - 1, Math.floor(x / r));
        const o = (sy * w + sx) * 4;
        const p = y * S + x;
        input[p] = rgba[o + 2]; // B
        input[S * S + p] = rgba[o + 1]; // G
        input[2 * S * S + p] = rgba[o]; // R
      }
    }
    const feeds: Record<string, ort.Tensor> = { [this.session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, S, S]) };
    const out = await this.session.run(feeds);
    const data = out[this.session.outputNames[0]].data as Float32Array;
    const A = this.grids.gx.length, C = 85;
    const dets: Detection[] = [];
    for (let a = 0; a < A; a++) {
      const o = a * C;
      const obj = data[o + 4];
      if (obj < minScore * 0.5) continue;
      for (const c of classes) {
        const score = obj * data[o + 5 + c];
        if (score < minScore) continue;
        const s = this.grids.stride[a];
        const cx = (data[o] + this.grids.gx[a]) * s, cy = (data[o + 1] + this.grids.gy[a]) * s;
        const bw = Math.exp(data[o + 2]) * s, bh = Math.exp(data[o + 3]) * s;
        dets.push({ x: (cx - bw / 2) / r, y: (cy - bh / 2) / r, w: bw / r, h: bh / r, score, cls: c });
      }
    }
    this.lastMs = performance.now() - t0;
    return nms(dets, 0.45);
  }
}

export function iou(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }) {
  const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y), x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-9);
}

export function nms(d: Detection[], thr: number): Detection[] {
  d.sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];
  for (const x of d) if (!keep.some((k) => k.cls === x.cls && iou(k, x) > thr)) keep.push(x);
  return keep;
}
