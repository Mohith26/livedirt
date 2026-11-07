/**
 * Ray / heightfield intersection with a maximum-height mip pyramid.
 *
 * The surface is the bilinear interpolation of cell-centred samples, i.e. patches spanning the "dual grid":
 * patch (i, j) has corners at cell centres (i, j), (i+1, j), (i, j+1), (i+1, j+1). The pyramid stores the
 * maximum corner height per patch block so rays can skip whole blocks they pass above.
 */
export interface RayHit { t: number; x: number; y: number; z: number; nx: number; ny: number; nz: number }

export class HeightRaycaster {
  readonly N: number;
  readonly cell: number;
  readonly half: number;
  /** surface heights, cell-centred, length N*N */
  surf: Float32Array;
  private levels: Float32Array[] = [];
  private sizes: number[] = [];

  constructor(N: number, cell: number, half: number, surf?: Float32Array) {
    this.N = N;
    this.cell = cell;
    this.half = half;
    this.surf = surf ?? new Float32Array(N * N);
    this.rebuild();
  }

  /** rebuild the max pyramid (call after the surface changes) */
  rebuild() {
    const P = this.N - 1; // patches per side
    let size = P;
    const l0 = new Float32Array(P * P);
    const s = this.surf, N = this.N;
    for (let j = 0; j < P; j++)
      for (let i = 0; i < P; i++) {
        const k = j * N + i;
        l0[j * P + i] = Math.max(s[k], s[k + 1], s[k + N], s[k + N + 1]);
      }
    this.levels = [l0];
    this.sizes = [P];
    while (size > 1) {
      const prev = this.levels[this.levels.length - 1];
      const ps = size;
      size = Math.ceil(size / 2);
      const cur = new Float32Array(size * size).fill(-Infinity);
      for (let j = 0; j < ps; j++)
        for (let i = 0; i < ps; i++) {
          const v = prev[j * ps + i];
          const k = (j >> 1) * size + (i >> 1);
          if (v > cur[k]) cur[k] = v;
        }
      this.levels.push(cur);
      this.sizes.push(size);
    }
  }

  /** bilinear height at world (x,z) */
  height(x: number, z: number): number {
    const N = this.N;
    const fx = (x + this.half) / this.cell - 0.5, fz = (z + this.half) / this.cell - 0.5;
    const i0 = Math.max(0, Math.min(N - 2, Math.floor(fx))), j0 = Math.max(0, Math.min(N - 2, Math.floor(fz)));
    const tx = Math.max(0, Math.min(1, fx - i0)), tz = Math.max(0, Math.min(1, fz - j0));
    const s = this.surf, k = j0 * N + i0;
    return (s[k] * (1 - tx) + s[k + 1] * tx) * (1 - tz) + (s[k + N] * (1 - tx) + s[k + N + 1] * tx) * tz;
  }

  /**
   * Cast a ray (unit direction). Returns the first hit within [tMin, tMax] or null.
   * Uses hierarchical skipping: at each step, find the coarsest level whose block max is below the ray's
   * minimum height across that block, and jump to the block exit.
   */
  cast(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, tMax: number, tMin = 0): RayHit | null {
    const c = this.cell, off = this.half - 0.5 * c; // patch-space origin: patch coord u = (x + off) / c
    const P = this.N - 1;
    let t = tMin;
    // clip to the patch domain in xz
    const lo = 0, hi = P;
    const ux0 = (ox + off) / c, uz0 = (oz + off) / c, du = dx / c, dw = dz / c;
    const clip = (p0: number, d: number): [number, number] => {
      if (Math.abs(d) < 1e-12) return p0 >= lo && p0 <= hi ? [-Infinity, Infinity] : [Infinity, -Infinity];
      const a = (lo - p0) / d, b = (hi - p0) / d;
      return a < b ? [a, b] : [b, a];
    };
    const [ax, bx] = clip(ux0, du), [az, bz] = clip(uz0, dw);
    t = Math.max(t, ax, az);
    const tEnd = Math.min(tMax, bx, bz);
    if (t > tEnd) return null;
    const eps = 1e-5;
    let guard = 0;
    while (t <= tEnd && guard++ < 4096) {
      const u = ux0 + du * t, w = uz0 + dw * t;
      const pi = Math.min(P - 1, Math.max(0, Math.floor(u))), pj = Math.min(P - 1, Math.max(0, Math.floor(w)));
      // choose the coarsest level we can skip
      let skipped = false;
      for (let L = this.levels.length - 1; L >= 0; L--) {
        const bs = 1 << L;
        const bi = pi >> L, bj = pj >> L;
        const size = this.sizes[L];
        if (bi >= size || bj >= size) continue;
        const bmax = this.levels[L][bj * size + bi];
        // block extent in patch coords and ray exit param
        const u0 = bi * bs, u1 = Math.min(P, u0 + bs), w0 = bj * bs, w1 = Math.min(P, w0 + bs);
        const tx = du > 0 ? (u1 - ux0) / du : du < 0 ? (u0 - ux0) / du : Infinity;
        const tz = dw > 0 ? (w1 - uz0) / dw : dw < 0 ? (w0 - uz0) / dw : Infinity;
        const tExit = Math.min(tx, tz, tEnd);
        const yIn = oy + dy * t, yOut = oy + dy * tExit;
        if (Math.min(yIn, yOut) > bmax) {
          if (L === 0 || tExit > t + eps) {
            t = tExit + eps;
            skipped = true;
            break;
          }
        }
        if (L === 0) {
          // candidate patch: march inside with bilinear evaluation, then bisect
          const hit = this.marchPatch(ox, oy, oz, dx, dy, dz, t, Math.max(t, tExit));
          if (hit) return hit;
          t = tExit + eps;
          skipped = true;
          break;
        }
      }
      if (!skipped) break;
    }
    return null;
  }

  private marchPatch(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, t0: number, t1: number): RayHit | null {
    const steps = 6;
    let prevT = t0;
    let prevD = oy + dy * t0 - this.height(ox + dx * t0, oz + dz * t0);
    if (prevD <= 0) return this.mkHit(ox, oy, oz, dx, dy, dz, t0);
    for (let s = 1; s <= steps; s++) {
      const t = t0 + ((t1 - t0) * s) / steps;
      const d = oy + dy * t - this.height(ox + dx * t, oz + dz * t);
      if (d <= 0) {
        let a = prevT, b = t;
        for (let it = 0; it < 12; it++) {
          const m = 0.5 * (a + b);
          const dm = oy + dy * m - this.height(ox + dx * m, oz + dz * m);
          if (dm > 0) a = m;
          else b = m;
        }
        return this.mkHit(ox, oy, oz, dx, dy, dz, b);
      }
      prevT = t;
      prevD = d;
    }
    return null;
  }

  private mkHit(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, t: number): RayHit {
    const x = ox + dx * t, z = oz + dz * t;
    const e = this.cell * 0.5;
    const gx = (this.height(x + e, z) - this.height(x - e, z)) / (2 * e);
    const gz = (this.height(x, z + e) - this.height(x, z - e)) / (2 * e);
    const inv = 1 / Math.hypot(gx, 1, gz);
    return { t, x, y: oy + dy * t, z, nx: -gx * inv, ny: inv, nz: -gz * inv };
  }
}
