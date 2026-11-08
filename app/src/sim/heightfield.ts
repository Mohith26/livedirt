import { SoilType, SOILS } from './soil';

/**
 * Two-layer earthwork heightfield on a regular grid.
 *
 *   H = B + L
 *   B: bank (in-situ, undisturbed) surface elevation, m
 *   L: loose (excavated or sloughed) material thickness on top of the bank, m
 *
 * Digging removes loose material first, then bank. Bank material swells by the soil's swell factor when it
 * becomes loose (in the bucket, or when a face sloughs). Loose material relaxes to the angle of repose;
 * bank faces steeper than the soil's critical angle slough into loose material, but faces that already
 * existed in the reconstructed scan are treated as stable (they demonstrably stood in reality).
 *
 * Grid convention: cell (i, j) covers x in [i*cell - half, (i+1)*cell - half), z likewise; k = j * N + i.
 */
export interface Rect { i0: number; j0: number; i1: number; j1: number } // inclusive bounds

export interface Ledger {
  /** in-situ volume excavated by the tool, m^3 (bank measure) */
  bankCut: number;
  /** loose volume re-handled by the tool, m^3 (loose measure) */
  looseHandled: number;
  /** loose volume currently in the bucket / carried, m^3 */
  carried: number;
  /** loose volume dumped, m^3 */
  dumped: number;
  /** bank volume that sloughed off over-steep faces, m^3 (bank measure) */
  sloughed: number;
  /** number of dig passes */
  passes: number;
}

export interface DesignTrench {
  p0: [number, number];
  p1: [number, number];
  bottomWidth: number;
  depth: number;
  /** side slope as horizontal run per unit of vertical rise (0 = vertical walls) */
  sideSlope: number;
}

export interface GradeStats {
  cells: number;
  onGradePct: number;
  remainingCut: number; // m^3 above design grade inside the footprint
  overDig: number; // m^3 below design grade
  unplannedCut: number; // m^3 dug outside the design footprint
  maxAbove: number;
  maxBelow: number;
  meanAbsError: number;
}

export interface DigResult { bank: number; loose: number }

const emptyLedger = (): Ledger => ({ bankCut: 0, looseHandled: 0, carried: 0, dumped: 0, sloughed: 0, passes: 0 });

export class Heightfield {
  readonly N: number;
  readonly cell: number;
  readonly half: number;
  readonly area: number;
  readonly H0: Float32Array;
  readonly B: Float32Array;
  readonly L: Float32Array;
  /** 1 where the cell is occupied by an object (machine, vehicle): no digging, no material flow */
  readonly mask: Uint8Array;
  /** 1 where digging/dumping is permitted */
  readonly zone: Uint8Array;
  /** design grade elevation, NaN where no design */
  readonly design: Float32Array;
  designSpec: DesignTrench | null = null;
  soil: SoilType;
  ledger: Ledger = emptyLedger();
  maxDigDepth = 6.59;
  /** region modified since last consume (for GPU upload / regeneration) */
  private dirty: Rect | null = null;
  /** region still settling (relaxation needed) */
  private active: Rect | null = null;
  private undoStack: { B: Float32Array; L: Float32Array; ledger: Ledger }[] = [];
  readonly maxUndo = 12;
  /** last time step's maximum moved thickness, for settle detection */
  lastRelaxMove = 0;

  constructor(opts: { N: number; cell: number; half: number; H0: Float32Array; mask?: Uint8Array; zoneRadius?: number; soil?: SoilType }) {
    this.N = opts.N;
    this.cell = opts.cell;
    this.half = opts.half;
    this.area = opts.cell * opts.cell;
    const n = this.N * this.N;
    if (opts.H0.length !== n) throw new Error(`H0 length ${opts.H0.length} != N*N ${n}`);
    this.H0 = Float32Array.from(opts.H0);
    this.B = Float32Array.from(opts.H0);
    this.L = new Float32Array(n);
    this.mask = opts.mask ? Uint8Array.from(opts.mask) : new Uint8Array(n);
    this.zone = new Uint8Array(n);
    this.design = new Float32Array(n).fill(NaN);
    this.soil = opts.soil ?? SOILS[0];
    const zr = opts.zoneRadius ?? Infinity;
    for (let j = 0; j < this.N; j++)
      for (let i = 0; i < this.N; i++) {
        const k = j * this.N + i;
        const [x, z] = this.center(i, j);
        this.zone[k] = !this.mask[k] && x * x + z * z <= zr * zr ? 1 : 0;
      }
  }

  // ---------- coordinates ----------
  center(i: number, j: number): [number, number] {
    return [(i + 0.5) * this.cell - this.half, (j + 0.5) * this.cell - this.half];
  }
  cellOf(x: number, z: number): [number, number] {
    return [Math.floor((x + this.half) / this.cell), Math.floor((z + this.half) / this.cell)];
  }
  inGrid(i: number, j: number) {
    return i >= 0 && j >= 0 && i < this.N && j < this.N;
  }
  h(k: number) {
    return this.B[k] + this.L[k];
  }
  /** bilinear terrain height at world (x, z) using cell-centred samples */
  heightAt(x: number, z: number, src: 'H' | 'H0' = 'H'): number {
    const N = this.N;
    const fx = (x + this.half) / this.cell - 0.5, fz = (z + this.half) / this.cell - 0.5;
    const i0 = Math.max(0, Math.min(N - 1, Math.floor(fx))), j0 = Math.max(0, Math.min(N - 1, Math.floor(fz)));
    const i1 = Math.min(N - 1, i0 + 1), j1 = Math.min(N - 1, j0 + 1);
    const tx = Math.max(0, Math.min(1, fx - i0)), tz = Math.max(0, Math.min(1, fz - j0));
    const g = src === 'H' ? (k: number) => this.B[k] + this.L[k] : (k: number) => this.H0[k];
    const a = g(j0 * N + i0), b = g(j0 * N + i1), c = g(j1 * N + i0), d = g(j1 * N + i1);
    return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz;
  }

  // ---------- dirty / active tracking ----------
  private touch(i: number, j: number) {
    const grow = (r: Rect | null): Rect =>
      r ? { i0: Math.min(r.i0, i), j0: Math.min(r.j0, j), i1: Math.max(r.i1, i), j1: Math.max(r.j1, j) } : { i0: i, j0: j, i1: i, j1: j };
    this.dirty = grow(this.dirty);
    this.active = grow(this.active);
  }
  /** returns and clears the dirty rect */
  consumeDirty(): Rect | null {
    const d = this.dirty;
    this.dirty = null;
    return d;
  }
  peekDirty(): Rect | null {
    return this.dirty;
  }
  isSettling() {
    return this.active !== null;
  }
  markAllDirty() {
    this.dirty = { i0: 0, j0: 0, i1: this.N - 1, j1: this.N - 1 };
  }

  // ---------- undo ----------
  beginStroke() {
    this.undoStack.push({ B: this.B.slice(), L: this.L.slice(), ledger: { ...this.ledger } });
    if (this.undoStack.length > this.maxUndo) this.undoStack.shift();
  }
  canUndo() {
    return this.undoStack.length > 0;
  }
  undo(): boolean {
    const s = this.undoStack.pop();
    if (!s) return false;
    this.B.set(s.B);
    this.L.set(s.L);
    this.ledger = s.ledger;
    this.active = null;
    this.markAllDirty();
    return true;
  }
  reset() {
    this.B.set(this.H0);
    this.L.fill(0);
    this.ledger = emptyLedger();
    this.undoStack = [];
    this.active = null;
    this.markAllDirty();
  }

  // ---------- tool operations ----------
  /**
   * Remove material under a circular tool footprint with a smooth (raised-cosine) depth profile.
   * Removes loose material first, then bank. Never cuts below max dig depth below original grade, and,
   * when `respectDesign` is set, never below the design grade (semi-automatic grade control). With
   * `blockOutside`, cells outside the design footprint are not cut below their original grade.
   */
  /** minY: no cell is cut below this height (the bucket's cutting edge); used without the design clamp */
  dig(x: number, z: number, radius: number, depth: number, respectDesign = false, blockOutside = false, minY = -Infinity): DigResult {
    const res: DigResult = { bank: 0, loose: 0 };
    if (depth <= 0 || radius <= 0) return res;
    const [ci, cj] = this.cellOf(x, z);
    const r = Math.ceil(radius / this.cell);
    for (let j = cj - r; j <= cj + r; j++)
      for (let i = ci - r; i <= ci + r; i++) {
        if (!this.inGrid(i, j)) continue;
        const k = j * this.N + i;
        if (!this.zone[k]) continue;
        const [px, pz] = this.center(i, j);
        const d = Math.hypot(px - x, pz - z);
        if (d > radius) continue;
        const w = 0.5 * (1 + Math.cos((Math.PI * d) / radius));
        const want = depth * w;
        let floor = Math.max(this.H0[k] - this.maxDigDepth, minY);
        if (respectDesign && this.designSpec) {
          const D = this.design[k];
          // machine-control semantics: inside the footprint never below design; outside it (when blocking) never
          // below the original grade, so the bucket only removes material the design asks for
          if (!Number.isNaN(D)) floor = Math.max(floor, D);
          else if (blockOutside) floor = Math.max(floor, this.H0[k]);
        }
        const hNow = this.h(k);
        const take = Math.min(want, Math.max(0, hNow - floor));
        if (take <= 0) continue;
        const fromLoose = Math.min(this.L[k], take);
        this.L[k] -= fromLoose;
        const fromBank = take - fromLoose;
        this.B[k] -= fromBank;
        res.loose += fromLoose * this.area;
        res.bank += fromBank * this.area;
        this.touch(i, j);
      }
    this.ledger.bankCut += res.bank;
    this.ledger.looseHandled += res.loose;
    this.ledger.carried += res.bank * this.soil.swell + res.loose;
    return res;
  }

  /**
   * Deposit loose volume (m^3, loose measure) as a mound centred at (x, z). Volume that cannot be placed
   * (outside the permitted zone) stays carried. Returns the placed volume.
   */
  /** place loose material on the terrain without touching the carried ledger (spill from an over-full bucket); returns placed volume */
  spill(x: number, z: number, volume: number, radius: number): number {
    if (volume <= 0) return 0;
    const [ci, cj] = this.cellOf(x, z);
    const r = Math.ceil(radius / this.cell);
    const cells: { k: number; i: number; j: number; w: number }[] = [];
    let wsum = 0;
    for (let j = cj - r; j <= cj + r; j++)
      for (let i = ci - r; i <= ci + r; i++) {
        if (!this.inGrid(i, j)) continue;
        const [px, pz] = this.center(i, j);
        const w = Math.max(0, 1 - Math.hypot(px - x, pz - z) / radius);
        if (w <= 0) continue;
        cells.push({ k: j * this.N + i, i, j, w });
        wsum += w;
      }
    if (wsum <= 0) return 0;
    for (const c of cells) { this.L[c.k] += (volume * (c.w / wsum)) / this.area; this.touch(c.i, c.j); }
    return volume;
  }

  dump(x: number, z: number, volume: number, radius: number): number {
    volume = Math.min(volume, this.ledger.carried);
    if (volume <= 0) return 0;
    const [ci, cj] = this.cellOf(x, z);
    const r = Math.ceil(radius / this.cell);
    const cells: { k: number; i: number; j: number; w: number }[] = [];
    let wsum = 0;
    for (let j = cj - r; j <= cj + r; j++)
      for (let i = ci - r; i <= ci + r; i++) {
        if (!this.inGrid(i, j)) continue;
        const k = j * this.N + i;
        if (!this.zone[k]) continue;
        const [px, pz] = this.center(i, j);
        const d = Math.hypot(px - x, pz - z);
        if (d > radius) continue;
        const w = Math.max(0, 1 - d / radius); // cone: dumped material lands peaked, then relaxes
        cells.push({ k, i, j, w });
        wsum += w;
      }
    if (wsum <= 0) return 0;
    for (const c of cells) {
      this.L[c.k] += (volume * (c.w / wsum)) / this.area;
      this.touch(c.i, c.j);
    }
    this.ledger.carried -= volume;
    this.ledger.dumped += volume;
    return volume;
  }

  /**
   * One relaxation step over the active region (expanded by a margin). Loose material flows downhill
   * where the local slope exceeds the angle of repose; bank faces newly steepened beyond the critical
   * angle slough into loose material (bank volume x swell). Returns max thickness moved this step.
   */
  relax(iterations = 2): number {
    if (!this.active) return 0;
    const N = this.N;
    const m = 3;
    const rect = { i0: Math.max(1, this.active.i0 - m), j0: Math.max(1, this.active.j0 - m), i1: Math.min(N - 2, this.active.i1 + m), j1: Math.min(N - 2, this.active.j1 + m) };
    const tanL = Math.tan((this.soil.reposeDeg * Math.PI) / 180) * this.cell;
    const tanB = Math.tan((this.soil.bankDeg * Math.PI) / 180) * this.cell;
    const swell = this.soil.swell;
    // 8-neighbourhood with true distances (4-neighbour flow produces square/diamond piles)
    const nb = [1, -1, N, -N, N + 1, N - 1, -N + 1, -N - 1];
    const dist = [1, 1, 1, 1, Math.SQRT2, Math.SQRT2, Math.SQRT2, Math.SQRT2];
    const share = [0.16, 0.16, 0.16, 0.16, 0.09, 0.09, 0.09, 0.09];
    let maxMove = 0;
    let next: Rect | null = null;
    const mark = (k: number) => {
      const i = k % N, j = (k / N) | 0;
      next = next ? { i0: Math.min(next.i0, i), j0: Math.min(next.j0, j), i1: Math.max(next.i1, i), j1: Math.max(next.j1, j) } : { i0: i, j0: j, i1: i, j1: j };
      this.touch(i, j);
    };
    for (let it = 0; it < iterations; it++) {
      const flip = it & 1;
      for (let jj = rect.j0; jj <= rect.j1; jj++) {
        const j = flip ? rect.j1 - (jj - rect.j0) : jj;
        for (let ii = rect.i0; ii <= rect.i1; ii++) {
          const i = flip ? rect.i1 - (ii - rect.i0) : ii;
          const k = j * N + i;
          if (this.mask[k]) continue;
          for (let q = 0; q < 8; q++) {
            const n = k + nb[q];
            if (this.mask[n]) continue;
            const hn = this.B[n] + this.L[n];
            // bank sloughing: only faces steeper than both the critical angle and what the scan already had
            const faceB = this.B[k] - hn;
            const allowedB = Math.max(tanB * dist[q], this.H0[k] - this.H0[n] + 0.02);
            if (faceB > allowedB) {
              const c = (faceB - allowedB) * share[q];
              this.B[k] -= c;
              this.L[k] += c * swell;
              this.ledger.sloughed += c * this.area;
              if (c > maxMove) maxMove = c;
              mark(k);
              mark(n);
            }
            // loose flow
            const lk = this.L[k];
            if (lk <= 0) continue;
            const d = this.B[k] + lk - (this.B[n] + this.L[n]);
            const excess = d - tanL * dist[q];
            if (excess > 1e-4) {
              const qv = Math.min(lk, excess * share[q]);
              this.L[k] -= qv;
              this.L[n] += qv;
              if (qv > maxMove) maxMove = qv;
              mark(k);
              mark(n);
            }
          }
        }
      }
    }
    this.lastRelaxMove = maxMove;
    this.active = maxMove > 2e-4 ? (next as Rect | null) : null;
    return maxMove;
  }

  /** relax until settled or step budget exhausted (used by tests / autonomy planning) */
  settle(maxSteps = 400) {
    for (let s = 0; s < maxSteps && this.active; s++) this.relax(2);
  }

  // ---------- design grade ----------
  /** 1 where the design cell is on the trench floor (bottom width), 0 on side/end slopes */
  designFloor: Uint8Array = new Uint8Array(0);
  setDesign(spec: DesignTrench | null) {
    this.designSpec = spec;
    this.design.fill(NaN);
    if (this.designFloor.length !== this.design.length) this.designFloor = new Uint8Array(this.design.length);
    this.designFloor.fill(0);
    if (spec) {
      const N = this.N;
      const [x0, z0] = spec.p0, [x1, z1] = spec.p1;
      const len = Math.hypot(x1 - x0, z1 - z0);
      if (len < 1e-3) return;
      const ux = (x1 - x0) / len, uz = (z1 - z0) / len;
      const f0 = this.heightAt(x0, z0, 'H0') - spec.depth, f1 = this.heightAt(x1, z1, 'H0') - spec.depth;
      const hw = spec.bottomWidth / 2;
      const s = spec.sideSlope;
      for (let j = 0; j < N; j++)
        for (let i = 0; i < N; i++) {
          const k = j * N + i;
          if (this.mask[k]) continue;
          const [px, pz] = this.center(i, j);
          const t = (px - x0) * ux + (pz - z0) * uz;
          const o = -(px - x0) * uz + (pz - z0) * ux;
          const tc = Math.max(0, Math.min(len, t));
          const floor = f0 + (f1 - f0) * (tc / len);
          const r = Math.hypot(Math.max(0, Math.abs(o) - hw), t < 0 ? -t : t > len ? t - len : 0);
          let D: number;
          if (r <= 1e-9) D = floor;
          else if (s <= 1e-6) continue; // vertical walls: outside the floor footprint there is no design cut
          else D = floor + r / s;
          if (D < this.H0[k] - 0.005) { this.design[k] = D; this.designFloor[k] = r <= 1e-9 ? 1 : 0; }
        }
    }
  }

  /** region: 'all' design cells (legacy), 'bottom' = trench floor only, 'slopes' = side/end slopes only */
  gradeStats(tol = 0.03, region: 'all' | 'bottom' | 'slopes' = 'all'): GradeStats {
    const st: GradeStats = { cells: 0, onGradePct: 0, remainingCut: 0, overDig: 0, unplannedCut: 0, maxAbove: 0, maxBelow: 0, meanAbsError: 0 };
    let on = 0, abs = 0;
    for (let k = 0; k < this.design.length; k++) {
      const D = this.design[k];
      const H = this.B[k] + this.L[k];
      if (!Number.isNaN(D) && region !== 'all' && (this.designFloor[k] === 1) !== (region === 'bottom')) continue;
      if (Number.isNaN(D)) {
        if (region !== 'all') continue;
        const cut = this.H0[k] - H;
        if (cut > 0.01 && this.zone[k]) st.unplannedCut += cut * this.area;
        continue;
      }
      st.cells++;
      const e = H - D;
      abs += Math.abs(e);
      if (Math.abs(e) <= tol) on++;
      if (e > 0) st.remainingCut += e * this.area;
      else st.overDig += -e * this.area;
      st.maxAbove = Math.max(st.maxAbove, e);
      st.maxBelow = Math.max(st.maxBelow, -e);
    }
    st.onGradePct = st.cells ? (100 * on) / st.cells : 0;
    st.meanAbsError = st.cells ? abs / st.cells : 0;
    return st;
  }

  /** add d metres of surface height at cell k (negative: remove loose first, then bank; positive: loose) */
  applyDelta(i: number, j: number, d: number) {
    if (!this.inGrid(i, j)) return;
    const k = j * this.N + i;
    if (this.mask[k] || d === 0) return;
    if (d > 0) this.L[k] += d;
    else {
      const fromL = Math.min(this.L[k], -d);
      this.L[k] -= fromL;
      this.B[k] -= -d - fromL;
    }
    this.touch(i, j);
  }

  // ---------- aggregates ----------
  /** loose volume lying on the field, m^3 */
  looseOnField(): number {
    let s = 0;
    for (let k = 0; k < this.L.length; k++) s += this.L[k];
    return s * this.area;
  }
  /** net bank change vs original, m^3 (negative = removed) */
  bankChange(): number {
    let s = 0;
    for (let k = 0; k < this.B.length; k++) s += this.B[k] - this.H0[k];
    return s * this.area;
  }
}
