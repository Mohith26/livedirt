import { solveIkFlexible, forwardPlanar, planarToWorld, type JointState, type LinkageGeometry } from './kinematics';
import type { Heightfield, DesignTrench } from '../sim/heightfield';
import { actuatorDemand, type ActuatorDemand } from './actuators';
import { NeuralSoilWindow, type NeuralModel, type NeuralGate, type FeatureStats } from '../sim/neuralSoil';

/** one recorded dig for MPM replay: terrain (window frame, window + margin on every side, 5 cm), poses [t, pivot x, pivot z, pitch, relief] */
export interface DigTrace { margin: number; cell: number; nx: number; ny: number; terrain: number[]; poses: number[][]; bite: number; pass: number; t0: number; soil: string; neural: boolean; r0: number; groundY: number; shift: number; attitude: string; swing?: number }

/**
 * Trench-to-grade dig cycle for the twin.
 *
 * Plan: the design trench is split into lanes (bucket width) and bites along each lane, ordered far-to-near
 * from the machine, repeated in passes until the grade survey says the floor is within tolerance.
 * Each bite: approach above the bite start -> penetrate -> drag toward the machine at cut depth (bounded by
 * design grade and the bite's max cut) -> curl -> lift -> swing to the spoil point -> dump -> swing back.
 * Joint motion is rate-limited per joint (rates typical for the class). Material is removed by the bucket
 * where its cutting edge is below the terrain, until the heaped capacity is reached.
 *
 * Grade survey uses the machine's OWN simulated lidar points (noisy, with dropouts), not the simulator's
 * ground truth; the difference between the two is reported as a sensing metric.
 */
export type AutoPhase = 'idle' | 'approach' | 'penetrate' | 'drag' | 'curl' | 'lift' | 'swing-dump' | 'dump' | 'swing-back' | 'survey' | 'done' | 'halted' | 'unreachable';

export interface Bite { a: [number, number]; b: [number, number]; lane: number }

export interface SurveyResult {
  t: number;
  points: number;
  lidarMeanAbove: number; // m, mean (measured H - design) over footprint cells with returns
  truthMeanAbove: number; // m, same cells, simulator truth
  coverage: number; // fraction of footprint cells with >=1 lidar return
  cellsAboveTol: number;
  /** fraction of covered cells within tolerance (lidar-measured) */
  inTolFrac: number;
}

export interface AutoStats {
  cycles: number;
  bankRemoved: number;
  avgCycleS: number;
  avgFill: number; // fraction of capacity per cycle
  elapsed: number;
  productivity: number; // bank m^3 / h
  passes: number;
  surveys: SurveyResult[];
  haltedS: number;
}

interface Waypoint { tip: [number, number, number]; bucketAbs: number; phase: AutoPhase; cut?: boolean; dwell?: number; tilt?: number }

const DEG = Math.PI / 180;

export class DigAutonomy {
  phase: AutoPhase = 'idle';
  joints: JointState;
  readonly g: LinkageGeometry;
  readonly hf: Heightfield;
  cx: number; cz: number; groundY: number;
  design: DesignTrench | null = null;
  spoil: [number, number] = [0, 0];
  bucketCapacity = 0.92 * 1.15; // m^3 heaped (spec 0.92 m^3 SAE) with ~15% heap allowance
  bucketWidth = 1.05;
  maxCutPerBite = 0.5; // m below current surface per bite
  /**
   * 'legacy': fixed 1.4 m bites on fixed lanes, every lane re-dug each pass; relies on the heightfield's per-cell design
   * clamp to shape the cut. 'adaptive': machine-control edge (the cutting edge never goes below design anywhere across
   * its width), one floor lane plus slope lanes, bite length planned from the scanned surface to fill the bucket using
   * a fill efficiency learned from measured payloads, cut depth backed off after relief stalls, later passes only dig
   * where the survey found material above grade. Works with any soil backend (no design clamp needed).
   */
  controller: 'legacy' | 'adaptive' = 'legacy';
  targetFill = 0.9; // fraction of heaped capacity the adaptive planner aims for
  /** heightfield cutting clamps every cell to the design (legacy machine-control shortcut inside the soil model). When
   * false, the heightfield only refuses to cut below the bucket's cutting edge, like any other soil backend. */
  designClamp = true;
  fillEff = 0.6; // learned: bank volume measured in the bucket / bank volume swept above the edge
  private adaptiveCut = 0.5; // current max cut below surface (m), backed off on stalls
  private lanesA: { off: number; kind: 'floor' | 'slope' }[] = [];
  private laneFar: [number, number] = [0, 0];
  private laneNear: [number, number] = [0, 0];
  private laneV: [number, number] = [0, 0];
  private laneIdx = 0;
  private laneT = 0; // progress along the current lane, 0 (far end) -> 1 (near end)
  private biteBank0 = 0;
  /** tiltrotator roll of the bucket about the drag direction (rad, + raises the +lateral side); 0 for the legacy controller */
  tilt = 0;
  maxTilt = 40 * DEG;
  private curTilt = 0; // tilt of the waypoint being executed
  private biteSwept = 0;
  lastBiteCompletion = 1; // set by external backends (1 = path completed)
  adaptiveLog: { lane: number; t0: number; t1: number; swept: number; measured: number; eff: number; cut: number; completion: number }[] = [];
  /** seconds the machine holds still to accumulate scans before judging grade */
  surveyDwell = 1.6;
  /** minimum fraction of design cells with lidar returns needed to judge grade */
  minCoverage = 0.4;
  private surveyT = -1;
  private surveySum: Float32Array | null = null;
  private surveyCnt: Uint16Array | null = null;
  private surveyPts = 0;
  private lastScanRef: Float32Array | null = null;
  tolerance = 0.03;
  /** joint rate limits, rad/s */
  rates = { swing: 24 * DEG, boom: 13 * DEG, arm: 17 * DEG, bucket: 34 * DEG };
  speed = 1;
  private queue: Waypoint[] = [];
  private bites: Bite[] = [];
  private biteIdx = 0;
  private load = 0;
  private cycleStart = 0;
  private cycleTimes: number[] = [];
  private fills: number[] = [];
  stats: AutoStats = { cycles: 0, bankRemoved: 0, avgCycleS: 0, avgFill: 0, elapsed: 0, productivity: 0, passes: 0, surveys: [], haltedS: 0 };
  /** called when the controller wants a survey; returns lidar points (xyz+i) */
  requestScan: (() => Float32Array | null) | null = null;
  haltReason = '';
  /** true when the final survey verified the grade with adequate coverage */
  verified: boolean | null = null;
  log: string[] = [];
  /**
   * Optional learned soil (DigBench surrogate trained on 3D MPM). When set, penetrate/drag/curl run the model on a
   * 3.4 m x 2 m window around the bucket instead of the kinematic cut, and the predicted drag force drives a relief
   * valve: above fMax (DX225LC-7 ISO arm digging force, 112.8 kN) the boom and arm slow down.
   */
  neural: { model: NeuralModel; mu: number; fMax: number; floor?: boolean; gate?: NeuralGate; stats?: FeatureStats; legacyRelief?: boolean; consistency?: boolean } | null = null;
  private win: NeuralSoilWindow | null = null;
  private winR0 = 0;
  private winT = 0;
  soilForce: [number, number, number] = [0, 0, 0];
  reliefScale = 1;
  peakDragN = 0;
  neuralClipped = 0;
  neuralSteps = 0;
  neuralFloored = 0;
  neuralSpilled = 0;

  constructor(hf: Heightfield, g: LinkageGeometry, cx: number, cz: number, groundY: number, initial: JointState) {
    this.hf = hf; this.g = g; this.cx = cx; this.cz = cz; this.groundY = groundY;
    this.joints = { ...initial };
  }

  private note(s: string) { this.log.push(`${this.stats.elapsed.toFixed(1)}s ${s}`); if (this.log.length > 60) this.log.shift(); }

  /** build lanes and bites for a design; spoil is placed beside the trench, away from the machine */
  plan(design: DesignTrench) {
    this.design = design;
    const [x0, z0] = design.p0, [x1, z1] = design.p1;
    const len = Math.hypot(x1 - x0, z1 - z0);
    const ux = (x1 - x0) / len, uz = (z1 - z0) / len, vx = -uz, vz = ux;
    // order far -> near relative to the machine
    const d0 = Math.hypot(x0 - this.cx, z0 - this.cz), d1 = Math.hypot(x1 - this.cx, z1 - this.cz);
    // bites run through the end slopes too (design extends depth*slope past each endpoint)
    const ext = design.depth * design.sideSlope * 0.85;
    const far0: [number, number] = d0 > d1 ? [x0, z0] : [x1, z1];
    const near0: [number, number] = d0 > d1 ? [x1, z1] : [x0, z0];
    const fdx = far0[0] - near0[0], fdz = far0[1] - near0[1], fl = Math.hypot(fdx, fdz) || 1;
    const far: [number, number] = [far0[0] + (fdx / fl) * ext, far0[1] + (fdz / fl) * ext];
    const near: [number, number] = [near0[0] - (fdx / fl) * ext, near0[1] - (fdz / fl) * ext];
    // lanes cover the full top width so side slopes get trimmed too (cut height is clamped to design per cell)
    const topW = design.bottomWidth + 2 * design.depth * design.sideSlope;
    const lanes = Math.max(1, Math.ceil(topW / (this.bucketWidth * 0.8)));
    const span = Math.max(0, topW - this.bucketWidth * 0.9);
    const biteLen = 1.4;
    const nb = Math.max(1, Math.ceil((len + 2 * ext) / biteLen));
    this.bites = [];
    for (let l = 0; l < lanes; l++) {
      const off = lanes === 1 ? 0 : -span / 2 + (span * l) / (lanes - 1);
      for (let i = 0; i < nb; i++) {
        const t0 = i / nb, t1 = (i + 1) / nb;
        const ax = far[0] + (near[0] - far[0]) * t0 + vx * off, az = far[1] + (near[1] - far[1]) * t0 + vz * off;
        const bx = far[0] + (near[0] - far[0]) * t1 + vx * off, bz = far[1] + (near[1] - far[1]) * t1 + vz * off;
        this.bites.push({ a: [ax, az], b: [bx, bz], lane: l });
      }
    }
    // spoil: beside the trench midpoint, on the side away from the machine, with OSHA 1926.651(j)(2)
    // style setback (>= 0.61 m from the edge) plus pile radius
    const mx = (x0 + x1) / 2, mz = (z0 + z1) / 2;
    const side = (this.cx - mx) * vx + (this.cz - mz) * vz > 0 ? -1 : 1;
    const topHalf = design.bottomWidth / 2 + design.depth * design.sideSlope;
    const setback = topHalf + 0.61 + 1.3;
    this.spoil = [mx + vx * side * setback, mz + vz * side * setback];
    this.biteIdx = 0;
    // adaptive lanes: the floor lane on the centreline, then slope lanes outward on both sides (0.8 bucket widths apart)
    this.lanesA = [{ off: 0, kind: 'floor' }];
    const step = this.bucketWidth * 0.8;
    for (let o = design.bottomWidth / 2 + this.bucketWidth / 2 - 0.1; o - this.bucketWidth / 2 < topW / 2; o += step) {
      this.lanesA.push({ off: o, kind: 'slope' }, { off: -o, kind: 'slope' });
    }
    this.laneFar = far; this.laneNear = near; this.laneV = [vx, vz];
    this.laneIdx = 0; this.laneT = 0;
    this.note(`plan: ${lanes} lane(s) x ${nb} bites, trench ${len.toFixed(1)} m, spoil ${setback.toFixed(1)} m off centreline` + (this.controller === 'adaptive' ? `; adaptive: ${this.lanesA.length} lanes` : ''));
  }

  start() {
    if (!this.design) return;
    this.verified = null;
    this.surveyT = -1;
    this.phase = 'approach';
    this.queue = [];
    this.cycleStart = this.stats.elapsed;
    this.stats.passes = Math.max(1, this.stats.passes);
    this.note('start');
  }
  stop() { this.phase = 'idle'; this.queue = []; this.note('stop'); }

  private surfaceAt(x: number, z: number) { return this.hf.heightAt(x, z); }

  private buildBiteWaypoints(b: Bite) {
    const D = this.design!;
    const down = -88 * DEG;
    const H0a = this.surfaceAt(b.a[0], b.a[1]);
    const designAt = (x: number, z: number) => {
      const [i, j] = this.hf.cellOf(x, z);
      const v = this.hf.design[j * this.hf.N + i];
      return Number.isNaN(v) ? this.hf.heightAt(x, z, 'H0') - D.depth : v;
    };
    // lowest design point under the bucket width (per-cell design clamp in dig() prevents over-cutting the rest)
    const sw = this.joints.swing;
    const lx = -Math.sin(sw), lz = Math.cos(sw);
    const designUnder = (x: number, z: number) => Math.min(designAt(x, z), designAt(x + lx * this.bucketWidth / 2, z + lz * this.bucketWidth / 2), designAt(x - lx * this.bucketWidth / 2, z - lz * this.bucketWidth / 2));
    const cutY = (x: number, z: number) => Math.max(designUnder(x, z), this.surfaceAt(x, z) - this.maxCutPerBite);
    const above = 0.9;
    this.queue.push({ tip: [b.a[0], H0a + above, b.a[1]], bucketAbs: -70 * DEG, phase: 'approach' });
    this.queue.push({ tip: [b.a[0], cutY(b.a[0], b.a[1]), b.a[1]], bucketAbs: down, phase: 'penetrate', cut: true, dwell: 0.4 });
    const steps = 5;
    for (let s = 1; s <= steps; s++) {
      const t = s / steps;
      const x = b.a[0] + (b.b[0] - b.a[0]) * t, z = b.a[1] + (b.b[1] - b.a[1]) * t;
      this.queue.push({ tip: [x, cutY(x, z), z], bucketAbs: down + 35 * DEG * t, phase: 'drag', cut: true });
    }
    const bx = b.b[0], bz = b.b[1];
    this.queue.push({ tip: [bx, this.surfaceAt(bx, bz) + 0.2, bz], bucketAbs: 20 * DEG, phase: 'curl' });
    this.queue.push({ tip: [bx, this.surfaceAt(bx, bz) + 2.2, bz], bucketAbs: 25 * DEG, phase: 'lift' });
    const sy = this.surfaceAt(this.spoil[0], this.spoil[1]);
    this.queue.push({ tip: [this.spoil[0], Math.max(sy + 2.2, this.groundY + 1.8), this.spoil[1]], bucketAbs: 25 * DEG, phase: 'swing-dump' });
    this.queue.push({ tip: [this.spoil[0], Math.max(sy + 1.9, this.groundY + 1.5), this.spoil[1]], bucketAbs: -115 * DEG, phase: 'dump', dwell: 1.1 });
  }

  private designOrGround(x: number, z: number) {
    const [i, j] = this.hf.cellOf(x, z);
    const v = this.hf.inGrid(i, j) ? this.hf.design[j * this.hf.N + i] : NaN;
    return Number.isNaN(v) ? this.hf.heightAt(x, z, 'H0') : v;
  }
  /** cross-slope of the design under the bucket at a lane point -> tilt (clamped to the tiltrotator range) */
  private laneTilt(x: number, z: number) {
    const lx = this.laneV[0], lz = this.laneV[1], h = this.bucketWidth / 2;
    const g = (this.designOrGround(x + lx * h, z + lz * h) - this.designOrGround(x - lx * h, z - lz * h)) / this.bucketWidth;
    return Math.max(-this.maxTilt, Math.min(this.maxTilt, Math.atan(g)));
  }
  /** machine-control edge height (at the lane centre) for the current tilt: the tilted edge line never goes below the
   * design (or the original ground outside the footprint) anywhere across the bucket width */
  private edgeFloor(x: number, z: number) {
    const lx = this.laneV[0], lz = this.laneV[1], tn = Math.tan(this.tilt);
    let m = -Infinity;
    for (let k = 0; k <= 6; k++) {
      const o = -this.bucketWidth / 2 + (this.bucketWidth * k) / 6;
      m = Math.max(m, this.designOrGround(x + lx * o, z + lz * o) - tn * o);
    }
    return m;
  }
  private lanePoint(lane: number, t: number): [number, number] {
    const o = this.lanesA[lane].off, [fx, fz] = this.laneFar, [nx, nz] = this.laneNear;
    return [fx + (nx - fx) * t + this.laneV[0] * o, fz + (nz - fz) * t + this.laneV[1] * o];
  }
  /** cut height at a lane point: never below the machine-control edge, at most adaptiveCut below the surface */
  private cutHeight(x: number, z: number) { return Math.max(this.edgeFloor(x, z), this.surfaceAt(x, z) - this.adaptiveCut); }
  /** bank volume (m^3) above the cut line under the bucket width per metre of lane at a point */
  private sweptPerM(x: number, z: number) {
    const lx = this.laneV[0], lz = this.laneV[1], e = this.cutHeight(x, z), tn = Math.tan(this.tilt);
    let a = 0;
    for (let k = 0; k < 6; k++) {
      const o = -this.bucketWidth / 2 + (this.bucketWidth * (k + 0.5)) / 6;
      a += Math.max(0, this.surfaceAt(x + lx * o, z + lz * o) - (e + tn * o)) * (this.bucketWidth / 6);
    }
    return a;
  }
  /** next adaptive bite, or null when the pass is complete */
  private nextAdaptiveBite(): Bite | null {
    const len = Math.hypot(this.laneNear[0] - this.laneFar[0], this.laneNear[1] - this.laneFar[1]) || 1;
    const ds = 0.05, minLen = 0.4, maxLen = 2.2;
    const want = (this.targetFill * this.bucketCapacity) / ((this.hf.soil?.swell ?? 1) * this.fillEff);
    while (this.laneIdx < this.lanesA.length) {
      { const [mx, mz] = this.lanePoint(this.laneIdx, 0.5); this.tilt = this.lanesA[this.laneIdx].kind === 'floor' ? 0 : this.laneTilt(mx, mz); }
      // skip ahead to the first point on this lane with material above the cut line
      while (this.laneT < 1) {
        const [x, z] = this.lanePoint(this.laneIdx, this.laneT);
        if (this.sweptPerM(x, z) > 0.02) break;
        this.laneT += ds / len;
      }
      if (this.laneT >= 1) { this.laneIdx++; this.laneT = 0; continue; }
      const t0 = this.laneT;
      let acc = 0, t = t0;
      while (t < 1 && (t - t0) * len < maxLen && (acc < want || (t - t0) * len < minLen)) {
        const [x, z] = this.lanePoint(this.laneIdx, t);
        acc += this.sweptPerM(x, z) * ds;
        t += ds / len;
      }
      t = Math.min(1, t);
      this.laneT = t;
      // not worth a cycle: less than 5% of a bucket of bank material left above the cut line on this stretch
      if (acc < 0.05 * this.bucketCapacity / (this.hf.soil?.swell ?? 1)) continue;
      this.biteSwept = acc;
      const a = this.lanePoint(this.laneIdx, t0), b = this.lanePoint(this.laneIdx, t);
      return { a, b, lane: this.laneIdx };
    }
    return null;
  }
  private buildAdaptiveWaypoints(b: Bite) {
    const down = -88 * DEG;
    const above = 0.9;
    const H0a = this.surfaceAt(b.a[0], b.a[1]);
    this.biteBank0 = this.stats.bankRemoved;
    this.queue.push({ tip: [b.a[0], H0a + above, b.a[1]], bucketAbs: -70 * DEG, phase: 'approach' });
    this.queue.push({ tip: [b.a[0], this.cutHeight(b.a[0], b.a[1]), b.a[1]], bucketAbs: down, phase: 'penetrate', cut: true, dwell: 0.4, tilt: this.tilt });
    const L = Math.hypot(b.b[0] - b.a[0], b.b[1] - b.a[1]);
    const steps = Math.max(2, Math.ceil(L / 0.3));
    for (let s = 1; s <= steps; s++) {
      const t = s / steps;
      const x = b.a[0] + (b.b[0] - b.a[0]) * t, z = b.a[1] + (b.b[1] - b.a[1]) * t;
      this.queue.push({ tip: [x, this.cutHeight(x, z), z], bucketAbs: down + 35 * DEG * t, phase: 'drag', cut: true, tilt: this.tilt });
    }
    const bx = b.b[0], bz = b.b[1];
    this.queue.push({ tip: [bx, this.surfaceAt(bx, bz) + 0.2, bz], bucketAbs: 20 * DEG, phase: 'curl', tilt: this.tilt });
    this.queue.push({ tip: [bx, this.surfaceAt(bx, bz) + 2.2, bz], bucketAbs: 25 * DEG, phase: 'lift' });
    const sy = this.surfaceAt(this.spoil[0], this.spoil[1]);
    this.queue.push({ tip: [this.spoil[0], Math.max(sy + 2.2, this.groundY + 1.8), this.spoil[1]], bucketAbs: 25 * DEG, phase: 'swing-dump' });
    this.queue.push({ tip: [this.spoil[0], Math.max(sy + 1.9, this.groundY + 1.5), this.spoil[1]], bucketAbs: -115 * DEG, phase: 'dump', dwell: 1.1 });
  }
  /** after each adaptive bite: learn fill efficiency from the measured payload, back the cut off after a stall */
  private adaptAfterBite() {
    const measured = this.stats.bankRemoved - this.biteBank0;
    if (this.biteSwept > 0.05) {
      const eff = Math.max(0.1, Math.min(2.0, measured / this.biteSwept));
      this.fillEff = 0.6 * this.fillEff + 0.4 * eff;
    }
    const c = this.lastBiteCompletion;
    if (c < 0.9) this.adaptiveCut = Math.max(0.15, this.adaptiveCut * 0.7);
    else this.adaptiveCut = Math.min(0.6, this.adaptiveCut * 1.1);
    this.adaptiveLog.push({ lane: this.laneIdx, t0: 0, t1: this.laneT, swept: +this.biteSwept.toFixed(4), measured: +measured.toFixed(4), eff: +this.fillEff.toFixed(3), cut: +this.adaptiveCut.toFixed(3), completion: +c.toFixed(3) });
    this.lastBiteCompletion = 1;
  }

  /** accumulate the latest merged scan into the survey grid (call each step during the dwell) */
  private accumulateSurvey() {
    const pts = this.requestScan?.();
    if (!pts || pts === this.lastScanRef) return;
    this.lastScanRef = pts;
    const hf = this.hf, N = hf.N;
    if (!this.surveySum) { this.surveySum = new Float32Array(N * N); this.surveyCnt = new Uint16Array(N * N); this.surveyPts = 0; }
    for (let i = 0; i < pts.length; i += 4) {
      const [ci, cj] = hf.cellOf(pts[i], pts[i + 2]);
      if (!hf.inGrid(ci, cj)) continue;
      const k = cj * N + ci;
      if (Number.isNaN(hf.design[k])) continue;
      this.surveySum[k] += pts[i + 1]; this.surveyCnt![k]++; this.surveyPts++;
    }
  }

  private survey(): SurveyResult | null {
    const D = this.design;
    if (!D) return null;
    const hf = this.hf, N = hf.N;
    const sum = this.surveySum ?? new Float32Array(N * N), cnt = this.surveyCnt ?? new Uint16Array(N * N);
    let cells = 0, covered = 0, lid = 0, tru = 0, above = 0, inTol = 0;
    for (let k = 0; k < N * N; k++) {
      if (Number.isNaN(hf.design[k])) continue;
      cells++;
      if (!cnt[k]) continue;
      covered++;
      const meas = sum[k] / cnt[k];
      const e = meas - hf.design[k];
      lid += e;
      tru += hf.B[k] + hf.L[k] - hf.design[k];
      if (e > this.tolerance * 1.5) above++;
      if (Math.abs(e) <= this.tolerance * 1.5) inTol++;
    }
    const r: SurveyResult = { t: this.stats.elapsed, points: this.surveyPts, lidarMeanAbove: covered ? lid / covered : NaN, truthMeanAbove: covered ? tru / covered : NaN, coverage: cells ? covered / cells : 0, cellsAboveTol: above, inTolFrac: covered ? inTol / covered : 0 };
    this.stats.surveys.push(r);
    this.surveySum = null; this.surveyCnt = null;
    this.note(`survey (${this.surveyDwell.toFixed(1)} s dwell): ${r.points} pts, coverage ${(r.coverage * 100).toFixed(0)}%, lidar ${(r.lidarMeanAbove * 100).toFixed(1)} cm vs truth ${(r.truthMeanAbove * 100).toFixed(1)} cm above grade, ${above} cells out of tol`);
    return r;
  }

  /**
   * Advance the controller. `personInZone` is the result of the safety monitor (lidar person returns
   * inside the swing exclusion zone); when true the machine holds position.
   */
  step(dt: number, personInZone: boolean): JointState {
    if (this.phase === 'idle' || this.phase === 'done' || this.phase === 'unreachable') return this.joints;
    dt *= this.speed;
    this.stats.elapsed += dt;
    if (personInZone) {
      if (this.phase !== 'halted') this.note('HALT: person detected in swing zone');
      this.phase = 'halted';
      this.stats.haltedS += dt;
      return this.joints;
    }
    if (this.phase === 'halted') { this.phase = 'approach'; this.note('resume: zone clear'); }

    let nextA: Bite | null = null;
    if (this.queue.length === 0 && this.controller === 'adaptive') nextA = this.nextAdaptiveBite();
    if (this.queue.length === 0) {
      if (this.controller === 'adaptive' ? nextA === null : this.biteIdx >= this.bites.length) {
        // pass complete: park the bucket clear of the trench, dwell while scans accumulate, then decide
        if (this.surveyT < 0) { this.surveyT = this.surveyDwell; this.surveySum = null; this.surveyCnt = null; this.lastScanRef = null; this.note('survey: holding still to scan'); }
        this.phase = 'survey';
        this.accumulateSurvey();
        this.surveyT -= dt;
        if (this.surveyT > 0) return this.joints;
        this.surveyT = -1;
        const r = this.survey();
        const blind = !r || r.coverage < this.minCoverage;
        if (blind) {
          this.phase = 'done';
          this.note(`stopped: survey coverage ${(100 * (r?.coverage ?? 0)).toFixed(0)}% is too low to verify grade; check sensor placement`);
          this.verified = false;
          return this.joints;
        }
        // acceptance: >= 85% of measured cells within tolerance (1.5x tol allows for sensor noise), or a pass that
        // no longer improves the measured mean error by >= 1 cm (diminishing returns); cap at 6 passes
        const prev = this.stats.surveys.length > 1 ? this.stats.surveys[this.stats.surveys.length - 2] : null;
        const improving = !prev || Math.abs(prev.lidarMeanAbove) - Math.abs(r.lidarMeanAbove) > 0.01;
        const accepted = r.inTolFrac >= 0.85;
        if (!accepted && improving && this.stats.passes < 6) {
          this.stats.passes++;
          this.biteIdx = 0;
          this.laneIdx = 0; this.laneT = 0;
          this.note(`pass ${this.stats.passes}`);
        } else {
          this.phase = 'done';
          this.verified = accepted;
          this.note(accepted ? `done: ${(r.inTolFrac * 100).toFixed(0)}% of measured cells within tolerance (lidar survey)` : this.stats.passes >= 6 ? 'stopped: pass limit reached' : `stopped: no further improvement; ${(r.inTolFrac * 100).toFixed(0)}% of measured cells within tolerance`);
          return this.joints;
        }
      }
      if (this.controller === 'adaptive') {
        // the survey branch may have reset the lanes for a new pass: plan from there
        const b = nextA ?? this.nextAdaptiveBite();
        if (!b) return this.joints;
        this.curBite = b;
        this.biteIdx++;
        this.buildAdaptiveWaypoints(b);
      } else {
        this.curBite = this.bites[this.biteIdx];
        this.buildBiteWaypoints(this.bites[this.biteIdx++]);
      }
      this.cycleStart = this.stats.elapsed;
    }
    const wp = this.queue[0];
    this.phase = wp.phase;
    this.curTilt = wp.tilt ?? 0;
    const ik = solveIkFlexible(this.g, this.cx, this.groundY, this.cz, wp.tip, wp.bucketAbs, this.joints.swing);
    if (!ik.reachable && ik.error > 0.25) {
      this.note(`waypoint unreachable (${ik.error.toFixed(2)} m), skipping`);
      this.queue.shift();
      return this.joints;
    }
    const t = ik.joints;
    let done = true;
    const move = (key: keyof JointState) => {
      const cur = this.joints[key], tgt = t[key];
      const maxd = this.rates[key] * dt * (key === 'boom' || key === 'arm' ? this.reliefScale : 1);
      const d = tgt - cur;
      if (Math.abs(d) > maxd) { this.joints[key] = cur + Math.sign(d) * maxd; done = false; }
      else this.joints[key] = tgt;
    };
    move('swing'); move('boom'); move('arm'); move('bucket');
    const digging = wp.phase === 'penetrate' || wp.phase === 'drag' || wp.phase === 'curl';
    if (this.externalSoil) {
      // soil lives outside (e.g. MPM): record the bite's poses, hand the whole bite over when the dig phases end
      if (digging) this.traceTick(dt, this.surrogatePose(wp.phase));
      else if (this.trace) { this.pendingBite = this.trace; if (this.traceRec) this.traceRec(this.trace); this.trace = null; }
    } else if (this.neural && digging) this.neuralTick(dt, wp.phase);
    else {
      if (this.win || this.prevPose) this.endNeuralBite();
      if (this.traceRec && digging) this.traceTick(dt, this.surrogatePose(wp.phase));
      else if (this.trace) this.endTrace();
      if (wp.cut) this.cutAtTip();
    }
    if (this.neural && !digging && this.trace) this.endTrace();
    if (done && wp.dwell && wp.dwell > 0) { wp.dwell -= dt; done = false; }
    if (done) {
      this.queue.shift();
      if (wp.phase === 'dump') {
        if (this.controller === 'adaptive') this.adaptAfterBite();
        const v = this.hf.dump(this.spoil[0], this.spoil[1], this.load, 1.0);
        this.fills.push(this.load / this.bucketCapacity);
        this.load = Math.max(0, this.load - v);
        this.stats.cycles++;
        this.cycleTimes.push(this.stats.elapsed - this.cycleStart);
        const n = this.cycleTimes.length;
        this.stats.avgCycleS = this.cycleTimes.reduce((a, b) => a + b, 0) / n;
        this.stats.avgFill = this.fills.reduce((a, b) => a + b, 0) / this.fills.length;
        this.stats.productivity = this.stats.elapsed > 0 ? (this.stats.bankRemoved / this.stats.elapsed) * 3600 : 0;
      }
    }
    return this.joints;
  }

  /** remove material above the bucket's cutting edge across its width */
  private cutAtTip() {
    if (this.load >= this.bucketCapacity) return;
    const fk = forwardPlanar(this.g, this.joints);
    const [tx, ty, tz] = planarToWorld(this.cx, this.groundY, this.cz, this.joints.swing, fk.tip[0], fk.tip[1]);
    const sx = -Math.sin(this.joints.swing), sz = Math.cos(this.joints.swing); // lateral axis
    const lanes = 4;
    let bank = 0, loose = 0;
    const before = this.hf.ledger.carried;
    for (let i = 0; i < lanes; i++) {
      const o = -this.bucketWidth / 2 + (this.bucketWidth * (i + 0.5)) / lanes;
      const x = tx + sx * o, z = tz + sz * o;
      const h = this.hf.heightAt(x, z);
      const ey = ty + Math.tan(this.curTilt) * o; // tilted cutting edge height at this lane
      const depth = h - ey;
      if (depth <= 0.005) continue;
      const r = this.designClamp ? this.hf.dig(x, z, 0.32, Math.min(depth, 0.25), true, true) : this.hf.dig(x, z, 0.32, Math.min(depth, 0.25), false, false, ey);
      bank += r.bank; loose += r.loose;
    }
    const got = this.hf.ledger.carried - before;
    this.load += got;
    this.stats.bankRemoved += bank;
    void loose;
  }

  /**
   * Surrogate pose for the current joints: the training rig's cutting edge (tool-frame lip at (-0.9, -0.9) from its
   * pivot) is placed exactly at LiveDirt's bucket tip; pitch is 0 while penetrating/dragging and follows curl progress
   * during the curl (0 -> -75 deg). LiveDirt's own bucket angle is not used because its curl rotates the tip the other
   * way from the training rig. Shared by the model window and the MPM trace exporter so both see the same poses.
   */
  private curlAk0: number | null = null;
  private curBite: Bite | null = null;
  /** 'physical' (default): teeth-first penetration at +40 deg, attack angle easing 40 -> 10 deg over the drag, curl
   * 10 -> -75 deg (digbench/attitude.py). 'legacy': pitch 0 (flat floor) until the curl, curl 0 -> -75 deg. */
  attitude: 'physical' | 'legacy' = 'physical';
  private surrogatePose(phase: AutoPhase) {
    const fk = forwardPlanar(this.g, this.joints);
    const rt = fk.tip[0], yt = this.groundY + fk.tip[1];
    const phys = this.attitude === 'physical';
    const P0 = phys ? 40 * DEG : 0, P1 = phys ? 10 * DEG : 0, P2 = -75 * DEG;
    let pitch = P0;
    if (phase === 'curl') {
      if (this.curlAk0 === null) this.curlAk0 = fk.bucketAbs;
      const span = 20 * DEG - this.curlAk0;
      const prog = span > 1e-3 ? Math.max(0, Math.min(1, (fk.bucketAbs - this.curlAk0) / span)) : 1;
      pitch = P1 + (P2 - P1) * prog;
    } else {
      this.curlAk0 = null;
      if (phase === 'drag' && this.curBite) {
        const [tx, , tz] = planarToWorld(this.cx, this.groundY, this.cz, this.joints.swing, fk.tip[0], fk.tip[1]);
        const a = this.curBite.a, b = this.curBite.b, ex = b[0] - a[0], ez = b[1] - a[1];
        const u = Math.max(0, Math.min(1, ((tx - a[0]) * ex + (tz - a[1]) * ez) / (ex * ex + ez * ez || 1)));
        pitch = P0 + (P1 - P0) * u;
      }
    }
    const ca = Math.cos(pitch), sa = Math.sin(pitch);
    const lipX = ca * -0.9 - sa * -0.9, lipZ = sa * -0.9 + ca * -0.9;
    return { px: rt - lipX, pz: yt - lipZ, pitch }; // pivot: radial (m from swing axis), world height, pitch
  }

  /** per-actuator demand of a soil wrench (rig pivot frame) on the current linkage pose */
  lastDemand: ActuatorDemand | null = null;
  peakDemand: Record<string, number> = {};
  private actuatorLoad(force: [number, number, number], pivotR: number, pivotY: number): ActuatorDemand {
    const fk = forwardPlanar(this.g, this.joints);
    const gy = 0; // boom-plane y is already height above the track ground
    const d = actuatorDemand(force[0], force[1], force[2], [pivotR, pivotY - this.groundY], [fk.wrist[0], fk.wrist[1] + gy], [fk.elbow[0], fk.elbow[1] + gy], [fk.foot[0], fk.foot[1] + gy], [fk.tip[0], fk.tip[1]]);
    this.lastDemand = d;
    for (const k of ['bucket', 'arm', 'boom', 'frontLiftoff', 'forwardTip', 'sliding'] as const) this.peakDemand[k] = Math.max(this.peakDemand[k] ?? 0, d[k]);
    return d;
  }

  /** model-step queue: poses at exact 0.1 s boundaries (interpolated inside a frame when a frame spans one) */
  private nq: { px: number; pz: number; pitch: number }[] = [];
  private prevPose: { px: number; pz: number; pitch: number } | null = null;
  private pumping: Promise<void> | null = null;
  maxPoseGapS = 0; // largest pose interval fed to one model step (should be 0.1)
  gateSteps = 0; // model steps where the gate passed (network handled)
  fallbackSteps = 0; // 0.1 s steps handled by cut+relax because the gate failed
  private fallback = false; // current 0.1 s interval is in fallback (cut+relax) mode
  private biteFallback = false; // consistency gate failed: cut+relax for the rest of this bite

  private neuralTick(dt: number, phase: AutoPhase) {
    const P = this.surrogatePose(phase);
    if (this.traceRec) this.traceTick(dt, P);
    if (this.biteFallback) { this.cutAtTip(); this.prevPose = P; return; }
    if (!this.win && !this.fallback) this.anchorWindow(P);
    if (this.fallback) this.cutAtTip();
    const prev = this.prevPose ?? P;
    this.winT += dt;
    while (this.winT >= 0.1 - 1e-9) {
      this.winT -= 0.1;
      const u = dt > 0 ? Math.max(0, Math.min(1, 1 - this.winT / dt)) : 1;
      this.nq.push({ px: prev.px + (P.px - prev.px) * u, pz: prev.pz + (P.pz - prev.pz) * u, pitch: prev.pitch + (P.pitch - prev.pitch) * u });
      this.maxPoseGapS = Math.max(this.maxPoseGapS, 0.1);
    }
    this.prevPose = P;
    if (!this.pumping && this.nq.length) this.pumping = this.pumpNeural().finally(() => { this.pumping = null; });
  }

  private anchorWindow(P: { px: number; pz: number; pitch: number }) {
    const nn = this.neural!;
    this.winR0 = P.px - 2.7; // pivot starts at x = 2.7 m in the window, as in training
    this.win = new NeuralSoilWindow(this.hf, nn.model, nn.mu, this.cx, this.cz, this.joints.swing, this.winR0);
    this.win.floorProject = nn.floor !== false;
    if (nn.stats) this.win.stats = nn.stats;
    this.win.anchor(P.px - this.winR0, P.pz, P.pitch);
  }

  /** run every queued model step in order; the force from each step drives the relief valve before the next */
  async pumpNeural() {
    const nn = this.neural!;
    while (this.nq.length) {
      const q = this.nq.shift()!;
      if (!this.win) this.anchorWindow(q);
      const w = this.win!;
      // physical consistency limits are per bite: the window may not remove more than the bucket's remaining heaped room
      const g = nn.gate && nn.consistency ? { ...nn.gate, maxCarried: 1.2 * Math.max(0, this.bucketCapacity - this.load), maxCreate: 0.2 } : nn.gate;
      const r = await w.step(q.px - this.winR0, q.pz, q.pitch, g);
      if (r.gated) {
        // gate failed: write back what the model did so far and let cut+relax handle this 0.1 s interval
        // (a consistency failure hands the rest of the bite to cut+relax)
        this.fallbackSteps++;
        this.commitWindow();
        this.fallback = true;
        if (r.reason === 'cap' || r.reason === 'create') this.biteFallback = true;
        continue;
      }
      this.fallback = false;
      this.gateSteps++;
      this.soilForce = r.force;
      this.neuralSteps++;
      this.peakDragN = Math.max(this.peakDragN, r.force[0]);
      // relief: per-actuator limits (cylinders at relief pressure, front lift-off, tip, sliding); proportional
      // back-off above any limit, recover below. fMax (single drag cap) is kept only for the legacy ablation.
      const over = nn.legacyRelief ? r.force[0] > nn.fMax : this.actuatorLoad(r.force, q.px, q.pz).maxFrac > 1;
      this.reliefScale = over ? Math.max(0.05, this.reliefScale * 0.7) : Math.min(1, this.reliefScale + 0.1);
    }
  }

  /** write the window back to the heightfield; the volume the model removed becomes the bucket load */
  private commitWindow() {
    const w = this.win;
    if (!w) return;
    w.commit();
    this.neuralClipped += w.clipped;
    this.neuralFloored += w.floored;
    this.neuralFloorAddedM3 += w.floorAddedM3;
    this.neuralModelDvM3 += w.modelDvM3;
    // mass accounting: what the model removed goes into the bucket (and the heightfield ledger, so dump() can place it);
    // anything beyond the bucket's room spills back onto the terrain at the cutting edge instead of vanishing
    const removed = Math.max(0, w.carried);
    const got = Math.min(Math.max(0, this.bucketCapacity - this.load), removed);
    this.load += got;
    this.hf.ledger.carried += got;
    this.hf.ledger.bankCut += got;
    this.stats.bankRemoved += got;
    this.neuralNetRemoved += w.carried;
    const excess = removed - got;
    if (excess > 1e-6) {
      const fk = forwardPlanar(this.g, this.joints);
      const [tx, , tz] = planarToWorld(this.cx, this.groundY, this.cz, this.joints.swing, fk.tip[0], fk.tip[1]);
      this.neuralSpilled += this.hf.spill(tx, tz, excess, 0.5);
    }
    this.win = null;
  }
  neuralNetRemoved = 0;
  neuralFloorAddedM3 = 0; // volume created by the floor projection (all windows)
  neuralModelDvM3 = 0; // net volume change from model dh (all windows) // sum of window volume removed by the model (negative = net material created)

  /** end of a dig (left penetrate/drag/curl): flush queued steps, commit, reset per-bite state */
  private endNeuralBite() {
    if (this.win) this.commitWindow();
    this.fallback = false;
    this.biteFallback = false;
    this.prevPose = null;
    this.winT = 0;
    this.reliefScale = 1;
  }

  /** await any queued model steps (headless benches call this every frame for exact 0.1 s synchronous stepping) */
  async settleNeural() { while (this.pumping) await this.pumping; }

  // ---------------- MPM replay trace export (DigBench data generation, DAgger) ----------------
  /** when set, every dig (penetrate..curl) is recorded with its anchored terrain and 0.1 s surrogate poses */
  traceRec: ((rec: DigTrace) => void) | null = null;
  /** external soil backend: no heightfield cutting; each finished bite is left in pendingBite for the caller */
  externalSoil = false;
  pendingBite: DigTrace | null = null;
  /**
   * Apply an external backend's bite result: `delta` (surface change, m) on the trace's bed grid (cell, margin, nx x ny,
   * window frame of the trace) is written to the heightfield; `carried` (bank m^3) becomes the bucket load.
   */
  applyExternal(tr: DigTrace, r: { delta: number[]; nx: number; ny: number; cell: number; margin: number; carried: number; completion?: number }) {
    this.lastBiteCompletion = r.completion ?? 1;
    const hf = this.hf, sw = tr.swing ?? this.joints.swing;
    const ux = Math.cos(sw), uz = Math.sin(sw), vx = -uz, vz = ux;
    const C = r.cell, M = r.margin;
    const sample = (a: number, b: number) => {
      const fi = (a + M) / C - 0.5, fj = (b + 1.0 + M) / C - 0.5;
      const i0 = Math.floor(fi), j0 = Math.floor(fj), ti = fi - i0, tj = fj - j0;
      let acc = 0, wsum = 0;
      for (const [di, dj, w] of [[0, 0, (1 - ti) * (1 - tj)], [1, 0, ti * (1 - tj)], [0, 1, (1 - ti) * tj], [1, 1, ti * tj]] as const) {
        const ii = i0 + di, jj = j0 + dj;
        if (ii < 0 || jj < 0 || ii >= r.nx || jj >= r.ny) continue;
        acc += w * r.delta[ii * r.ny + jj]; wsum += w;
      }
      return wsum > 0 ? acc / wsum : 0;
    };
    // every heightfield cell whose centre lies inside the bed
    const x0 = -M, x1 = 3.4 + M, y0 = -1.0 - M, y1 = 1.0 + M;
    const corners = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]].map(([a, b]) => [this.cx + ux * (tr.r0 + a) + vx * b, this.cz + uz * (tr.r0 + a) + vz * b]);
    const xs = corners.map((c) => c[0]), zs = corners.map((c) => c[1]);
    const [ia, ja] = hf.cellOf(Math.min(...xs), Math.min(...zs)), [ib, jb] = hf.cellOf(Math.max(...xs), Math.max(...zs));
    let removed = 0;
    for (let j = ja; j <= jb; j++)
      for (let i = ia; i <= ib; i++) {
        const [px, pz] = hf.center(i, j);
        const dx = px - this.cx, dz = pz - this.cz;
        const a = dx * ux + dz * uz - tr.r0, b = dx * vx + dz * vz;
        if (a < x0 || a > x1 || b < y0 || b > y1) continue;
        const d = sample(a, b);
        if (d !== 0) { hf.applyDelta(i, j, d); removed -= d * hf.area; }
      }
    const bank = Math.max(0, r.carried);
    const loose = bank * (hf.soil?.swell ?? 1);
    this.load += loose;
    hf.ledger.carried += loose;
    hf.ledger.bankCut += bank;
    this.stats.bankRemoved += bank;
    this.externalLog.push({ bite: tr.bite, pass: tr.pass, carried: +bank.toFixed(4), netRemoved: +removed.toFixed(4) });
  }
  externalLog: { bite: number; pass: number; carried: number; netRemoved: number }[] = [];
  traceMargin = 1.5;
  private trace: DigTrace | null = null;
  private traceT = 0;
  private traceShift = 0;
  private traceR0 = 0;
  private traceTick(dt: number, P: { px: number; pz: number; pitch: number }) {
    if (!this.trace) {
      const M = this.traceMargin, C = 0.05;
      const r0 = P.px - 2.7, sw = this.joints.swing;
      const ux = Math.cos(sw), uz = Math.sin(sw), vx = -uz, vz = ux;
      const at = (xw: number, lat: number) => this.hf.heightAt(this.cx + ux * (r0 + xw) + vx * lat, this.cz + uz * (r0 + xw) + vz * lat);
      // anchor exactly as NeuralSoilWindow.anchor: shift = 1.2 - mean(window heights)
      let sum = 0;
      for (let i = 0; i < 68; i++) for (let j = 0; j < 40; j++) sum += at((i + 0.5) * C, -1.0 + (j + 0.5) * C);
      const shift = 1.2 - sum / (68 * 40);
      const nx = Math.round((3.4 + 2 * M) / C), ny = Math.round((2.0 + 2 * M) / C);
      const terr = new Array<number>(nx * ny);
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) terr[i * ny + j] = +(at(-M + (i + 0.5) * C, -1.0 - M + (j + 0.5) * C) + shift).toFixed(4);
      this.trace = { margin: M, cell: C, nx, ny, terrain: terr, poses: [], bite: this.biteIdx, pass: this.stats.passes, t0: this.stats.elapsed, soil: this.hf.soil?.id ?? '', neural: !!this.neural, r0: +r0.toFixed(4), groundY: +this.groundY.toFixed(4), shift: +shift.toFixed(4), attitude: this.attitude, swing: +sw.toFixed(6) };
      this.traceShift = shift; this.traceR0 = r0; this.traceT = 0;
      this.trace.poses.push([0, +(P.px - r0).toFixed(4), +(P.pz + shift).toFixed(4), +P.pitch.toFixed(4), +this.reliefScale.toFixed(3), +this.curTilt.toFixed(4)]);
    }
    this.traceT += dt;
    if (this.traceT >= 0.1 - 1e-9) {
      this.traceT -= 0.1;
      const tr = this.trace;
      tr.poses.push([+(tr.poses.length * 0.1).toFixed(2), +(P.px - this.traceR0).toFixed(4), +(P.pz + this.traceShift).toFixed(4), +P.pitch.toFixed(4), +this.reliefScale.toFixed(3), +this.curTilt.toFixed(4)]);
    }
  }
  private endTrace() { if (this.trace && this.traceRec) this.traceRec(this.trace); this.trace = null; }

  get bucketLoad() { return this.load; }
  get bitesTotal() { return this.bites.length; }
  get biteIndex() { return this.biteIdx; }
}
