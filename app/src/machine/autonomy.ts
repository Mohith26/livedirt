import { solveIkFlexible, forwardPlanar, planarToWorld, type JointState, type LinkageGeometry } from './kinematics';
import type { Heightfield, DesignTrench } from '../sim/heightfield';

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

interface Waypoint { tip: [number, number, number]; bucketAbs: number; phase: AutoPhase; cut?: boolean; dwell?: number }

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
    this.note(`plan: ${lanes} lane(s) x ${nb} bites, trench ${len.toFixed(1)} m, spoil ${setback.toFixed(1)} m off centreline`);
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

    if (this.queue.length === 0) {
      if (this.biteIdx >= this.bites.length) {
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
          this.note(`pass ${this.stats.passes}`);
        } else {
          this.phase = 'done';
          this.verified = accepted;
          this.note(accepted ? `done: ${(r.inTolFrac * 100).toFixed(0)}% of measured cells within tolerance (lidar survey)` : this.stats.passes >= 6 ? 'stopped: pass limit reached' : `stopped: no further improvement; ${(r.inTolFrac * 100).toFixed(0)}% of measured cells within tolerance`);
          return this.joints;
        }
      }
      this.buildBiteWaypoints(this.bites[this.biteIdx++]);
      this.cycleStart = this.stats.elapsed;
    }
    const wp = this.queue[0];
    this.phase = wp.phase;
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
      const maxd = this.rates[key] * dt;
      const d = tgt - cur;
      if (Math.abs(d) > maxd) { this.joints[key] = cur + Math.sign(d) * maxd; done = false; }
      else this.joints[key] = tgt;
    };
    move('swing'); move('boom'); move('arm'); move('bucket');
    if (wp.cut) this.cutAtTip();
    if (done && wp.dwell && wp.dwell > 0) { wp.dwell -= dt; done = false; }
    if (done) {
      this.queue.shift();
      if (wp.phase === 'dump') {
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
      const depth = h - ty;
      if (depth <= 0.005) continue;
      const r = this.hf.dig(x, z, 0.32, Math.min(depth, 0.25), true, true);
      bank += r.bank; loose += r.loose;
    }
    const got = this.hf.ledger.carried - before;
    this.load += got;
    this.stats.bankRemoved += bank;
    void loose;
  }

  get bucketLoad() { return this.load; }
  get bitesTotal() { return this.bites.length; }
  get biteIndex() { return this.biteIdx; }
}
