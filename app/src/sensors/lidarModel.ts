import { HeightRaycaster } from '../sim/heightRay';

/**
 * Spinning multi-beam lidar model over a heightfield world plus analytic actors.
 *
 * Beam layout follows a 64-channel, 45 deg vertical FOV spinning lidar (Ouster OS1-64 class, uniform
 * vertical spacing). Effects modelled, each with an explicit, documented approximation:
 *   - range noise: zero-mean Gaussian, sigma grows mildly with range
 *   - dropout: probability rises with grazing incidence, range, and low surface albedo
 *   - intensity: albedo * cos(incidence), with a soft range falloff
 *   - airborne dust: per-ray early returns from a homogeneous particle medium (Beer-Lambert extinction)
 *   - rain: sparse short-range drop returns plus mild extinction
 * Every point carries a ground-truth label so perception can be scored against it.
 */
export interface LidarSpec {
  channels: number;
  vFovUpDeg: number;
  vFovDownDeg: number;
  hRes: number;
  rateHz: number;
  minRange: number;
  maxRange: number;
  rangeNoise: number; // sigma at 10 m, metres
}

export const OS1_64_LIKE: LidarSpec = {
  channels: 64,
  vFovUpDeg: 22.5,
  vFovDownDeg: -22.5,
  hRes: 1024,
  rateHz: 10,
  minRange: 0.5,
  maxRange: 90,
  rangeNoise: 0.015,
};

export enum PointLabel { Terrain = 0, Obstacle = 1, Person = 2, Dust = 3, Rain = 4 }

export interface Capsule { x: number; z: number; y0: number; y1: number; r: number; label: PointLabel }

export interface LidarWorld {
  ray: HeightRaycaster;
  /** label per cell for terrain vs obstacle surfaces (1 = obstacle), length N*N */
  obstacleMask?: Uint8Array;
  /** albedo per cell 0..1, length N*N */
  albedo?: Float32Array;
  capsules: Capsule[];
  dustDensity: number; // 0..1 (1 = heavy dust, ~15 m visibility)
  rain: number; // 0..1
}

export interface LidarPose {
  x: number; y: number; z: number;
  yaw: number; // radians, rotation about +Y of the sensor frame
  /** forward tilt, radians (positive = nose down); 0 for a level spinning lidar */
  pitch?: number;
  /** optional self-return crop: ignore hits within this horizontal radius of (cropX, cropZ) and above cropY */
  crop?: { x: number; z: number; r: number; yMin: number };
}

export interface LidarScan {
  /** xyz world + intensity, 4 floats per point */
  points: Float32Array;
  ring: Uint8Array;
  label: Uint8Array;
  /** azimuth column index per point (for range images) */
  col: Uint16Array;
  count: number;
  /** stats */
  rays: number;
  /** returns lost to dropout/attenuation (excludes rays that never hit anything) */
  dropped: number;
  /** rays with no surface within range (open sky) */
  noHit: number;
  timestamp: number;
}

/** small deterministic PRNG (mulberry32) so scans are reproducible for tests and MCAP replays */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(r: () => number) {
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** ray vs vertical capsule (axis along +Y), returns t or -1 */
export function rayCapsule(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, c: Capsule): number {
  // infinite cylinder in xz
  const px = ox - c.x, pz = oz - c.z;
  const a = dx * dx + dz * dz;
  const b = 2 * (px * dx + pz * dz);
  const cc = px * px + pz * pz - c.r * c.r;
  let best = -1;
  if (a > 1e-12) {
    const disc = b * b - 4 * a * cc;
    if (disc >= 0) {
      const t = (-b - Math.sqrt(disc)) / (2 * a);
      const y = oy + dy * t;
      if (t > 0 && y >= c.y0 && y <= c.y1) best = t;
    }
  }
  // end caps as spheres
  for (const yc of [c.y0, c.y1]) {
    const qx = ox - c.x, qy = oy - yc, qz = oz - c.z;
    const bb = qx * dx + qy * dy + qz * dz;
    const c2 = qx * qx + qy * qy + qz * qz - c.r * c.r;
    const d = bb * bb - c2;
    if (d >= 0) {
      const t = -bb - Math.sqrt(d);
      if (t > 0 && (best < 0 || t < best)) best = t;
    }
  }
  return best;
}

export function beamElevations(spec: LidarSpec): Float32Array {
  const e = new Float32Array(spec.channels);
  for (let c = 0; c < spec.channels; c++) {
    const f = spec.channels === 1 ? 0.5 : c / (spec.channels - 1);
    e[c] = ((spec.vFovUpDeg + (spec.vFovDownDeg - spec.vFovUpDeg) * f) * Math.PI) / 180;
  }
  return e;
}

export function scan(world: LidarWorld, pose: LidarPose, spec: LidarSpec, seed: number, timestamp = 0, hResOverride?: number): LidarScan {
  const hRes = hResOverride ?? spec.hRes;
  const rays = spec.channels * hRes;
  const points = new Float32Array(rays * 4);
  const ring = new Uint8Array(rays);
  const label = new Uint8Array(rays);
  const colIdx = new Uint16Array(rays);
  const R = rng(seed);
  const elev = beamElevations(spec);
  const ray = world.ray;
  const N = ray.N, cell = ray.cell, half = ray.half;
  // extinction coefficients (1/m). Heavy dust ~ 15 m meteorological visibility => sigma ~ 3/15.
  const sigmaDust = world.dustDensity * 0.2;
  const sigmaRain = world.rain * 0.01;
  let n = 0, dropped = 0, noHit = 0;
  const ct = Math.cos(pose.pitch ?? 0), st = Math.sin(pose.pitch ?? 0);
  const chh = Math.cos(pose.yaw), shh = Math.sin(pose.yaw);
  for (let h = 0; h < hRes; h++) {
    const az = (2 * Math.PI * h) / hRes; // relative to the sensor's forward axis
    const ca = Math.cos(az), sa = Math.sin(az);
    for (let c = 0; c < spec.channels; c++) {
      const ce = Math.cos(elev[c]), se = Math.sin(elev[c]);
      // sensor frame (forward, up, lateral) -> tilt about lateral -> yaw about up
      const lx = ce * ca, ly = se, lz = ce * sa;
      const tx = lx * ct + ly * st, ty = -lx * st + ly * ct;
      const dx = tx * chh - lz * shh, dy = ty, dz = tx * shh + lz * chh;
      // geometry
      let tHit = Infinity, lab = PointLabel.Terrain, nx = 0, ny = 1, nz = 0, alb = 0.3;
      const hit = ray.cast(pose.x, pose.y, pose.z, dx, dy, dz, spec.maxRange, spec.minRange);
      if (hit) {
        tHit = hit.t; nx = hit.nx; ny = hit.ny; nz = hit.nz;
        const i = Math.min(N - 1, Math.max(0, Math.floor((hit.x + half) / cell)));
        const j = Math.min(N - 1, Math.max(0, Math.floor((hit.z + half) / cell)));
        const k = j * N + i;
        if (world.obstacleMask && world.obstacleMask[k]) lab = PointLabel.Obstacle;
        if (world.albedo) alb = world.albedo[k];
      }
      for (const cap of world.capsules) {
        const t = rayCapsule(pose.x, pose.y, pose.z, dx, dy, dz, cap);
        if (t > spec.minRange && t < tHit) {
          tHit = t; lab = cap.label; alb = 0.55;
          const hx = pose.x + dx * t - cap.x, hz = pose.z + dz * t - cap.z, hl = Math.hypot(hx, hz) || 1;
          nx = hx / hl; ny = 0; nz = hz / hl;
        }
      }
      // atmospheric early returns (sampled along the ray up to the surface)
      const sigma = sigmaDust + sigmaRain;
      if (sigma > 0) {
        const limit = Math.min(tHit, spec.maxRange);
        const tAtm = -Math.log(1 - R() * 0.999999) / sigma;
        if (tAtm > spec.minRange && tAtm < limit) {
          // a particle return happens only if the backscatter is strong enough; most extinction is silent
          const pReturn = sigmaDust > 0 ? 0.35 : 0.15;
          if (R() < pReturn) {
            tHit = tAtm; lab = R() * sigma < sigmaDust ? PointLabel.Dust : PointLabel.Rain; alb = 0.08 + 0.1 * R();
            nx = -dx; ny = -dy; nz = -dz;
          } else if (R() < 0.5) {
            dropped++;
            continue; // signal fully attenuated
          }
        }
      }
      if (!Number.isFinite(tHit)) { noHit++; continue; }
      if (pose.crop) {
        const hx = pose.x + dx * tHit, hz = pose.z + dz * tHit, hy = pose.y + dy * tHit;
        if (Math.hypot(hx - pose.crop.x, hz - pose.crop.z) < pose.crop.r && hy > pose.crop.yMin) { continue; }
      }
      const cosInc = Math.max(0, -(nx * dx + ny * dy + nz * dz));
      // dropout model
      const graze = Math.pow(1 - cosInc, 4);
      const far = Math.pow(tHit / spec.maxRange, 2);
      const dark = Math.max(0, 0.25 - alb) * 1.2;
      const pDrop = Math.min(0.95, 0.01 + 0.55 * graze + 0.5 * far + dark * far * 4);
      if (lab !== PointLabel.Dust && lab !== PointLabel.Rain && R() < pDrop) { dropped++; continue; }
      const sigmaR = spec.rangeNoise * (0.7 + 0.03 * tHit);
      const tm = tHit + gauss(R) * sigmaR;
      const o = n * 4;
      points[o] = pose.x + dx * tm;
      points[o + 1] = pose.y + dy * tm;
      points[o + 2] = pose.z + dz * tm;
      points[o + 3] = Math.min(1, alb * (0.35 + 0.65 * cosInc) * (1 / (1 + tHit / 60)) * 1.6);
      ring[n] = c;
      label[n] = lab;
      colIdx[n] = h;
      n++;
    }
  }
  return { points, ring, label, col: colIdx, count: n, rays, dropped, noHit, timestamp };
}
