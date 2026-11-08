/**
 * DX225LC-7 per-actuator limits and machine stability for a soil wrench on the bucket. Mirror of
 * digbench/actuators.py (see there for sources and assumptions): cylinder torque capacities from the implement relief
 * pressure with moment arms calibrated to the ISO digging forces (bucket 153.0 kN x 1.45 m, arm 106.9 kN x 4.35 m),
 * retract side scaled by annulus/bore; boom moment arm 0.65 m is an ASSUMPTION; machine 23.6 t with its CG on the
 * swing axis (ASSUMPTION), tipping edges at +/- 1.825 m (tumbler distance 3.65 m), track traction 0.6 (ASSUMPTION).
 * Boom plane: r away from the machine, y up from the track ground, CCW positive.
 */
const P_RELIEF = 34.3e6;
const G = 9.81;
const ann = (bore: number, rod: number) => 1 - (rod / bore) ** 2;
const T_BUCKET_EXT = 153.0e3 * 1.45;
const T_ARM_EXT = 106.9e3 * 4.35;
const T_BUCKET_RET = T_BUCKET_EXT * ann(120, 80);
const T_ARM_RET = T_ARM_EXT * ann(140, 100);
const R_BOOM = 0.65;
const T_BOOM_EXT = P_RELIEF * 2 * (Math.PI / 4) * 0.125 ** 2 * R_BOOM;
const T_BOOM_RET = T_BOOM_EXT * ann(125, 85);
const MASS = 23600;
const HALF_TRACK = 3.65 / 2;
const MU_TRACK = 0.6;

export interface ActuatorDemand { bucket: number; arm: number; boom: number; frontLiftoff: number; forwardTip: number; sliding: number; maxFrac: number; governing: string }

const frac = (M: number, ext: number, ret: number, extendPositive = true) => ((M > 0) === extendPositive ? Math.abs(M) / ext : Math.abs(M) / ret);

/** soil force (Fr, Fy) N with torque T (N m) about `pivot`; pins: wrist, elbow, foot (all [r, y] in the boom plane) */
export function actuatorDemand(Fr: number, Fy: number, T: number, pivot: [number, number], wrist: [number, number], elbow: [number, number], foot: [number, number], bucket: [number, number]): ActuatorDemand {
  const m = (q: [number, number]) => T + (pivot[0] - q[0]) * Fy - (pivot[1] - q[1]) * Fr;
  const W = MASS * G;
  const d: ActuatorDemand = {
    bucket: frac(m(wrist), T_BUCKET_EXT, T_BUCKET_RET),
    arm: frac(m(elbow), T_ARM_EXT, T_ARM_RET),
    boom: frac(m(foot), T_BOOM_EXT, T_BOOM_RET, false),
    frontLiftoff: ((bucket[0] + HALF_TRACK) * Fy - bucket[1] * Fr + T) / (W * HALF_TRACK),
    forwardTip: -((bucket[0] - HALF_TRACK) * Fy - bucket[1] * Fr + T) / (W * HALF_TRACK),
    sliding: Math.abs(Fr) / (MU_TRACK * W),
    maxFrac: 0, governing: '',
  };
  for (const k of ['bucket', 'arm', 'boom', 'frontLiftoff', 'forwardTip', 'sliding'] as const) if (d[k] > d.maxFrac) { d.maxFrac = d[k]; d.governing = k; }
  return d;
}
