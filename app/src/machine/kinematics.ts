/**
 * Excavator front-linkage kinematics (planar boom / arm / bucket chain on a swinging upper structure).
 *
 * Dimensions default to a Develon DX225LC-7 class machine (spec sheet: boom 5.7 m, arm 2.9 m). The boom
 * foot pin position and joint limits are typical values for this class, not manufacturer data.
 *
 * Frames:
 *   world:   Y up, the machine's swing centre at (cx, groundY, cz)
 *   machine: swing yaw `swing` (radians, CCW seen from above, measured from +X toward +Z = atan2(z, x))
 *   boom plane: r = horizontal distance from the swing axis along the heading, y = height above ground
 * Joint angles (radians):
 *   boom   b: boom line angle above horizontal
 *   arm    a: arm line angle relative to the boom line (negative folds the arm under)
 *   bucket k: bucket tip line angle relative to the arm line
 */
export interface LinkageGeometry {
  boomFootR: number; // boom foot pin forward of swing axis
  boomFootY: number; // boom foot pin height above ground
  boom: number;
  arm: number;
  bucket: number; // pin-to-tip radius
  limits: { boom: [number, number]; arm: [number, number]; bucket: [number, number]; };
}

export const DX225_GEOMETRY: LinkageGeometry = {
  boomFootR: 0.18,
  boomFootY: 2.05,
  boom: 5.7,
  arm: 2.9,
  bucket: 1.45,
  limits: {
    boom: [(-58 * Math.PI) / 180, (68 * Math.PI) / 180],
    arm: [(-158 * Math.PI) / 180, (-28 * Math.PI) / 180],
    bucket: [(-170 * Math.PI) / 180, (35 * Math.PI) / 180],
  },
};

export interface JointState { swing: number; boom: number; arm: number; bucket: number }

export interface PlanarPose { foot: [number, number]; elbow: [number, number]; wrist: [number, number]; tip: [number, number]; bucketAbs: number }

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** forward kinematics in the boom plane */
export function forwardPlanar(g: LinkageGeometry, j: JointState): PlanarPose {
  const foot: [number, number] = [g.boomFootR, g.boomFootY];
  const ab = j.boom;
  const elbow: [number, number] = [foot[0] + g.boom * Math.cos(ab), foot[1] + g.boom * Math.sin(ab)];
  const aa = ab + j.arm;
  const wrist: [number, number] = [elbow[0] + g.arm * Math.cos(aa), elbow[1] + g.arm * Math.sin(aa)];
  const ak = aa + j.bucket;
  const tip: [number, number] = [wrist[0] + g.bucket * Math.cos(ak), wrist[1] + g.bucket * Math.sin(ak)];
  return { foot, elbow, wrist, tip, bucketAbs: ak };
}

/** world-space position of a boom-plane point (r, y) */
export function planarToWorld(cx: number, groundY: number, cz: number, swing: number, r: number, y: number): [number, number, number] {
  return [cx + r * Math.cos(swing), groundY + y, cz + r * Math.sin(swing)];
}

export interface IkResult { joints: JointState; reachable: boolean; error: number }

/**
 * Inverse kinematics: place the bucket tip at a world target with a given absolute bucket angle
 * (angle of the pin->tip line above horizontal; about -90 deg is "teeth pointing down").
 * Solves swing analytically, then the boom/arm 2-link problem for the wrist (elbow-up branch,
 * which is how excavators operate), clamps to joint limits and reports residual error.
 */
export function solveIk(g: LinkageGeometry, cx: number, groundY: number, cz: number, target: [number, number, number], bucketAbs: number, prevSwing?: number): IkResult {
  const dx = target[0] - cx, dz = target[2] - cz;
  let swing = Math.atan2(dz, dx);
  if (prevSwing !== undefined) {
    // unwrap toward the previous swing to avoid 2*pi jumps
    while (swing - prevSwing > Math.PI) swing -= 2 * Math.PI;
    while (swing - prevSwing < -Math.PI) swing += 2 * Math.PI;
  }
  const r = Math.hypot(dx, dz);
  const y = target[1] - groundY;
  const wr = r - g.bucket * Math.cos(bucketAbs);
  const wy = y - g.bucket * Math.sin(bucketAbs);
  const px = wr - g.boomFootR, py = wy - g.boomFootY;
  const d = Math.hypot(px, py);
  const L1 = g.boom, L2 = g.arm;
  let reachable = d <= L1 + L2 - 1e-6 && d >= Math.abs(L1 - L2) + 1e-6;
  const dc = clamp(d, Math.abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6);
  const cosA = clamp((dc * dc - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1);
  const arm = -Math.acos(cosA); // elbow-up: arm folds downward relative to the boom
  const boom = Math.atan2(py, px) - Math.atan2(L2 * Math.sin(arm), L1 + L2 * Math.cos(arm));
  const bucket = bucketAbs - boom - arm;
  const j: JointState = {
    swing,
    boom: clamp(boom, g.limits.boom[0], g.limits.boom[1]),
    arm: clamp(arm, g.limits.arm[0], g.limits.arm[1]),
    bucket: clamp(wrapPi(bucket), g.limits.bucket[0], g.limits.bucket[1]),
  };
  const fk = forwardPlanar(g, j);
  const tw = planarToWorld(cx, groundY, cz, swing, fk.tip[0], fk.tip[1]);
  const error = Math.hypot(tw[0] - target[0], tw[1] - target[1], tw[2] - target[2]);
  if (error > 0.03) reachable = false;
  return { joints: j, reachable, error };
}

export function wrapPi(a: number) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}

/** maximum horizontal reach at a given height (approximate, bucket aligned with the arm) */
export function reachAtHeight(g: LinkageGeometry, y: number): number {
  const L = g.boom + g.arm + g.bucket;
  const dy = y - g.boomFootY;
  return g.boomFootR + Math.sqrt(Math.max(0, L * L - dy * dy));
}

/**
 * IK with a preferred bucket angle: if the preferred attitude cannot reach the target, search nearby
 * attitudes (operators extend the bucket at long reach) and take the reachable one closest to the preference.
 */
export function solveIkFlexible(g: LinkageGeometry, cx: number, groundY: number, cz: number, target: [number, number, number], preferred: number, prevSwing?: number, range = 100 * Math.PI / 180): IkResult & { bucketAbs: number } {
  const first = solveIk(g, cx, groundY, cz, target, preferred, prevSwing);
  if (first.reachable) return { ...first, bucketAbs: preferred };
  let best = { ...first, bucketAbs: preferred };
  const step = 4 * Math.PI / 180;
  for (let d = step; d <= range; d += step) {
    for (const sgn of [1, -1]) {
      const a = preferred + sgn * d;
      const r = solveIk(g, cx, groundY, cz, target, a, prevSwing);
      if (r.reachable) return { ...r, bucketAbs: a };
      if (r.error < best.error) best = { ...r, bucketAbs: a };
    }
  }
  return best;
}
