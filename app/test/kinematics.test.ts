import { describe, it, expect } from 'vitest';
import { DX225_GEOMETRY as G, forwardPlanar, solveIk, planarToWorld, reachAtHeight } from '../src/machine/kinematics';

describe('excavator kinematics', () => {
  it('FK -> IK round trip recovers the tip position across the working envelope', () => {
    let worst = 0, n = 0;
    for (let b = -0.6; b <= 1.0; b += 0.2)
      for (let a = -2.5; a <= -0.6; a += 0.25)
        for (let k = -2.4; k <= 0.4; k += 0.4) {
          const j = { swing: 0.7, boom: b, arm: a, bucket: k };
          const fk = forwardPlanar(G, j);
          const tip = planarToWorld(1, 2, -3, j.swing, fk.tip[0], fk.tip[1]);
          const ik = solveIk(G, 1, 2, -3, tip, fk.bucketAbs, 0.7);
          if (!ik.reachable) continue;
          worst = Math.max(worst, ik.error); n++;
        }
    expect(n).toBeGreaterThan(100);
    expect(worst).toBeLessThan(1e-6);
  });
  it('reports unreachable targets and clamps', () => {
    const ik = solveIk(G, 0, 0, 0, [20, 0, 0], -Math.PI / 2);
    expect(ik.reachable).toBe(false);
  });
  it('reach at ground is in the right ballpark for a 22 t machine (spec max reach at ground 9.68 m)', () => {
    const r = reachAtHeight(G, 0);
    expect(r).toBeGreaterThan(9.0);
    expect(r).toBeLessThan(10.5);
  });
  it('can dig 6 m deep at a sensible radius (spec max dig depth 6.59 m)', () => {
    const ik = solveIk(G, 0, 0, 0, [4.0, -6.0, 0], -Math.PI / 2);
    expect(ik.reachable).toBe(true);
  });
});
