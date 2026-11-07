import { describe, it, expect } from 'vitest';
import { HeightRaycaster } from '../src/sim/heightRay';
import { scan, OS1_64_LIKE, PointLabel, rayCapsule, beamElevations } from '../src/sensors/lidarModel';

const flatRay = (N = 200, cell = 0.1, h = 0) => new HeightRaycaster(N, cell, (N * cell) / 2, new Float32Array(N * N).fill(h));

describe('HeightRaycaster', () => {
  it('hits a flat plane where geometry says it should', () => {
    const r = flatRay(200, 0.1, 1);
    // from (0,3,0) heading down at 45 deg along +x: hits y=1 at x=2
    const d = Math.SQRT1_2;
    const hit = r.cast(0, 3, 0, d, -d, 0, 50)!;
    expect(hit).not.toBeNull();
    expect(hit.x).toBeCloseTo(2, 3);
    expect(hit.y).toBeCloseTo(1, 3);
    expect(hit.ny).toBeCloseTo(1, 5);
  });
  it('agrees with brute-force marching on a random bumpy field', () => {
    const N = 128, cell = 0.1, half = 6.4; const s = new Float32Array(N * N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) s[j * N + i] = 0.6 * Math.sin(i * 0.21) * Math.cos(j * 0.17) + 0.3 * Math.sin(i * 0.05 + j * 0.09);
    const r = new HeightRaycaster(N, cell, half, s);
    let agree = 0, total = 0;
    for (let q = 0; q < 300; q++) {
      const ox = -5 + (q % 17) * 0.6, oz = -5 + ((q * 7) % 19) * 0.5, oy = 2.5;
      const az = q * 0.37, el = -0.15 - (q % 9) * 0.08;
      const dx = Math.cos(el) * Math.cos(az), dy = Math.sin(el), dz = Math.cos(el) * Math.sin(az);
      const fast = r.cast(ox, oy, oz, dx, dy, dz, 30);
      // brute force
      let brute: number | null = null;
      for (let t = 0; t < 30; t += 0.002) { const x = ox + dx * t, z = oz + dz * t; if (Math.abs(x) > half - cell || Math.abs(z) > half - cell) break; if (oy + dy * t <= r.height(x, z)) { brute = t; break; } }
      total++;
      if ((fast === null && brute === null) || (fast && brute !== null && Math.abs(fast.t - brute) < 0.01)) agree++;
    }
    expect(agree / total).toBeGreaterThan(0.97);
  });
});

describe('lidar model', () => {
  it('beam elevations span the vertical FOV', () => {
    const e = beamElevations(OS1_64_LIKE);
    expect(e.length).toBe(64);
    expect((e[0] * 180) / Math.PI).toBeCloseTo(22.5, 5);
    expect((e[63] * 180) / Math.PI).toBeCloseTo(-22.5, 5);
  });
  it('ray-capsule intersection', () => {
    const t = rayCapsule(0, 1, 0, 1, 0, 0, { x: 5, z: 0, y0: 0.3, y1: 1.5, r: 0.25, label: PointLabel.Person });
    expect(t).toBeCloseTo(4.75, 4);
  });
  it('produces a deterministic scan with labelled person returns and realistic ground ring geometry', () => {
    const world = { ray: flatRay(400, 0.1, 0), capsules: [{ x: 6, z: 0, y0: 0.3, y1: 1.5, r: 0.25, label: PointLabel.Person }], dustDensity: 0, rain: 0 };
    const pose = { x: 0, y: 3, z: 0, yaw: 0 };
    const a = scan(world, pose, OS1_64_LIKE, 42, 0, 512);
    const b = scan(world, pose, OS1_64_LIKE, 42, 0, 512);
    expect(a.count).toBe(b.count);
    expect(Array.from(a.points.slice(0, 40))).toEqual(Array.from(b.points.slice(0, 40)));
    let person = 0; for (let i = 0; i < a.count; i++) if (a.label[i] === PointLabel.Person) person++;
    expect(person).toBeGreaterThan(5);
    // lowest ring (-22.5 deg from 3 m) should hit ground at ~7.24 m radius
    let rs: number[] = []; for (let i = 0; i < a.count; i++) if (a.ring[i] === 63 && a.label[i] === PointLabel.Terrain) rs.push(Math.hypot(a.points[i * 4], a.points[i * 4 + 2]));
    rs.sort((x, y) => x - y); const med = rs[rs.length >> 1];
    expect(med).toBeGreaterThan(7.0); expect(med).toBeLessThan(7.5);
  });
  it('dust adds early labelled returns', () => {
    const world = { ray: flatRay(400, 0.1, 0), capsules: [], dustDensity: 1, rain: 0 };
    const s = scan(world, { x: 0, y: 3, z: 0, yaw: 0 }, OS1_64_LIKE, 7, 0, 256);
    let dust = 0; for (let i = 0; i < s.count; i++) if (s.label[i] === PointLabel.Dust) dust++;
    expect(dust).toBeGreaterThan(50);
  });
});
