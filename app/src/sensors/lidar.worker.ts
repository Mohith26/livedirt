/// <reference lib="webworker" />
import { HeightRaycaster } from '../sim/heightRay';
import { scan, type LidarSpec, type LidarPose, type Capsule } from './lidarModel';

let ray: HeightRaycaster | null = null;
let obstacleMask: Uint8Array | undefined;
let albedo: Float32Array | undefined;

export type LidarRequest =
  | { type: 'init'; N: number; cell: number; half: number; surf: Float32Array; obstacleMask: Uint8Array; albedo: Float32Array }
  | { type: 'surface'; surf: Float32Array }
  | { type: 'scan'; id: number; pose: LidarPose; spec: LidarSpec; capsules: Capsule[]; dust: number; rain: number; seed: number; t: number; hRes: number };

self.onmessage = (e: MessageEvent<LidarRequest>) => {
  const m = e.data;
  if (m.type === 'init') {
    ray = new HeightRaycaster(m.N, m.cell, m.half, m.surf);
    obstacleMask = m.obstacleMask;
    albedo = m.albedo;
  } else if (m.type === 'surface') {
    if (!ray) return;
    ray.surf = m.surf;
    ray.rebuild();
  } else if (m.type === 'scan') {
    if (!ray) return;
    const t0 = performance.now();
    const s = scan({ ray, obstacleMask, albedo, capsules: m.capsules, dustDensity: m.dust, rain: m.rain }, m.pose, m.spec, m.seed, m.t, m.hRes);
    const ms = performance.now() - t0;
    const points = s.points.slice(0, s.count * 4), ring = s.ring.slice(0, s.count), label = s.label.slice(0, s.count), col = s.col.slice(0, s.count);
    (self as unknown as Worker).postMessage({ type: 'scan', id: m.id, points, ring, label, col, count: s.count, rays: s.rays, dropped: s.dropped, noHit: s.noHit, t: m.t, ms }, [points.buffer, ring.buffer, label.buffer, col.buffer]);
  }
};
