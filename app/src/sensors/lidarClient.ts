import { OS1_64_LIKE, type LidarSpec, type LidarPose, type Capsule, PointLabel } from './lidarModel';

export interface ScanResult {
  points: Float32Array; ring: Uint8Array; label: Uint8Array; col: Uint16Array;
  count: number; rays: number; dropped: number; noHit: number; t: number; ms: number;
  pose: LidarPose; hRes: number;
}

/** main-thread handle to the lidar worker; keeps at most one scan in flight */
export class LidarClient {
  private w: Worker;
  private busy = false;
  private id = 0;
  private pendingPose: LidarPose | null = null;
  private pendingHRes = 512;
  spec: LidarSpec = { ...OS1_64_LIKE };
  hRes = 1024;
  enabled = true;
  last: ScanResult | null = null;
  onScan: ((s: ScanResult) => void) | null = null;
  private acc = 0;
  private seed = 1;

  constructor(N: number, cell: number, half: number, surf: Float32Array, obstacleMask: Uint8Array, albedo: Float32Array) {
    this.w = new Worker(new URL('./lidar.worker.ts', import.meta.url), { type: 'module' });
    this.w.postMessage({ type: 'init', N, cell, half, surf: surf.slice(), obstacleMask, albedo });
    this.w.onmessage = (e) => {
      const m = e.data;
      if (m.type !== 'scan') return;
      this.busy = false;
      const res: ScanResult = { ...m, pose: this.pendingPose!, hRes: this.pendingHRes };
      this.last = res;
      this.onScan?.(res);
    };
  }

  updateSurface(surf: Float32Array) {
    this.w.postMessage({ type: 'surface', surf: surf.slice() });
  }

  /** call every frame; issues a scan at the configured rate when the worker is free */
  tick(dt: number, t: number, pose: LidarPose, capsules: Capsule[], dust: number, rain: number) {
    if (!this.enabled) return;
    this.acc += dt;
    if (this.busy || this.acc < 1 / this.spec.rateHz) return;
    this.acc = 0;
    this.busy = true;
    this.pendingPose = pose;
    this.pendingHRes = this.hRes;
    this.w.postMessage({ type: 'scan', id: ++this.id, pose, spec: this.spec, capsules, dust, rain, seed: this.seed++, t, hRes: this.hRes });
  }
}

export const LABEL_NAMES: Record<number, string> = {
  [PointLabel.Terrain]: 'terrain', [PointLabel.Obstacle]: 'obstacle', [PointLabel.Person]: 'person', [PointLabel.Dust]: 'dust', [PointLabel.Rain]: 'rain',
};
