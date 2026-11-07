import * as THREE from 'three';
import { YoloxDetector, iou, COCO_PERSON, type Detection } from './detector';
import { scan, OS1_64_LIKE, PointLabel, type Capsule } from '../sensors/lidarModel';
import type { HeightRaycaster } from '../sim/heightRay';

/**
 * Perception sweep: how well do the roof camera (YOLOX on splat renders) and the two lidars see a ground
 * worker, by distance and condition? Ground truth comes from the worker's known pose (projected bounding
 * box + terrain occlusion test). Lidar "detection" is >= MIN_RETURNS person-labelled returns, which is a
 * sensor-visibility measure, not a trained lidar detector.
 */
export interface SweepHost {
  scene: THREE.Scene;
  cabCamera: THREE.PerspectiveCamera;
  cabSize: { w: number; h: number };
  renderCab(): Promise<Uint8Array>;
  setCabPose(yaw: number): void;
  placeWorker(x: number, z: number, faceYaw: number): void;
  hideWorker(): void;
  workerCapsule(): Capsule;
  setCondition(c: Condition): void;
  sensorRay: HeightRaycaster;
  lidarPoses(): { roof: [number, number, number]; front: [number, number, number]; yaw: number };
  heightAt(x: number, z: number): number;
  /** mesh occluders (the machine twin) tested exactly for line of sight */
  occluders: THREE.Object3D[];
}

export interface Condition { id: string; label: string; look: string; rain: number; dust: number }
export const CONDITIONS: Condition[] = [
  { id: 'captured', label: 'Overcast (as captured)', look: 'captured', rain: 0, dust: 0 },
  { id: 'dusk', label: 'Dusk', look: 'dusk', rain: 0, dust: 0 },
  { id: 'rain', label: 'Heavy rain', look: 'captured', rain: 0.85, dust: 0 },
  { id: 'dust', label: 'Heavy dust', look: 'captured', rain: 0, dust: 0.85 },
];
export const DISTANCES = [4, 6, 8, 10, 12, 15, 18, 22];
export const BEARINGS_DEG = [-28, -14, 0, 14];
const MIN_RETURNS = 5;
const SCORE_THR = 0.35;

export interface Sample {
  condition: string; distance: number; bearing: number;
  gtVisible: boolean; gtBox: { x: number; y: number; w: number; h: number } | null; visibleFrac: number;
  camDetected: boolean; camScore: number; camIou: number;
  roofReturns: number; frontReturns: number; lidarDetected: boolean;
}
export interface SweepResult {
  createdAt: string; model: string; scoreThreshold: number; iouThreshold: number; minLidarReturns: number;
  samples: Sample[];
  negatives: { condition: string; frames: number; falsePositives: number }[];
  summary: { condition: string; distance: number; n: number; gtVisible: number; camRecall: number | null; lidarRecall: number; meanScore: number | null }[];
  byHeight: { condition: string; bucket: string; n: number; camRecall: number | null }[];
  inferMsMean: number;
}

function project(cam: THREE.PerspectiveCamera, p: THREE.Vector3, w: number, h: number) {
  const v = p.clone().project(cam);
  return { x: (v.x * 0.5 + 0.5) * w, y: (-v.y * 0.5 + 0.5) * h, inFront: v.z < 1 && v.z > -1 };
}

export async function runSweep(host: SweepHost, det: YoloxDetector, modelName: string, onProgress: (f: number, msg: string) => void): Promise<SweepResult> {
  const rc = new THREE.Raycaster();
  const { w, h } = host.cabSize;
  const samples: Sample[] = [];
  const negatives: SweepResult['negatives'] = [];
  const inferMs: number[] = [];
  const total = CONDITIONS.length * (DISTANCES.length * BEARINGS_DEG.length + 5);
  let done = 0;
  host.setCabPose(NaN); // reset to the machine heading
  const cam = host.cabCamera;
  for (const cond of CONDITIONS) {
    host.setCondition(cond);
    // negatives: no person, five headings
    host.hideWorker();
    let fp = 0;
    for (const dyaw of [-0.5, -0.25, 0, 0.25, 0.5]) {
      host.setCabPose(dyaw);
      const px = await host.renderCab();
      const d = await det.detect(px, w, h, [COCO_PERSON], SCORE_THR);
      inferMs.push(det.lastMs);
      fp += d.length;
      onProgress(++done / total, `${cond.label}: negatives`);
    }
    negatives.push({ condition: cond.id, frames: 5, falsePositives: fp });
    host.setCabPose(0);
    const fwd = new THREE.Vector3();
    cam.getWorldDirection(fwd);
    const camYaw = Math.atan2(fwd.z, fwd.x);
    for (const dist of DISTANCES) {
      for (const bdeg of BEARINGS_DEG) {
        const a = camYaw + (bdeg * Math.PI) / 180;
        const x = cam.position.x + Math.cos(a) * dist, z = cam.position.z + Math.sin(a) * dist;
        host.placeWorker(x, z, a + Math.PI);
        const cap = host.workerCapsule();
        // ground truth box from the capsule's bounding cylinder
        const pts: THREE.Vector3[] = [];
        for (const y of [cap.y0 - cap.r, cap.y1 + cap.r]) for (let k = 0; k < 8; k++) { const t = (k / 8) * Math.PI * 2; pts.push(new THREE.Vector3(cap.x + Math.cos(t) * cap.r, y, cap.z + Math.sin(t) * cap.r)); }
        const pp = pts.map((p) => project(cam, p, w, h));
        let gtBox: Sample['gtBox'] = null;
        if (pp.every((q) => q.inFront)) {
          const x0 = Math.max(0, Math.min(...pp.map((q) => q.x))), x1 = Math.min(w, Math.max(...pp.map((q) => q.x)));
          const y0 = Math.max(0, Math.min(...pp.map((q) => q.y))), y1 = Math.min(h, Math.max(...pp.map((q) => q.y)));
          if (x1 - x0 > 2 && y1 - y0 > 4) gtBox = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
        }
        // occlusion: fraction of body sample points with a clear line of sight over the terrain
        let vis = 0, nvis = 0;
        for (const fy of [0.15, 0.4, 0.65, 0.9]) for (const lat of [-0.6, 0, 0.6]) {
          const py = cap.y0 + (cap.y1 - cap.y0) * fy;
          const px = cap.x - Math.sin(a) * lat * cap.r, pz = cap.z + Math.cos(a) * lat * cap.r;
          const dv = new THREE.Vector3(px, py, pz).sub(cam.position);
          const L = dv.length(); dv.normalize();
          const hit = host.sensorRay.cast(cam.position.x, cam.position.y, cam.position.z, dv.x, dv.y, dv.z, L - 0.05, 0.2);
          rc.set(cam.position, dv); rc.far = L - 0.05; rc.near = 0.05;
          const meshHit = host.occluders.length ? rc.intersectObjects(host.occluders, true).length > 0 : false;
          nvis++; if (!hit && !meshHit) vis++;
        }
        const visibleFrac = vis / nvis;
        const gtVisible = !!gtBox && visibleFrac >= 0.3 && gtBox.h >= 8;
        const px = await host.renderCab();
        const dets: Detection[] = await det.detect(px, w, h, [COCO_PERSON], SCORE_THR);
        inferMs.push(det.lastMs);
        let camScore = 0, camIou = 0;
        if (gtBox) for (const d of dets) { const u = iou(d, gtBox); if (u > camIou) { camIou = u; camScore = d.score; } }
        const camDetected = gtVisible && camIou >= 0.3;
        // lidar visibility at the same instant (deterministic seed per sample)
        const lp = host.lidarPoses();
        const world = { ray: host.sensorRay, capsules: [cap], dustDensity: cond.dust, rain: cond.rain };
        const seed = Math.round(dist * 100 + bdeg + 1000 * CONDITIONS.indexOf(cond));
        const roof = scan(world, { x: lp.roof[0], y: lp.roof[1], z: lp.roof[2], yaw: lp.yaw }, OS1_64_LIKE, seed, 0, 1024);
        const front = scan(world, { x: lp.front[0], y: lp.front[1], z: lp.front[2], yaw: lp.yaw, pitch: (35 * Math.PI) / 180 }, OS1_64_LIKE, seed + 1, 0, 512);
        const cnt = (s: ReturnType<typeof scan>) => { let c = 0; for (let i = 0; i < s.count; i++) if (s.label[i] === PointLabel.Person) c++; return c; };
        const roofReturns = cnt(roof), frontReturns = cnt(front);
        samples.push({ condition: cond.id, distance: dist, bearing: bdeg, gtVisible, gtBox, visibleFrac, camDetected, camScore, camIou, roofReturns, frontReturns, lidarDetected: roofReturns + frontReturns >= MIN_RETURNS });
        onProgress(++done / total, `${cond.label}: ${dist} m`);
      }
    }
  }
  host.hideWorker();
  host.setCondition(CONDITIONS[0]);
  host.setCabPose(NaN);
  const summary: SweepResult['summary'] = [];
  for (const cond of CONDITIONS) for (const dist of DISTANCES) {
    const s = samples.filter((q) => q.condition === cond.id && q.distance === dist);
    const vis = s.filter((q) => q.gtVisible);
    const det = vis.filter((q) => q.camDetected);
    summary.push({
      condition: cond.id, distance: dist, n: s.length, gtVisible: vis.length,
      camRecall: vis.length ? det.length / vis.length : null,
      lidarRecall: s.filter((q) => q.lidarDetected).length / s.length,
      meanScore: det.length ? det.reduce((a, q) => a + q.camScore, 0) / det.length : null,
    });
  }
  const buckets: [string, number, number][] = [['<24 px', 0, 24], ['24-40 px', 24, 40], ['40-64 px', 40, 64], ['>=64 px', 64, 1e9]];
  const byHeight: SweepResult['byHeight'] = [];
  for (const cond of CONDITIONS) for (const [name, lo, hi] of buckets) {
    const v = samples.filter((q) => q.condition === cond.id && q.gtVisible && q.gtBox && q.gtBox.h >= lo && q.gtBox.h < hi);
    byHeight.push({ condition: cond.id, bucket: name, n: v.length, camRecall: v.length ? v.filter((q) => q.camDetected).length / v.length : null });
  }
  return { byHeight, createdAt: new Date().toISOString(), model: modelName, scoreThreshold: SCORE_THR, iouThreshold: 0.3, minLidarReturns: MIN_RETURNS, samples, negatives, summary, inferMsMean: inferMs.reduce((a, b) => a + b, 0) / Math.max(1, inferMs.length) };
}
