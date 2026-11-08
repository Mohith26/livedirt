/**
 * Headless benchmark: the real captured terrain + the app's earthwork sim, dig-cycle controller and lidar model,
 * stepped at a fixed dt. Writes bench/results/autonomy-<soil>.json. Usage: npx tsx bench/autonomy.ts [soil] [slope]
 */
import fs from 'node:fs';
import path from 'node:path';
import { Heightfield } from '../src/sim/heightfield';
import { HeightRaycaster } from '../src/sim/heightRay';
import { soilById } from '../src/sim/soil';
import { DigAutonomy } from '../src/machine/autonomy';
import { DX225_GEOMETRY, solveIk, forwardPlanar, planarToWorld } from '../src/machine/kinematics';
import { scan, OS1_64_LIKE } from '../src/sensors/lidarModel';

const argv = process.argv.slice(2);
const ni = argv.indexOf('--neural');
const neuralModel = ni >= 0 ? argv[ni + 1] : null;
const neuralMu = ni >= 0 ? parseFloat(argv[ni + 2] ?? '0.5') : 0;
if (ni >= 0) argv.splice(ni, 3);
const soilId = argv[0] ?? 'wet-clay';
const slope = parseFloat(argv[1] ?? '1');
const scene = path.resolve('public/scene');
const meta = JSON.parse(fs.readFileSync(path.join(scene, 'terrain.json'), 'utf8'));
const machine = JSON.parse(fs.readFileSync(path.join(scene, 'machine.json'), 'utf8'));
const bin = fs.readFileSync(path.join(scene, 'terrain.bin'));
const N = meta.N, n = N * N;
const ab = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const H0 = new Float32Array(ab, 0, n), M = new Uint8Array(ab, n * 8, n);
const hf = new Heightfield({ N, cell: meta.cell, half: meta.half, H0, mask: M, zoneRadius: 11.5 });
hf.soil = soilById(soilId);
const DEG = Math.PI / 180;
const cx = machine.center.x, cz = machine.center.z, gy = machine.groundY, yaw = machine.houseYawDeg * DEG;
const ux = Math.cos(yaw), uz = Math.sin(yaw);
const design = { p0: [cx + ux * 4.2, cz + uz * 4.2] as [number, number], p1: [cx + ux * 7.8, cz + uz * 7.8] as [number, number], bottomWidth: 0.9, depth: 1.0, sideSlope: slope };
hf.setDesign(design);
const tip: [number, number, number] = [cx + ux * 6.2, hf.heightAt(cx + ux * 6.2, cz + uz * 6.2) + 0.25, cz + uz * 6.2];
const rest = solveIk(DX225_GEOMETRY, cx, gy, cz, tip, -100 * DEG, yaw).joints;
const auto = new DigAutonomy(hf, DX225_GEOMETRY, cx, cz, gy, rest);
const ray = new HeightRaycaster(N, meta.cell, meta.half, new Float32Array(n));
const refreshRay = () => { for (let k = 0; k < n; k++) ray.surf[k] = hf.B[k] + hf.L[k]; ray.rebuild(); };
refreshRay();
// sensor mounts, same offsets as the mesh twin (cab front-left); computed from swing
const mounts = (sw: number) => {
  const fwd = [Math.cos(sw), Math.sin(sw)], lat = [-Math.sin(sw), Math.cos(sw)];
  const at = (f: number, l: number, h: number) => [cx + fwd[0] * f + lat[0] * l, gy + h, cz + fwd[1] * f + lat[1] * l];
  return { roof: at(0.35, -0.85, 3.16), front: at(1.55, -0.57, 3.08) };
};
let seed = 1, latest: Float32Array | null = null;
auto.requestScan = () => latest;
if (neuralModel) {
  const ort = await import('onnxruntime-node');
  const { loadNeuralModel } = await import('../src/sim/neuralSoil');
  auto.neural = { model: await loadNeuralModel(ort, neuralModel), mu: neuralMu, fMax: 112800 };
}
auto.plan(design);
auto.start();
const dt = 0.05;
let t = 0, scanAcc = 0, rayAcc = 0;
const t0 = Date.now();
while (auto.phase !== 'done' && auto.phase !== 'unreachable' && t < 3600) {
  auto.step(dt, false);
  if (neuralModel) await auto.settleNeural();
  hf.relax(3);
  t += dt; scanAcc += dt; rayAcc += dt;
  if (rayAcc > 0.5) { refreshRay(); rayAcc = 0; }
  if (scanAcc >= 0.1) {
    scanAcc = 0;
    const m = mounts(auto.joints.swing);
    const a = scan({ ray, capsules: [], dustDensity: 0, rain: 0 }, { x: m.front[0], y: m.front[1], z: m.front[2], yaw: auto.joints.swing, pitch: 35 * DEG }, OS1_64_LIKE, seed++, t, 512);
    const b = scan({ ray, capsules: [], dustDensity: 0, rain: 0 }, { x: m.roof[0], y: m.roof[1], z: m.roof[2], yaw: auto.joints.swing }, OS1_64_LIKE, seed++, t, 1024);
    const out = new Float32Array((a.count + b.count) * 4); out.set(a.points.subarray(0, a.count * 4)); out.set(b.points.subarray(0, b.count * 4), a.count * 4);
    latest = out;
  }
}
hf.settle(400);
const g = hf.gradeStats(0.03);
const res = {
  soil: soilId, design, simSeconds: +t.toFixed(1), wallSeconds: (Date.now() - t0) / 1000, phase: auto.phase, verified: auto.verified,
  cycles: auto.stats.cycles, avgCycleS: +auto.stats.avgCycleS.toFixed(2), avgFill: +auto.stats.avgFill.toFixed(3), productivityM3h: +auto.stats.productivity.toFixed(1),
  bankRemoved: +auto.stats.bankRemoved.toFixed(3), passes: auto.stats.passes,
  surveys: auto.stats.surveys.map((s) => ({ t: +s.t.toFixed(1), points: s.points, coverage: +s.coverage.toFixed(3), inTolFrac: +s.inTolFrac.toFixed(3), lidarMeanAboveCm: +(s.lidarMeanAbove * 100).toFixed(2), truthMeanAboveCm: +(s.truthMeanAbove * 100).toFixed(2) })),
  final: { onGradePct: +g.onGradePct.toFixed(1), meanAbsErrorCm: +(g.meanAbsError * 100).toFixed(2), remainingCutM3: +g.remainingCut.toFixed(3), overDigM3: +g.overDig.toFixed(3), unplannedCutM3: +g.unplannedCut.toFixed(3), sloughedM3: +hf.ledger.sloughed.toFixed(3), maxAboveCm: +(g.maxAbove * 100).toFixed(1), maxBelowCm: +(g.maxBelow * 100).toFixed(1) },
  log: auto.log,
  neural: neuralModel ? { model: path.basename(neuralModel), mu: neuralMu, peakDragKN: +(auto.peakDragN / 1e3).toFixed(1), steps: auto.neuralSteps, clippedCells: auto.neuralClipped } : null,
};
void forwardPlanar; void planarToWorld;
fs.mkdirSync('bench/results', { recursive: true });
fs.writeFileSync(`bench/results/autonomy-${soilId}-slope${slope}${neuralModel ? '-neural' : ''}.json`, JSON.stringify(res, null, 2));
console.log(JSON.stringify({ ...res, log: res.log.slice(-4) }, null, 1));
