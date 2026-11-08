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
// --flag value options (removed from argv so positional soil / slope still work)
const opt = (name: string, n = 1): string[] | null => { const i = argv.indexOf(name); if (i < 0) return null; const v = argv.slice(i + 1, i + 1 + n); argv.splice(i, n + 1); return v; };
const flag = (name: string) => { const i = argv.indexOf(name); if (i < 0) return false; argv.splice(i, 1); return true; };
const nOpt = opt('--neural', 2);
const neuralModel = nOpt ? nOpt[0] : null;            // one .onnx, or comma-separated members (ensemble)
const neuralMu = nOpt ? parseFloat(nOpt[1] ?? '0.5') : 0;
const noFloor = flag('--no-floor');
const gateOpt = opt('--gate', 2);                      // maxStd maxOOR
const statsFile = opt('--train-stats')?.[0] ?? null;  // trainer's feature percentiles (json)
const traceFile = opt('--trace')?.[0] ?? null;        // write every dig as an MPM replay trace
const tag = opt('--tag')?.[0] ?? '';
const yawOff = parseFloat(opt('--yaw-offset')?.[0] ?? '0');
const startR = parseFloat(opt('--start')?.[0] ?? '4.2');
const lenT = parseFloat(opt('--length')?.[0] ?? '3.6');
const depthT = parseFloat(opt('--depth')?.[0] ?? '1.0');
const legacyPitch = flag('--legacy-pitch');
const legacyRelief = flag('--legacy-relief');
const noRelief = flag('--no-relief'); // ablation: model forces never slow the machine
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
const cx = machine.center.x, cz = machine.center.z, gy = machine.groundY, yaw = (machine.houseYawDeg + yawOff) * DEG;
const ux = Math.cos(yaw), uz = Math.sin(yaw);
const design = { p0: [cx + ux * startR, cz + uz * startR] as [number, number], p1: [cx + ux * (startR + lenT), cz + uz * (startR + lenT)] as [number, number], bottomWidth: 0.9, depth: depthT, sideSlope: slope };
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
const fstats = new (await import('../src/sim/neuralSoil')).FeatureStats();
const trainStats = statsFile ? JSON.parse(fs.readFileSync(statsFile, 'utf8')) : null;
const ranges: [number, number][] | undefined = trainStats ? trainStats.channels.map((c: any) => [c.p001, c.p999]) : undefined;
if (neuralModel) {
  const ort = await import('onnxruntime-node');
  const { loadNeuralModel, loadEnsemble } = await import('../src/sim/neuralSoil');
  const urls = neuralModel.split(',');
  const mdl = urls.length > 1 ? await loadEnsemble(ort, urls) : await loadNeuralModel(ort, urls[0]);
  const gate = gateOpt ? { maxStd: parseFloat(gateOpt[0]), maxOOR: parseFloat(gateOpt[1]), ranges } : undefined;
  auto.neural = { model: mdl, mu: neuralMu, fMax: noRelief ? Infinity : 112800, floor: !noFloor, gate, stats: fstats, legacyRelief: legacyRelief || noRelief };
}
if (legacyPitch) auto.attitude = 'legacy';
const traces: unknown[] = [];
if (traceFile) auto.traceRec = (r) => traces.push(r);
const volNow = () => { let v = 0; for (let k = 0; k < n; k++) if (M[k]) v += hf.B[k] + hf.L[k]; return v * hf.area; };
const vol0 = volNow();
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
const vol1 = volNow();
const g = hf.gradeStats(0.03);
const res = {
  soil: soilId, design, simSeconds: +t.toFixed(1), wallSeconds: (Date.now() - t0) / 1000, phase: auto.phase, verified: auto.verified,
  cycles: auto.stats.cycles, avgCycleS: +auto.stats.avgCycleS.toFixed(2), avgFill: +auto.stats.avgFill.toFixed(3), productivityM3h: +auto.stats.productivity.toFixed(1),
  bankRemoved: +auto.stats.bankRemoved.toFixed(3), passes: auto.stats.passes,
  surveys: auto.stats.surveys.map((s) => ({ t: +s.t.toFixed(1), points: s.points, coverage: +s.coverage.toFixed(3), inTolFrac: +s.inTolFrac.toFixed(3), lidarMeanAboveCm: +(s.lidarMeanAbove * 100).toFixed(2), truthMeanAboveCm: +(s.truthMeanAbove * 100).toFixed(2) })),
  final: { onGradePct: +g.onGradePct.toFixed(1), meanAbsErrorCm: +(g.meanAbsError * 100).toFixed(2), remainingCutM3: +g.remainingCut.toFixed(3), overDigM3: +g.overDig.toFixed(3), unplannedCutM3: +g.unplannedCut.toFixed(3), sloughedM3: +hf.ledger.sloughed.toFixed(3), maxAboveCm: +(g.maxAbove * 100).toFixed(1), maxBelowCm: +(g.maxBelow * 100).toFixed(1) },
  log: auto.log,
  volume: { terrainStartM3: +vol0.toFixed(3), terrainEndM3: +vol1.toFixed(3), inBucketM3: +auto.bucketLoad.toFixed(3), carriedLedgerM3: +hf.ledger.carried.toFixed(3),
    deltaM3: +(vol1 + hf.ledger.carried - vol0).toFixed(3), bankCutM3: +hf.ledger.bankCut.toFixed(3), swellExpectedM3: +(hf.ledger.bankCut * (hf.soil.swell - 1)).toFixed(3) },
  neural: neuralModel ? { model: neuralModel.split(',').map((u) => path.basename(u)), mu: neuralMu, peakDragKN: +(auto.peakDragN / 1e3).toFixed(1), steps: auto.neuralSteps,
    clippedCells: auto.neuralClipped, flooredCells: auto.neuralFloored, spilledM3: +auto.neuralSpilled.toFixed(3), netRemovedM3: +auto.neuralNetRemoved.toFixed(3), modelDvM3: +auto.neuralModelDvM3.toFixed(3), floorAddedM3: +auto.neuralFloorAddedM3.toFixed(3), floorProjection: !noFloor,
    gate: gateOpt ? { maxStd: +gateOpt[0], maxOOR: +gateOpt[1] } : null, gateSteps: auto.gateSteps, fallbackSteps: auto.fallbackSteps,
    networkShare: auto.gateSteps + auto.fallbackSteps ? +(auto.gateSteps / (auto.gateSteps + auto.fallbackSteps)).toFixed(3) : null,
    attitude: auto.attitude, relief: noRelief ? 'off (ablation)' : legacyRelief ? 'legacy fMax on Fx' : 'dx225 per-actuator', peakDemand: auto.peakDemand,
    maxPoseGapS: auto.maxPoseGapS, inputRanges: fstats.summary(ranges), oorSteps: fstats.oorSteps, meanEnsembleStd: fstats.steps ? +(fstats.stdSum / Math.max(1, auto.gateSteps + auto.fallbackSteps)).toFixed(5) : null } : null,
  traces: traceFile ? traces.length : undefined,
};
void forwardPlanar; void planarToWorld;
fs.mkdirSync('bench/results', { recursive: true });
const outName = `bench/results/autonomy-${soilId}-slope${slope}${neuralModel ? '-neural' + (noFloor ? '-nofloor' : '') : ''}${tag ? '-' + tag : ''}.json`;
fs.writeFileSync(outName, JSON.stringify(res, null, 2));
if (traceFile) fs.writeFileSync(traceFile, JSON.stringify({ soil: soilId, slope, design, yawOffset: yawOff, traces }));
console.log('wrote', outName, traceFile ? `traces ${traces.length}` : '');
console.log(JSON.stringify({ ...res, log: res.log.slice(-4) }, null, 1));
