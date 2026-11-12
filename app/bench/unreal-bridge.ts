/**
 * LiveDirt -> Unreal bridge. Runs the real simulation headless (captured terrain, earthwork heightfield, adaptive dig
 * controller, per-scoop learned soil v3 (model a16, 5-member ensemble) or the v1 heightfield cut, lidar survey) and serves its state over HTTP on
 * localhost so an Unreal Editor Python script (unreal/LiveDirtUE/Content/Python/livedirt_ue.py) can render it.
 *
 *   cd app && npx tsx bench/unreal-bridge.ts [--port 8790]
 *   open http://127.0.0.1:8790/   (control page: start / pause / reset, soil model, speed, colour mode)
 *
 * Coordinates: everything sent is already in Unreal space: centimetres, Z up, X = app x, Y = app z, Z = app y
 * (swapping two axes converts three.js right-handed Y-up to Unreal left-handed Z-up without mirroring).
 */
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { Heightfield } from '../src/sim/heightfield';
import { HeightRaycaster } from '../src/sim/heightRay';
import { soilById, SOILS } from '../src/sim/soil';
import { DigAutonomy } from '../src/machine/autonomy';
import { DX225_GEOMETRY, solveIk, forwardPlanar, planarToWorld, type JointState } from '../src/machine/kinematics';
import { scan, OS1_64_LIKE } from '../src/sensors/lidarModel';
import { ScoopSoil, presetParams } from '../src/sim/scoopSoil';

const argv = process.argv.slice(2);
const port = parseInt(argv[argv.indexOf('--port') + 1] || '8790', 10) || 8790;
const DEG = Math.PI / 180;
const scene = path.resolve('public/scene');
const meta = JSON.parse(fs.readFileSync(path.join(scene, 'terrain.json'), 'utf8'));
const machine = JSON.parse(fs.readFileSync(path.join(scene, 'machine.json'), 'utf8'));
const bin = fs.readFileSync(path.join(scene, 'terrain.bin'));
const N: number = meta.N, n = N * N;
const ab = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const H0 = new Float32Array(ab, 0, n), M = new Uint8Array(ab, n * 8, n), ALB = new Uint8Array(ab, n * 9, n * 3);
const cx = machine.center.x, cz = machine.center.z, gy = machine.groundY;
const houseYaw0 = machine.houseYawDeg * DEG, trackYaw = machine.trackYawDeg * DEG;

// ---------------------------------------------------------------- simulation state
const ortN = await import('onnxruntime-node');
// learned soil v3 = model a16: 5-member ensemble (members averaged), same files the browser v3 loads
const SCOOP_MODEL = 'a16';
const sess = await Promise.all([0, 1, 2, 3, 4].map((i) => ortN.InferenceSession.create(path.resolve(`public/models/scoop-${SCOOP_MODEL}-s${i}.onnx`))));
const scoop = ScoopSoil.fromParts(ortN as never, sess as never, JSON.parse(fs.readFileSync(path.resolve('public/models/scoop-v4-bucket.json'), 'utf8')));
console.log(`learned soil v3: model ${SCOOP_MODEL}, ${scoop.members}-member ensemble`);

const cfg = { soil: 'sandy-gravel', model: 'v3' as 'v3' | 'v1', controller: 'adaptive' as 'adaptive' | 'legacy', speed: 4, color: 'albedo' as 'albedo' | 'grade' | 'change', running: false };
let hf: Heightfield, auto: DigAutonomy, ray: HeightRaycaster, restJoints: JointState;
let terrainVersion = 0, simT = 0, scanSeed = 1, latestScan: Float32Array | null = null, busy = false, lastScoop: { carried: number; ms: number } | null = null;
const log: string[] = [];
const design = (() => {
  const yaw = houseYaw0 + 50 * DEG, ux = Math.cos(yaw), uz = Math.sin(yaw);
  return { p0: [cx + ux * 4.0, cz + uz * 4.0] as [number, number], p1: [cx + ux * 8.0, cz + uz * 8.0] as [number, number], bottomWidth: 1.1, depth: 1.0, sideSlope: 1.5 };
})();

function reset() {
  hf = new Heightfield({ N, cell: meta.cell, half: meta.half, H0, mask: M, zoneRadius: 11.5 });
  hf.soil = soilById(cfg.soil);
  hf.setDesign(design);
  const ux = Math.cos(houseYaw0), uz = Math.sin(houseYaw0);
  const tip: [number, number, number] = [cx + ux * 6.2, hf.heightAt(cx + ux * 6.2, cz + uz * 6.2) + 0.25, cz + uz * 6.2];
  restJoints = solveIk(DX225_GEOMETRY, cx, gy, cz, tip, -100 * DEG, houseYaw0).joints;
  auto = new DigAutonomy(hf, DX225_GEOMETRY, cx, cz, gy, restJoints);
  auto.controller = cfg.controller;
  auto.externalSoil = cfg.model === 'v3';
  auto.traceMargin = 1.0;
  auto.designClamp = cfg.model === 'v1' && cfg.controller === 'legacy';
  auto.requestScan = () => latestScan;
  ray = new HeightRaycaster(N, meta.cell, meta.half, new Float32Array(n));
  refreshRay();
  simT = 0; latestScan = null; lastScoop = null; cfg.running = false; terrainVersion++;
  log.length = 0;
}
function refreshRay() { for (let k = 0; k < n; k++) ray.surf[k] = hf.B[k] + hf.L[k]; ray.rebuild(); }

async function stepSim(dtWall: number) {
  if (!cfg.running || busy) return;
  busy = true;
  try {
    const steps = Math.max(1, Math.round((dtWall * cfg.speed) / 0.05));
    for (let s = 0; s < steps; s++) {
      if (auto.phase === 'done' || auto.phase === 'unreachable') { cfg.running = false; break; }
      auto.step(0.05, false);
      simT += 0.05;
      if (auto.pendingBite) {
        const tr = auto.pendingBite; auto.pendingBite = null;
        const t0 = performance.now();
        const r = await scoop.bite(tr, presetParams(hf.soil));
        auto.applyExternal(tr, r);
        hf.clearActive();
        lastScoop = { carried: r.carried, ms: performance.now() - t0 };
        terrainVersion++;
        break; // one scoop per tick so the renderer sees each one
      }
      if (hf.isSettling()) { hf.relax(3); terrainVersion++; }
      // lidar only while surveying (the controller judges grade from its own scans)
      if (auto.phase === 'survey' && Math.round(simT / 0.05) % 2 === 0) {
        refreshRay();
        const sw = auto.joints.swing, fwd = [Math.cos(sw), Math.sin(sw)], lat = [-Math.sin(sw), Math.cos(sw)];
        const at = (f: number, l: number, h: number) => [cx + fwd[0] * f + lat[0] * l, gy + h, cz + fwd[1] * f + lat[1] * l];
        const fr = at(1.55, -0.57, 3.08), rf = at(0.35, -0.85, 3.16);
        const a = scan({ ray, capsules: [], dustDensity: 0, rain: 0 }, { x: fr[0], y: fr[1], z: fr[2], yaw: sw, pitch: 35 * DEG }, OS1_64_LIKE, scanSeed++, simT, 512);
        const b = scan({ ray, capsules: [], dustDensity: 0, rain: 0 }, { x: rf[0], y: rf[1], z: rf[2], yaw: sw }, OS1_64_LIKE, scanSeed++, simT, 1024);
        const out = new Float32Array((a.count + b.count) * 4); out.set(a.points.subarray(0, a.count * 4)); out.set(b.points.subarray(0, b.count * 4), a.count * 4);
        latestScan = out;
      }
    }
  } finally { busy = false; }
}

// ---------------------------------------------------------------- geometry for Unreal
const ue = (x: number, y: number, z: number): [number, number, number] => [x * 100, z * 100, y * 100];
interface Part { name: string; c: [number, number, number]; s: [number, number, number]; r: [number, number, number]; color: string }
/** box between two world points (app coords), thickness w x h (m); r = [roll, pitch, yaw] degrees in Unreal */
function segment(name: string, a: number[], b: number[], w: number, h: number, color: string, roll = 0): Part {
  const A = ue(a[0], a[1], a[2]), B = ue(b[0], b[1], b[2]);
  const d = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], len = Math.hypot(d[0], d[1], d[2]);
  const yaw = Math.atan2(d[1], d[0]) / DEG, pitch = Math.atan2(d[2], Math.hypot(d[0], d[1])) / DEG;
  return { name, c: [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2], s: [len, w * 100, h * 100], r: [roll, pitch, yaw], color };
}
/** yaw in app (x,z) plane -> Unreal yaw (X,Y plane, same handedness after the swap) */
const ueYaw = (appYaw: number) => appYaw / DEG;
function parts(): Part[] {
  const j = auto.joints, g = DX225_GEOMETRY, fk = forwardPlanar(g, j);
  const W = (r: number, y: number) => planarToWorld(cx, gy, cz, j.swing, r, y);
  const P: Part[] = [];
  const tc = [Math.cos(trackYaw), Math.sin(trackYaw)], tl = [-Math.sin(trackYaw), Math.cos(trackYaw)];
  for (const s of [-1, 1]) {
    const p = [cx + tl[0] * s * 1.195, gy + 0.47, cz + tl[1] * s * 1.195];
    P.push({ name: `track${s}`, c: ue(p[0], p[1], p[2]), s: [445, 60, 94], r: [0, 0, ueYaw(trackYaw)], color: 'dark' });
  }
  const hc = [cx - Math.cos(j.swing) * 0.4, gy + 1.45, cz - Math.sin(j.swing) * 0.4];
  P.push({ name: 'house', c: ue(hc[0], hc[1], hc[2]), s: [430, 280, 110], r: [0, 0, ueYaw(j.swing)], color: 'paint' });
  const cab = [cx + Math.cos(j.swing) * 0.55 + Math.sin(j.swing) * 0.85, gy + 2.45, cz + Math.sin(j.swing) * 0.55 - Math.cos(j.swing) * 0.85];
  P.push({ name: 'cab', c: ue(cab[0], cab[1], cab[2]), s: [180, 100, 160], r: [0, 0, ueYaw(j.swing)], color: 'glass' });
  const cw = [cx - Math.cos(j.swing) * 2.0, gy + 1.55, cz - Math.sin(j.swing) * 2.0];
  P.push({ name: 'counterweight', c: ue(cw[0], cw[1], cw[2]), s: [70, 270, 105], r: [0, 0, ueYaw(j.swing)], color: 'paint' });
  P.push(segment('boom', W(fk.foot[0], fk.foot[1]), W(fk.elbow[0], fk.elbow[1]), 0.45, 0.7, 'paint'));
  P.push(segment('arm', W(fk.elbow[0], fk.elbow[1]), W(fk.wrist[0], fk.wrist[1]), 0.38, 0.5, 'paint'));
  // bucket: wrist -> cutting edge, tilted (tiltrotator roll) when the controller is on a slope lane
  const tilt = (auto as unknown as { curTilt?: number }).curTilt ?? 0;
  P.push(segment('bucket', W(fk.wrist[0], fk.wrist[1]), W(fk.tip[0], fk.tip[1]), 1.05, 0.9, 'steel', tilt / DEG));
  // soil carried in the bucket (heaped), scaled by the load the soil model reported; hidden when empty
  const load = auto.bucketLoad, sz = load > 0.01 ? Math.cbrt(load / 1.06) : 0.001;
  const mid = W((fk.wrist[0] + fk.tip[0]) / 2, (fk.wrist[1] + fk.tip[1]) / 2 + 0.35 * sz);
  P.push({ name: 'payload', c: ue(mid[0], mid[1], mid[2]), s: [110 * sz, 95 * sz, 70 * sz], r: [0, 0, ueYaw(j.swing)], color: 'soil' });
  return P;
}

/** dig zone: design bounding box + spoil point, padded 2.5 m (all the cells the dig cycle can change) */
const zone = (() => {
  hf = new Heightfield({ N, cell: meta.cell, half: meta.half, H0, mask: M, zoneRadius: 11.5 });
  hf.setDesign(design);
  let i0 = N, i1 = -1, j0 = N, j1 = -1;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) if (!Number.isNaN(hf.design[j * N + i])) { i0 = Math.min(i0, i); i1 = Math.max(i1, i); j0 = Math.min(j0, j); j1 = Math.max(j1, j); }
  const tmp = new DigAutonomy(hf, DX225_GEOMETRY, cx, cz, gy, { swing: 0, boom: 0, arm: -1, bucket: 0 });
  tmp.plan(design);
  const [si, sj] = hf.cellOf(tmp.spoil[0], tmp.spoil[1]);
  i0 = Math.min(i0, si); i1 = Math.max(i1, si); j0 = Math.min(j0, sj); j1 = Math.max(j1, sj);
  const pad = Math.round(2.5 / meta.cell);
  return { i0: Math.max(0, i0 - pad), j0: Math.max(0, j0 - pad), i1: Math.min(N - 1, i1 + pad), j1: Math.min(N - 1, j1 + pad) };
})();

function cellColor(k: number): [number, number, number] {
  const a = [ALB[3 * k], ALB[3 * k + 1], ALB[3 * k + 2]];
  const h = hf.B[k] + hf.L[k], cut = H0[k] - hf.B[k];
  if (cfg.color === 'grade') {
    const D = hf.design[k];
    if (Number.isNaN(D)) return [a[0] * 0.55, a[1] * 0.55, a[2] * 0.55] as [number, number, number];
    const e = h - D;
    if (Math.abs(e) <= 0.03) return [60, 200, 110];
    if (e > 0) { const t = Math.min(1, e / 0.5); return [40 + 20 * (1 - t), 90 + 60 * (1 - t), 255]; }
    const t = Math.min(1, -e / 0.3); return [255, 120 - 80 * t, 80 - 60 * t];
  }
  if (cfg.color === 'change') {
    const d = h - H0[k];
    if (Math.abs(d) < 0.02) return [a[0] * 0.6, a[1] * 0.6, a[2] * 0.6] as [number, number, number];
    const t = Math.min(1, Math.abs(d) / 0.8);
    return d < 0 ? [40, 90 + 80 * t, 200 + 55 * t] : [220 + 35 * t, 150 - 60 * t, 60];
  }
  // albedo, with exposed subsoil and loose spoil tinted like the app does
  const s = hf.soil;
  if (hf.L[k] > 0.02) { const tl = s.spoilTint, m = 0.82; return [a[0] * tl[0] * m, a[1] * tl[1] * m, a[2] * tl[2] * m].map((v) => Math.min(255, v)) as [number, number, number]; }
  if (cut > 0.03) { const ts = s.subsoilTint, dk = Math.max(0.6, 0.85 - 0.15 * Math.min(1, cut)); return [a[0] * ts[0] * dk, a[1] * ts[1] * dk, a[2] * ts[2] * dk].map((v) => Math.min(255, v)) as [number, number, number]; }
  return a as [number, number, number];
}

function gridPayload(i0: number, j0: number, i1: number, j1: number, stride: number) {
  const ni = Math.floor((i1 - i0) / stride) + 1, nj = Math.floor((j1 - j0) / stride) + 1;
  const hb = new Float32Array(ni * nj), cb = new Uint8Array(ni * nj * 3);
  let q = 0;
  for (let jj = 0; jj < nj; jj++) for (let ii = 0; ii < ni; ii++) {
    const i = i0 + ii * stride, j = j0 + jj * stride, k = j * N + i;
    hb[q] = (hf.B[k] + hf.L[k]) * 100;
    const c = cellColor(k);
    cb[3 * q] = c[0]; cb[3 * q + 1] = c[1]; cb[3 * q + 2] = c[2];
    q++;
  }
  const [ox, oz] = hf.center(i0, j0);
  // cell (i, j) centre in app coords = (ox + ii*cell*stride, oz + jj*cell*stride) -> Unreal X, Y
  return { ni, nj, stepCm: meta.cell * stride * 100, originCm: [ox * 100, oz * 100], heights: Buffer.from(hb.buffer).toString('base64'), colors: Buffer.from(cb.buffer).toString('base64') };
}

function statePayload() {
  const g = hf.gradeStats(0.03), gB = hf.gradeStats(0.03, 'bottom');
  return {
    t: +simT.toFixed(2), running: cfg.running, phase: auto.phase, cfg, terrainVersion, parts: parts(),
    bucketLoadM3: +auto.bucketLoad.toFixed(3), lastScoop,
    stats: { cycles: auto.stats.cycles, passes: auto.stats.passes, bankRemovedM3: +auto.stats.bankRemoved.toFixed(2), onGradePct: +g.onGradePct.toFixed(1), floorOnGradePct: +gB.onGradePct.toFixed(1), floorErrCm: +(gB.meanAbsError * 100).toFixed(1), remainingCutM3: +g.remainingCut.toFixed(2) },
    log: auto.log.slice(-6),
  };
}

const page = `<!doctype html><meta charset=utf-8><title>LiveDirt UE control</title>
<style>body{font:14px system-ui;background:#16130f;color:#eee;margin:24px;max-width:640px}button{margin:3px;padding:8px 12px;border-radius:8px;border:0;background:#333;color:#eee;cursor:pointer}button.on{background:#e8701e}pre{background:#000;padding:10px;border-radius:8px;white-space:pre-wrap}</style>
<h2>LiveDirt → Unreal</h2>
<div><button onclick="c('start')">▶ Run</button><button onclick="c('pause')">⏸ Pause</button><button onclick="c('reset')">↺ Reset site</button></div>
<div>Soil model: <button id=m_v3 onclick="c('model=v3')">Learned per-scoop (v3, a16 ensemble)</button><button id=m_v1 onclick="c('model=v1')">Simple heightfield (v1)</button></div>
<div>Controller: <button id=k_adaptive onclick="c('controller=adaptive')">Adaptive</button><button id=k_legacy onclick="c('controller=legacy')">Legacy</button></div>
<div>Soil: ${SOILS.map((s) => `<button id=s_${s.id} onclick="c('soil=${s.id}')">${s.label}</button>`).join('')}</div>
<div>Speed: ${[1, 2, 4, 8, 16].map((v) => `<button id=v_${v} onclick="c('speed=${v}')">${v}×</button>`).join('')}</div>
<div>Colour: <button id=c_albedo onclick="c('color=albedo')">Scan colours</button><button id=c_grade onclick="c('color=grade')">Grade error</button><button id=c_change onclick="c('color=change')">Cut / fill</button></div>
<p style="color:#aaa">Model / controller / soil changes reset the site.</p><pre id=st></pre>
<script>async function c(a){await fetch('/cmd?'+a,{method:'POST'});}
async function poll(){try{const s=await (await fetch('/state')).json();const f=s.cfg;
for(const b of document.querySelectorAll('button[id]'))b.classList.remove('on');
for(const id of ['m_'+f.model,'k_'+f.controller,'s_'+f.soil,'v_'+f.speed,'c_'+f.color]){const e=document.getElementById(id);if(e)e.classList.add('on');}
document.getElementById('st').textContent=JSON.stringify({t:s.t,running:s.running,phase:s.phase,bucket:s.bucketLoadM3,lastScoop:s.lastScoop,...s.stats},null,1)+'\\n\\n'+s.log.join('\\n');}catch(e){}setTimeout(poll,500);}poll();</script>`;

reset();
let lastWall = performance.now();
setInterval(() => { const now = performance.now(); const dt = Math.min(0.25, (now - lastWall) / 1000); lastWall = now; void stepSim(dt); }, 33);

http.createServer((req, res) => {
  const u = new URL(req.url ?? '/', 'http://x');
  const json = (o: unknown) => { res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }); res.end(JSON.stringify(o)); };
  if (u.pathname === '/') { res.writeHead(200, { 'Content-Type': 'text/html' }); res.end(page); return; }
  if (u.pathname === '/state') return json(statePayload());
  if (u.pathname === '/scene') return json({ zone, terrain: gridPayload(0, 0, N - 1, N - 1, 2), design, machine: { center: ue(cx, gy, cz) }, unrealAxes: 'X=app x, Y=app z, Z=app y; cm' });
  if (u.pathname === '/terrain') return json({ version: terrainVersion, ...gridPayload(zone.i0, zone.j0, zone.i1, zone.j1, 1) });
  if (u.pathname === '/cmd') {
    for (const [k, v] of u.searchParams) {
      if (k === 'start') { if (auto.phase === 'idle' || auto.phase === 'done') { auto.plan(design); auto.start(); } cfg.running = true; }
      else if (k === 'pause') cfg.running = false;
      else if (k === 'reset') reset();
      else if (k === 'speed') cfg.speed = Math.max(1, Math.min(32, parseFloat(v) || 4));
      else if (k === 'color') { cfg.color = v as typeof cfg.color; terrainVersion++; }
      else if (k === 'model' && (v === 'v1' || v === 'v3')) { cfg.model = v; reset(); }
      else if (k === 'controller' && (v === 'adaptive' || v === 'legacy')) { cfg.controller = v; reset(); }
      else if (k === 'soil' && SOILS.some((s) => s.id === v)) { cfg.soil = v; reset(); }
    }
    return json({ ok: true, cfg });
  }
  res.writeHead(404); res.end();
}).listen(port, '127.0.0.1', () => console.log(`LiveDirt bridge on http://127.0.0.1:${port}/  (zone ${zone.i1 - zone.i0 + 1} x ${zone.j1 - zone.j0 + 1} cells)`));
