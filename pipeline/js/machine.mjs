// Estimate the photographed excavator's pose from the cleaned metric splat cloud + terrain products.
import fs from 'node:fs'; import { readPly } from './ply.mjs';
const [,, plyPath, sceneDir] = process.argv;
const meta = JSON.parse(fs.readFileSync(sceneDir + '/terrain.json')); const bin = fs.readFileSync(sceneDir + '/terrain.bin');
const { N, cell, half } = meta; const H0 = new Float32Array(bin.buffer, bin.byteOffset, N * N); const M = bin.subarray(N * N * 8, N * N * 9);
const at = (A, x, z) => A[Math.min(N - 1, Math.max(0, Math.floor((z + half) / cell))) * N + Math.min(N - 1, Math.max(0, Math.floor((x + half) / cell)))];
// connected component of mask nearest the origin
const seen = new Uint8Array(N * N); let best = null;
for (let s = 0; s < N * N; s++) { if (!M[s] || seen[s]) continue; const comp = [s]; seen[s] = 1; for (let q = 0; q < comp.length; q++) { const k = comp[q], x = k % N, z = (k / N) | 0; for (const [dx, dz] of [[1,0],[-1,0],[0,1],[0,-1]]) { const xx = x + dx, zz = z + dz; if (xx < 0 || zz < 0 || xx >= N || zz >= N) continue; const kk = zz * N + xx; if (M[kk] && !seen[kk]) { seen[kk] = 1; comp.push(kk); } } }
  let cx = 0, cz = 0; for (const k of comp) { cx += (k % N + .5) * cell - half; cz += (((k / N) | 0) + .5) * cell - half; } cx /= comp.length; cz /= comp.length; const d = Math.hypot(cx, cz);
  if (comp.length > 300 && (!best || d < best.d)) best = { d, comp, cx, cz }; }
const inComp = new Uint8Array(N * N); for (const k of best.comp) inComp[k] = 1;
const p = readPly(plyPath); const { count, stride, data, idx } = p; const sig = v => 1 / (1 + Math.exp(-v)); const C0 = 0.28209479177387814;
const track = [], boom = [];
for (let i = 0; i < count; i++) { const b = i * stride; if (sig(data[b + idx.opacity]) < 0.5) continue; const x = data[b + idx.x], y = data[b + idx.y], z = data[b + idx.z];
  const cx = Math.floor((x + half) / cell), cz = Math.floor((z + half) / cell); if (cx < 0 || cz < 0 || cx >= N || cz >= N || !inComp[cz * N + cx]) continue;
  const rel = y - at(H0, x, z); const R = 0.5 + C0 * data[b + idx.f_dc_0], B = 0.5 + C0 * data[b + idx.f_dc_2];
  if (rel > 0.05 && rel < 0.8) track.push([x, z, y]); if (rel > 1.6 && R - B > 0.25) boom.push([x, z, y]); }
// min-area rectangle of track band
let rect = null; for (let a = 0; a < 180; a += 0.5) { const t = a * Math.PI / 180, u = [Math.cos(t), Math.sin(t)];
  const P = track.map(q => q[0] * u[0] + q[1] * u[1]).sort((a, b) => a - b), Q = track.map(q => -q[0] * u[1] + q[1] * u[0]).sort((a, b) => a - b);
  const lo = (A) => A[Math.floor(A.length * 0.02)], hi = (A) => A[Math.floor(A.length * 0.98)]; const L = hi(P) - lo(P), W = hi(Q) - lo(Q);
  if (!rect || L * W < rect.area) rect = { a, L, W, area: L * W, cp: (hi(P) + lo(P)) / 2, cq: (hi(Q) + lo(Q)) / 2, u }; }
const t = rect.a * Math.PI / 180; const center = [rect.cp * Math.cos(t) - rect.cq * Math.sin(t), rect.cp * Math.sin(t) + rect.cq * Math.cos(t)];
// ensure L is the long axis
let trackYaw = rect.L >= rect.W ? t : t + Math.PI / 2; const trackLen = Math.max(rect.L, rect.W), trackWid = Math.min(rect.L, rect.W);
// boom direction: far-most boom splats from center
const far = boom.map(q => ({ q, d: Math.hypot(q[0] - center[0], q[1] - center[1]) })).sort((a, b) => b.d - a.d).slice(0, Math.max(20, Math.floor(boom.length * 0.05)));
let bx = 0, bz = 0; for (const f of far) { bx += f.q[0]; bz += f.q[1]; } bx /= far.length; bz /= far.length;
const houseYaw = Math.atan2(bz - center[1], bx - center[0]);
let gy = 0, gn = 0; for (const q of track) { gy += at(H0, q[0], q[1]); gn++; } gy /= gn;
let ymin = Infinity; for (const q of track) ymin = Math.min(ymin, q[2]);
const out = { center: { x: +center[0].toFixed(3), z: +center[1].toFixed(3) }, groundY: +gy.toFixed(3), trackBottomY: +ymin.toFixed(3), trackYawRad: +trackYaw.toFixed(4), houseYawRad: +houseYaw.toFixed(4),
  measured: { trackLength: +trackLen.toFixed(2), trackWidth: +trackWid.toFixed(2), boomReachFromCenter: +far[0].d.toFixed(2) }, spec: { model: 'Develon DX225LC-7 class', trackLength: 4.445, undercarriageWidth: 2.99, boom: 5.7, arm: 2.9, heightOverCab: 2.98, tailSwingRadius: 2.909, trackHeight: 0.945, houseWidth: 2.71, maxDigDepth: 6.59 }, nTrack: track.length, nBoom: boom.length, maskComponentCells: best.comp.length };
fs.writeFileSync(sceneDir + '/machine.json', JSON.stringify(out, null, 2)); console.log(JSON.stringify(out, null, 1));
