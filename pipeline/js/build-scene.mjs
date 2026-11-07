// End-to-end scene build: leveled+scaled PLY -> cleaned PLY -> terrain products + soil palette.
import fs from 'node:fs'; import path from 'node:path';
import { transformPly } from './level.mjs'; import { buildTerrain } from './terrain.mjs'; import { readPly } from './ply.mjs'; import { writeBmp } from './bmp.mjs';
const [,, inPly, outDir, workDir] = process.argv; fs.mkdirSync(outDir, { recursive: true });
const cfg = JSON.parse(fs.readFileSync(new URL('../scene.config.json', import.meta.url)));
const S = cfg.metricScale, R_SITE = cfg.siteRadius;
// pass 1: level + scale + crop to site disk
const p1 = path.join(workDir, 'metric.ply');
const n1 = transformPly(inPly, p1, { R: cfg.R, origin: cfg.origin, scale: S, yaw: cfg.yaw || 0, filter: (x, y, z) => x * x + z * z < R_SITE * R_SITE && y > -3 && y < 9 });
console.log('pass1 kept', n1);
// terrain from metric cloud
const t = buildTerrain(p1, cfg.terrain);
const { N, cell, half } = t; const sample = (A, x, z) => { const cx = Math.min(N - 1, Math.max(0, Math.floor((x + half) / cell))), cz = Math.min(N - 1, Math.max(0, Math.floor((z + half) / cell))); return A[cz * N + cx]; };
// pass 2: remove sub-surface reconstruction noise (would show through when digging) and near-invisible splats
const p = readPly(p1); const { count, stride, data, idx } = p; const sig = v => 1 / (1 + Math.exp(-v));
const keep = []; let below = 0, faint = 0;
for (let i = 0; i < count; i++) { const b = i * stride; const x = data[b + idx.x], y = data[b + idx.y], z = data[b + idx.z]; const op = sig(data[b + idx.opacity]);
  if (op < 0.02) { faint++; continue; }
  if (y < sample(t.H0, x, z) - 0.35 && !sample(t.M, x, z)) { below++; continue; }
  keep.push(i); }
console.log('removed below-ground', below, 'faint', faint, 'kept', keep.length);
const out = new Float32Array(keep.length * stride); keep.forEach((i, j) => out.set(data.subarray(i * stride, i * stride + stride), j * stride));
const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${keep.length}\n` + p.names.map(nm => `property float ${nm}`).join('\n') + '\nend_header\n';
const cleanPly = path.join(workDir, 'clean.ply'); fs.writeFileSync(cleanPly, Buffer.concat([Buffer.from(header, 'latin1'), Buffer.from(out.buffer)]));
// soil palette: learn colors of terrain-bound splats, bucketed by local slope (flat surface vs cut faces) and height
const C0 = 0.28209479177387814; const slopeAt = (x, z) => { const d = cell * 2; const gx = (sample(t.H0, x + d, z) - sample(t.H0, x - d, z)) / (2 * d), gz = (sample(t.H0, x, z + d) - sample(t.H0, x, z - d)) / (2 * d); return Math.atan(Math.hypot(gx, gz)) * 180 / Math.PI; };
const buckets = { flat: [], gentle: [], steep: [], pile: [] };
for (let j = 0; j < keep.length; j += 3) { const b = j * stride; const x = out[b + idx.x], y = out[b + idx.y], z = out[b + idx.z]; if (sample(t.M, x, z)) continue; if (Math.abs(y - sample(t.H0, x, z)) > 0.15 || sig(out[b + idx.opacity]) < 0.6) continue;
  const s = slopeAt(x, z); const h = sample(t.H0, x, z); const c = [0, 1, 2].map(k => Math.max(0, Math.min(1, 0.5 + C0 * out[b + idx['f_dc_' + k]])));
  (h > 0.8 ? buckets.pile : s > 28 ? buckets.steep : s > 12 ? buckets.gentle : buckets.flat).push(c); }
const stats = (arr) => { const n = arr.length; if (!n) return null; const m = [0, 1, 2].map(k => arr.reduce((a, c) => a + c[k], 0) / n); const sd = [0, 1, 2].map(k => Math.sqrt(arr.reduce((a, c) => a + (c[k] - m[k]) ** 2, 0) / n)); const lum = arr.map(c => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]).sort((a, b) => a - b); return { n, mean: m.map(v => +v.toFixed(4)), sd: sd.map(v => +v.toFixed(4)), lumP10: +lum[Math.floor(n * .1)].toFixed(4), lumP50: +lum[Math.floor(n * .5)].toFixed(4), lumP90: +lum[Math.floor(n * .9)].toFixed(4) }; };
const palette = Object.fromEntries(Object.entries(buckets).map(([k, v]) => [k, stats(v)]));
// write terrain.bin: [H0 f32 N*N][T f32 N*N][M u8 N*N]
const bin = Buffer.concat([Buffer.from(t.H0.buffer), Buffer.from(t.T.buffer), Buffer.from(t.M.buffer)]);
fs.writeFileSync(path.join(outDir, 'terrain.bin'), bin);
const meta = { version: 1, N, cell, half, layout: ['H0:f32', 'T:f32', 'M:u8'], metricScale: S, scaleSource: cfg.scaleSource, siteRadius: R_SITE, splats: keep.length, palette, source: cfg.source };
fs.writeFileSync(path.join(outDir, 'terrain.json'), JSON.stringify(meta, null, 2));
// debug images
const img = new Uint8Array(N * N * 3); let hmin = Infinity, hmax = -Infinity; for (const v of t.H0) { hmin = Math.min(hmin, v); hmax = Math.max(hmax, v); }
for (let k = 0; k < N * N; k++) { const v = (t.H0[k] - hmin) / (hmax - hmin); const g = Math.round(40 + 200 * v); img.set(t.M[k] ? [220, 60, 40] : [g, g, g], k * 3); }
writeBmp(path.join(workDir, 'terrain_dbg.bmp'), N, N, img);
console.log(JSON.stringify({ N, hmin, hmax, palette }, null, 1));
// albedo bake (top-down color of terrain-near splats) appended to terrain.bin as RGB u8 N*N
{ const alb = new Float32Array(N * N * 3), w = new Float32Array(N * N);
  for (let j = 0; j < keep.length; j++) { const b = j * stride; const x = out[b + idx.x], y = out[b + idx.y], z = out[b + idx.z]; const cx = Math.floor((x + half) / cell), cz = Math.floor((z + half) / cell); if (cx < 0 || cz < 0 || cx >= N || cz >= N) continue; const k = cz * N + cx;
    if (Math.abs(y - t.T[k]) > 0.25) continue; const a = sig(out[b + idx.opacity]); for (let c = 0; c < 3; c++) alb[k * 3 + c] += a * Math.max(0, Math.min(1, 0.5 + C0 * out[b + idx['f_dc_' + c]])); w[k] += a; }
  const rgb = new Uint8Array(N * N * 3); for (let k = 0; k < N * N; k++) for (let c = 0; c < 3; c++) rgb[k * 3 + c] = w[k] > 0 ? Math.round(255 * alb[k * 3 + c] / w[k]) : 0;
  // fill empty
  for (let pass = 0; pass < 50; pass++) { let left = 0; for (let k = 0; k < N * N; k++) { if (w[k] > 0) continue; const x = k % N, z = (k / N) | 0; let s = [0, 0, 0], c = 0; for (const [dx, dz] of [[1,0],[-1,0],[0,1],[0,-1]]) { const xx = x + dx, zz = z + dz; if (xx < 0 || zz < 0 || xx >= N || zz >= N) continue; const kk = zz * N + xx; if (w[kk] > 0) { for (let q = 0; q < 3; q++) s[q] += rgb[kk * 3 + q]; c++; } } if (c) { for (let q = 0; q < 3; q++) rgb[k * 3 + q] = Math.round(s[q] / c); w[k] = 1e-6; } else left++; } if (!left) break; }
  fs.appendFileSync(path.join(outDir, 'terrain.bin'), Buffer.from(rgb.buffer));
  meta.layout.push('albedo:rgb8'); fs.writeFileSync(path.join(outDir, 'terrain.json'), JSON.stringify(meta, null, 2));
  writeBmp(path.join(workDir, 'albedo.bmp'), N, N, rgb); }
