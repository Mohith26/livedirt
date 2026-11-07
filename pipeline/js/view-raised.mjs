import fs from 'node:fs'; import { readPly } from './ply.mjs'; import { writeBmp } from './bmp.mjs';
const [,, plyPath, sceneDir, out, cx0, cz0, span, lo, hi] = process.argv; const CX = +cx0, CZ = +cz0, S = +span, LO = +lo, HI = +hi;
const meta = JSON.parse(fs.readFileSync(sceneDir + '/terrain.json')); const bin = fs.readFileSync(sceneDir + '/terrain.bin'); const { N, cell, half } = meta; const H0 = new Float32Array(bin.buffer, bin.byteOffset, N * N);
const at = (x, z) => H0[Math.min(N - 1, Math.max(0, Math.floor((z + half) / cell))) * N + Math.min(N - 1, Math.max(0, Math.floor((x + half) / cell)))];
const p = readPly(plyPath); const { count, stride, data, idx } = p; const C0 = 0.28209479177387814; const W = 800; const img = new Uint8Array(W * W * 3).fill(20); const zb = new Float32Array(W * W).fill(-1e9);
for (let i = 0; i < count; i++) { const b = i * stride; if (1 / (1 + Math.exp(-data[b + idx.opacity])) < 0.5) continue; const x = data[b + idx.x], y = data[b + idx.y], z = data[b + idx.z];
  const px = Math.floor((x - CX + S / 2) / S * W), pz = Math.floor((z - CZ + S / 2) / S * W); if (px < 0 || pz < 0 || px >= W || pz >= W) continue; const rel = y - at(x, z); if (rel < LO || rel > HI) continue;
  const k = pz * W + px; if (y > zb[k]) { zb[k] = y; for (let c = 0; c < 3; c++) img[k * 3 + c] = Math.max(0, Math.min(255, Math.round((0.5 + C0 * data[b + idx['f_dc_' + c]]) * 255))); } }
for (let v = Math.ceil(CX - S / 2); v < CX + S / 2; v++) { const px = Math.floor((v - CX + S / 2) / S * W); for (let t = 0; t < W; t += 3) img.set([200, 0, 0], (t * W + px) * 3); }
for (let v = Math.ceil(CZ - S / 2); v < CZ + S / 2; v++) { const pz = Math.floor((v - CZ + S / 2) / S * W); for (let t = 0; t < W; t += 3) img.set([200, 0, 0], (pz * W + t) * 3); }
writeBmp(out, W, W, img);
