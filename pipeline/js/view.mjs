// Ortho preview renders of a leveled PLY: top view (x right, z down) and side view.
import { readPly } from './ply.mjs'; import { writeBmp } from './bmp.mjs';
const [,, path, out, x0, x1, z0, z1, W = 900] = process.argv; const X0 = +x0, X1 = +x1, Z0 = +z0, Z1 = +z1, w = +W, h = Math.round(w * (Z1 - Z0) / (X1 - X0));
const p = readPly(path); const { count, stride, data, idx } = p; const C0 = 0.28209479177387814;
const cl = v => Math.max(0, Math.min(255, Math.round((0.5 + C0 * v) * 255)));
const img = new Uint8Array(w * h * 3).fill(25), zb = new Float32Array(w * h).fill(-1e9);
const sh = 300, side = new Uint8Array(w * sh * 3).fill(25), sb = new Float32Array(w * sh).fill(-1e9);
for (let i = 0; i < count; i++) { const b = i * stride; if (1/(1+Math.exp(-data[b+idx.opacity])) < 0.35) continue;
  const x = data[b+idx.x], y = data[b+idx.y], z = data[b+idx.z]; if (x < X0 || x >= X1 || z < Z0 || z >= Z1) continue;
  const c = [cl(data[b+idx.f_dc_0]), cl(data[b+idx.f_dc_1]), cl(data[b+idx.f_dc_2])];
  const px = Math.floor((x - X0) / (X1 - X0) * w), pz = Math.floor((z - Z0) / (Z1 - Z0) * h); const k = pz * w + px; if (y > zb[k] && y < 8) { zb[k] = y; img.set(c, k * 3); }
  const yy = Math.floor((6 - y) / 8 * sh); if (yy >= 0 && yy < sh) { const k2 = yy * w + px; if (-z > sb[k2]) { sb[k2] = -z; side.set(c, k2 * 3); } } }
// 1-unit tick grid
for (let v = Math.ceil(X0); v < X1; v++) { const px = Math.floor((v - X0) / (X1 - X0) * w); for (let t = 0; t < h; t += 4) img.set(v % 5 ? [90,90,0] : [255,0,0], (t * w + px) * 3); }
for (let v = Math.ceil(Z0); v < Z1; v++) { const pz = Math.floor((v - Z0) / (Z1 - Z0) * h); for (let t = 0; t < w; t += 4) img.set(v % 5 ? [90,90,0] : [255,0,0], (pz * w + t) * 3); }
for (let v = -1; v <= 6; v++) { const yy = Math.floor((6 - v) / 8 * sh); for (let t = 0; t < w; t += 4) side.set(v ? [90,90,0] : [255,0,0], (yy * w + t) * 3); }
writeBmp(out + '_top.bmp', w, h, img); writeBmp(out + '_side.bmp', w, sh, side);
