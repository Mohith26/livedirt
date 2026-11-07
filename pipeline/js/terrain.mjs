// Derive terrain products from a leveled, metric 3DGS PLY:
//  T  = top surface (robust high percentile of opaque splats per cell)
//  G  = object-free ground via morphological opening of T
//  H0 = terrain used by the sim (T where it is ground, G under objects)
//  M  = object mask (excavator, vehicles, clutter) where dig is disabled
//  plus a soil palette learned from the scene's own surfaces.
import fs from 'node:fs';
import { readPly } from './ply.mjs';
export function buildTerrain(plyPath, { cell = 0.1, half = 24, openRadius = 2.6, machineMin = 4, raise = 0.3, grow = 0.5 }) {
  const p = readPly(plyPath); const { count, stride, data, idx } = p;
  const N = Math.round(2 * half / cell); const sig = v => 1 / (1 + Math.exp(-v)); const C0 = 0.28209479177387814;
  const lists = new Array(N * N);
  for (let i = 0; i < count; i++) { const b = i * stride; if (sig(data[b + idx.opacity]) < 0.5) continue;
    const x = data[b + idx.x], y = data[b + idx.y], z = data[b + idx.z];
    const cx = Math.floor((x + half) / cell), cz = Math.floor((z + half) / cell); if (cx < 0 || cz < 0 || cx >= N || cz >= N) continue;
    (lists[cz * N + cx] ||= []).push(y); }
  const T = new Float32Array(N * N).fill(NaN); const dens = new Uint16Array(N * N);
  for (let k = 0; k < N * N; k++) { const L = lists[k]; if (!L || L.length < 2) continue; L.sort((a, b) => a - b); T[k] = L[Math.floor((L.length - 1) * 0.85)]; dens[k] = Math.min(65535, L.length); }
  fillNaN(T, N);
  const Tm = median3(T, N);
  // coarse ground reference: opening removes narrow tall things (machines); only used to decide "is this splat raised"
  const r = Math.round(openRadius / cell);
  const G = boxBlur(dilate(erode(Tm, N, r), N, r), N, Math.round(0.6 / cell));
  // machine evidence: saturated orange paint or bright neutral panels/glass, raised above local ground
  const ev = new Uint16Array(N * N);
  for (let i = 0; i < count; i++) { const b = i * stride; if (sig(data[b + idx.opacity]) < 0.4) continue;
    const x = data[b + idx.x], y = data[b + idx.y], z = data[b + idx.z];
    const cx = Math.floor((x + half) / cell), cz = Math.floor((z + half) / cell); if (cx < 0 || cz < 0 || cx >= N || cz >= N) continue;
    const k = cz * N + cx; if (y < G[k] + raise) continue;
    const R = 0.5 + C0 * data[b + idx.f_dc_0], Gc = 0.5 + C0 * data[b + idx.f_dc_1], B = 0.5 + C0 * data[b + idx.f_dc_2];
    const mx = Math.max(R, Gc, B), mn = Math.min(R, Gc, B), sat = mx > 0 ? (mx - mn) / mx : 0, lum = 0.299 * R + 0.587 * Gc + 0.114 * B;
    const orange = R > 0.45 && R - B > 0.22 && sat > 0.35; const panel = lum > 0.78; const blue = B - R > 0.12 && sat > 0.3;
    if (orange || panel || blue) ev[k]++; }
  let M = new Uint8Array(N * N); for (let k = 0; k < N * N; k++) M[k] = ev[k] >= machineMin ? 1 : 0;
  // close small gaps (dilate then erode), then grow a safety margin
  M = erodeMask(dilateMask(M, N, 4), N, 3); M = dilateMask(M, N, Math.round(grow / cell));
  // drop tiny isolated blobs (< 1.5 m^2)
  M = dropSmall(M, N, Math.round(2.5 / (cell * cell)));
  // terrain: top surface outside objects; under objects, inpaint from surrounding ground
  const H0 = new Float32Array(N * N); for (let k = 0; k < N * N; k++) H0[k] = M[k] ? NaN : Tm[k];
  fillNaN(H0, N);
  return { N, cell, half, T: Tm, H0: boxBlur(H0, N, 1), M, dens, ev };
}
function erodeMask(M, N, r) { const out = new Uint8Array(N * N); for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) { let all = 1; for (let dz = -r; dz <= r && all; dz++) for (let dx = -r; dx <= r; dx++) { if (dx * dx + dz * dz > r * r) continue; const xx = x + dx, zz = z + dz; if (xx < 0 || zz < 0 || xx >= N || zz >= N || !M[zz * N + xx]) { all = 0; break; } } out[z * N + x] = all; } return out; }
function dropSmall(M, N, minCells) { const out = M.slice(); const seen = new Uint8Array(N * N); const st = [];
  for (let s = 0; s < N * N; s++) { if (!M[s] || seen[s]) continue; const comp = []; st.push(s); seen[s] = 1;
    while (st.length) { const k = st.pop(); comp.push(k); const x = k % N, z = (k / N) | 0; for (const [dx, dz] of [[1,0],[-1,0],[0,1],[0,-1]]) { const xx = x + dx, zz = z + dz; if (xx < 0 || zz < 0 || xx >= N || zz >= N) continue; const kk = zz * N + xx; if (M[kk] && !seen[kk]) { seen[kk] = 1; st.push(kk); } } }
    if (comp.length < minCells) for (const k of comp) out[k] = 0; }
  return out; }
function fillNaN(A, N) { for (let pass = 0; pass < 400; pass++) { let left = 0; const B = A.slice(); for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) { const k = z * N + x; if (!isNaN(A[k])) continue; let s = 0, c = 0; for (const [dx, dz] of [[1,0],[-1,0],[0,1],[0,-1]]) { const xx = x + dx, zz = z + dz; if (xx < 0 || zz < 0 || xx >= N || zz >= N) continue; const v = A[zz * N + xx]; if (!isNaN(v)) { s += v; c++; } } if (c) B[k] = s / c; else left++; } A.set(B); if (!left) break; } for (let k = 0; k < A.length; k++) if (isNaN(A[k])) A[k] = 0; }
function median3(A, N) { const B = new Float32Array(N * N); const w = []; for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) { w.length = 0; for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) { const xx = Math.min(N - 1, Math.max(0, x + dx)), zz = Math.min(N - 1, Math.max(0, z + dz)); w.push(A[zz * N + xx]); } w.sort((a, b) => a - b); B[z * N + x] = w[4]; } return B; }
// separable square erosion/dilation via van Herk style sliding (simple O(N*r) is fine for N=480)
function morph(A, N, r, fn) { const tmp = new Float32Array(N * N), out = new Float32Array(N * N);
  for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) { let v = fn === 'min' ? Infinity : -Infinity; for (let d = -r; d <= r; d++) { const xx = x + d; if (xx < 0 || xx >= N) continue; const a = A[z * N + xx]; v = fn === 'min' ? Math.min(v, a) : Math.max(v, a); } tmp[z * N + x] = v; }
  for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) { let v = fn === 'min' ? Infinity : -Infinity; for (let d = -r; d <= r; d++) { const zz = z + d; if (zz < 0 || zz >= N) continue; const a = tmp[zz * N + x]; v = fn === 'min' ? Math.min(v, a) : Math.max(v, a); } out[z * N + x] = v; }
  return out; }
const erode = (A, N, r) => morph(A, N, r, 'min'), dilate = (A, N, r) => morph(A, N, r, 'max');
function boxBlur(A, N, r) { if (r <= 0) return A.slice(); const tmp = new Float32Array(N * N), out = new Float32Array(N * N);
  for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) { let s = 0, c = 0; for (let d = -r; d <= r; d++) { const xx = x + d; if (xx < 0 || xx >= N) continue; s += A[z * N + xx]; c++; } tmp[z * N + x] = s / c; }
  for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) { let s = 0, c = 0; for (let d = -r; d <= r; d++) { const zz = z + d; if (zz < 0 || zz >= N) continue; s += tmp[zz * N + x]; c++; } out[z * N + x] = s / c; }
  return out; }
function dilateMask(M, N, r) { const out = new Uint8Array(N * N); for (let z = 0; z < N; z++) for (let x = 0; x < N; x++) { if (!M[z * N + x]) continue; for (let dz = -r; dz <= r; dz++) for (let dx = -r; dx <= r; dx++) { if (dx * dx + dz * dz > r * r) continue; const xx = x + dx, zz = z + dz; if (xx < 0 || zz < 0 || xx >= N || zz >= N) continue; out[zz * N + xx] = 1; } } return out; }
