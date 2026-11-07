import { readPly } from './ply.mjs';
import { writeBmp } from './bmp.mjs';
const OUT = process.argv[3];
const p = readPly(process.argv[2]); const { count, stride, data, idx } = p;
const g = (i, n) => data[i * stride + idx[n]];
const sig = v => 1 / (1 + Math.exp(-v));
// candidates: opaque, small
const cand = [];
for (let i = 0; i < count; i++) if (sig(g(i, 'opacity')) > 0.6 && Math.max(g(i,'scale_0'),g(i,'scale_1'),g(i,'scale_2')) < -2.5) cand.push(i);
console.log('cand', cand.length);
let rng = 12345; const rnd = () => (rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648;
let best = null;
const P = i => [g(i,'x'), g(i,'y'), g(i,'z')];
const sub = cand.filter((_, k) => k % 20 === 0);
for (let it = 0; it < 3000; it++) {
  const a = P(cand[Math.floor(rnd()*cand.length)]), b = P(cand[Math.floor(rnd()*cand.length)]), c = P(cand[Math.floor(rnd()*cand.length)]);
  const u = [b[0]-a[0], b[1]-a[1], b[2]-a[2]], v = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
  let nrm = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]; const L = Math.hypot(...nrm); if (L < 1e-6) continue; nrm = nrm.map(x => x/L);
  if (Math.abs(nrm[1]) < 0.7) continue; // expect roughly y-axis normal
  const d = -(nrm[0]*a[0]+nrm[1]*a[1]+nrm[2]*a[2]);
  let inl = 0; for (const i of sub) { const q = P(i); if (Math.abs(nrm[0]*q[0]+nrm[1]*q[1]+nrm[2]*q[2]+d) < 0.08) inl++; }
  if (!best || inl > best.inl) best = { inl, nrm, d };
}
// refine with least squares on inliers (y = ax + bz + c)
const inl = cand.filter(i => { const q = P(i); return Math.abs(best.nrm[0]*q[0]+best.nrm[1]*q[1]+best.nrm[2]*q[2]+best.d) < 0.08; });
let S = [[0,0,0],[0,0,0],[0,0,0]], r = [0,0,0];
for (const i of inl) { const [x,y,z] = P(i); const f = [x, z, 1]; for (let a = 0; a < 3; a++) { r[a] += f[a]*y; for (let b = 0; b < 3; b++) S[a][b] += f[a]*f[b]; } }
const det = M => M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
const D = det(S); const sol = [0,1,2].map(k => det(S.map((row, i) => row.map((v, j) => j === k ? r[i] : v))) / D);
let n = [sol[0], -1, sol[1]]; const L = Math.hypot(...n); n = n.map(v => v/L);
// choose sign so "up" side has more off-plane mass
let above = 0, below = 0; for (let i = 0; i < count; i += 7) { const q = P(i); const s = n[0]*q[0]+n[1]*q[1]+n[2]*q[2] + sol[2]*(-n[1]); if (s > 0.15) above++; else if (s < -0.15) below++; }
if (below > above) n = n.map(v => -v);
console.log('inliers', inl.length, 'plane y = %f x + %f z + %f', ...sol.map(v=>+v.toFixed(5)), 'up normal', n.map(v=>+v.toFixed(4)), 'above/below', above, below);
// Build rotation taking n -> +Y, translate plane point to origin
const up = n; const tmp = Math.abs(up[0]) < 0.9 ? [1,0,0] : [0,0,1];
let ex = [tmp[1]*up[2]-tmp[2]*up[1], tmp[2]*up[0]-tmp[0]*up[2], tmp[0]*up[1]-tmp[1]*up[0]]; let lx = Math.hypot(...ex); ex = ex.map(v=>v/lx);
// we want ex to be close to original +x: project
ex = [1,0,0]; const dt = ex[0]*up[0]+ex[1]*up[1]+ex[2]*up[2]; ex = ex.map((v,i)=>v-dt*up[i]); lx = Math.hypot(...ex); ex = ex.map(v=>v/lx);
const ez = [ex[1]*up[2]-ex[2]*up[1], ex[2]*up[0]-ex[0]*up[2], ex[0]*up[1]-ex[1]*up[0]];
// rows of R: ex, up, ez  (new = R * old)
const origin = [0, sol[2], 0];
const R = [ex, up, ez];
console.log('R', JSON.stringify(R.map(r=>r.map(v=>+v.toFixed(6)))), 'origin', origin);
const tf = q => { const d = [q[0]-origin[0], q[1]-origin[1], q[2]-origin[2]]; return R.map(row => row[0]*d[0]+row[1]*d[1]+row[2]*d[2]); };
// top-down image
const W = 1024, H = 1024, X0 = -32, X1 = 32, Z0 = -32, Z1 = 32;
const img = new Uint8Array(W*H*3).fill(30); const zb = new Float32Array(W*H).fill(-1e9);
const side = new Uint8Array(W*400*3).fill(30); const sb = new Float32Array(W*400).fill(-1e9);
const C0 = 0.28209479177387814; const cl = v => Math.max(0, Math.min(255, Math.round((0.5 + C0*v)*255)));
const hist = new Float64Array(200);
for (let i = 0; i < count; i++) {
  if (sig(g(i,'opacity')) < 0.3) continue;
  const [x, y, z] = tf(P(i));
  const hb = Math.floor((y + 5) * 10); if (hb >= 0 && hb < 200) hist[hb]++;
  const px = Math.floor((x - X0)/(X1-X0)*W), pz = Math.floor((z - Z0)/(Z1-Z0)*H);
  const c = [cl(g(i,'f_dc_0')), cl(g(i,'f_dc_1')), cl(g(i,'f_dc_2'))];
  if (px >= 0 && px < W && pz >= 0 && pz < H && y < 6) { const k = pz*W+px; if (y > zb[k]) { zb[k] = y; img.set(c, k*3); } }
  // side view: x vs y, looking along -z (depth = z)
  const sy = Math.floor((4 - y)/10*400);
  if (px >= 0 && px < W && sy >= 0 && sy < 400) { const k = sy*W+px; if (z > sb[k]) { sb[k] = z; side.set(c, k*3); } }
}
// grid lines every 5 units
for (let v = -30; v <= 30; v += 5) { const px = Math.floor((v - X0)/(X1-X0)*W); for (let t = 0; t < H; t += 3) { img.set([255,0,0], (t*W+px)*3); img.set([255,0,0], (px*W+t)*3); } }
writeBmp(OUT + '/top.bmp', W, H, img); writeBmp(OUT + '/side.bmp', W, 400, side);
console.log('height hist (y from -5, 0.1 bins):', Array.from(hist).map((v,i)=>[(i/10-5).toFixed(1), v]).filter(a=>a[1]>2000).map(a=>a.join(':')).join(' '));
