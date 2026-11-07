import { readPly } from './ply.mjs';
const p = readPly(process.argv[2]);
const { count, stride, data, idx } = p;
console.log('count', count, 'props', p.names.join(','));
const col = (name) => { const o = idx[name]; const a = new Float32Array(count); for (let i = 0; i < count; i++) a[i] = data[i * stride + o]; return a; };
const pct = (a, qs) => { const s = Float32Array.from(a).sort(); return qs.map(q => +s[Math.floor(q * (s.length - 1))].toFixed(4)); };
const Q = [0, .01, .05, .25, .5, .75, .95, .99, 1];
for (const n of ['x', 'y', 'z', 'scale_0', 'opacity', 'f_dc_0']) console.log(n, pct(col(n), Q));
// opacity as sigmoid
const op = col('opacity').map(v => 1 / (1 + Math.exp(-v)));
console.log('sigmoid opacity', pct(op, Q));
// PCA on opaque, small splats near center
const x = col('x'), y = col('y'), z = col('z');
const s0 = col('scale_0'), s1 = col('scale_1'), s2 = col('scale_2');
let n = 0, m = [0, 0, 0];
const sel = [];
for (let i = 0; i < count; i++) { if (op[i] > 0.5 && Math.max(s0[i], s1[i], s2[i]) < -3 && Math.hypot(x[i], y[i], z[i]) < 3) { sel.push(i); m[0] += x[i]; m[1] += y[i]; m[2] += z[i]; n++; } }
m = m.map(v => v / n);
const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
for (const i of sel) { const d = [x[i] - m[0], y[i] - m[1], z[i] - m[2]]; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += d[a] * d[b] / n; }
console.log('sel', n, 'mean', m.map(v => v.toFixed(3)), 'cov', JSON.stringify(C.map(r => r.map(v => +v.toFixed(4)))));
// power iteration for smallest eigenvector via inverse: just print all via Jacobi
function jacobi(A) { A = A.map(r => r.slice()); let V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; for (let it = 0; it < 50; it++) { let p = 0, q = 1; for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) if (Math.abs(A[i][j]) > Math.abs(A[p][q])) { p = i; q = j; } if (Math.abs(A[p][q]) < 1e-12) break; const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]); const c = Math.cos(th), s = Math.sin(th); const J = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; J[p][p] = c; J[q][q] = c; J[p][q] = s; J[q][p] = -s; const mul = (X, Y) => X.map((r, i) => Y[0].map((_, j) => r.reduce((acc, _, k) => acc + X[i][k] * Y[k][j], 0))); const JT = J[0].map((_, i) => J.map(r => r[i])); A = mul(mul(JT, A), J); V = mul(V, J); } return { vals: [A[0][0], A[1][1], A[2][2]], vecs: [0, 1, 2].map(k => [V[0][k], V[1][k], V[2][k]]) }; }
const e = jacobi(C); console.log('eig', e.vals.map(v => v.toFixed(4)), e.vecs.map(v => v.map(c => c.toFixed(3))));
