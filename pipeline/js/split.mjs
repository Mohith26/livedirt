// Split a PLY radially into inner (r < R) and outer (r >= R) tiles.
import fs from 'node:fs'; import { readPly } from './ply.mjs';
const [,, inPly, R, innerOut, outerOut] = process.argv; const p = readPly(inPly); const { count, stride, data, idx } = p; const r2 = (+R) ** 2;
const inner = [], outer = []; for (let i = 0; i < count; i++) { const b = i * stride; const x = data[b + idx.x], z = data[b + idx.z]; (x * x + z * z < r2 ? inner : outer).push(i); }
const write = (ids, path) => { const out = new Float32Array(ids.length * stride); ids.forEach((i, j) => out.set(data.subarray(i * stride, i * stride + stride), j * stride));
  const h = `ply\nformat binary_little_endian 1.0\nelement vertex ${ids.length}\n` + p.names.map(n => `property float ${n}`).join('\n') + '\nend_header\n'; fs.writeFileSync(path, Buffer.concat([Buffer.from(h, 'latin1'), Buffer.from(out.buffer)])); };
write(inner, innerOut); write(outer, outerOut); console.log('inner', inner.length, 'outer', outer.length);
