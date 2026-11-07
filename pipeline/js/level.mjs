// Apply rigid transform (rotation R rows ex,up,ez + origin) and uniform scale to a 3DGS PLY.
import fs from 'node:fs';
import { readPly } from './ply.mjs';
export function matToQuat(m) { // m rows; returns [w,x,y,z]
  const t = m[0][0] + m[1][1] + m[2][2]; let w, x, y, z;
  if (t > 0) { const s = Math.sqrt(t + 1) * 2; w = s / 4; x = (m[2][1] - m[1][2]) / s; y = (m[0][2] - m[2][0]) / s; z = (m[1][0] - m[0][1]) / s; }
  else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2; w = (m[2][1] - m[1][2]) / s; x = s / 4; y = (m[0][1] + m[1][0]) / s; z = (m[0][2] + m[2][0]) / s; }
  else if (m[1][1] > m[2][2]) { const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2; w = (m[0][2] - m[2][0]) / s; x = (m[0][1] + m[1][0]) / s; y = s / 4; z = (m[1][2] + m[2][1]) / s; }
  else { const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2; w = (m[1][0] - m[0][1]) / s; x = (m[0][2] + m[2][0]) / s; y = (m[1][2] + m[2][1]) / s; z = s / 4; }
  return [w, x, y, z];
}
const qmul = (a, b) => [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2], a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];
export function transformPly(inPath, outPath, { R, origin, scale, yaw = 0, filter }) {
  const p = readPly(inPath); const { count, stride, data, idx } = p;
  // optional yaw about up axis applied after leveling
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const Y = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const M = Y.map(r => [0, 1, 2].map(j => r[0]*R[0][j] + r[1]*R[1][j] + r[2]*R[2][j]));
  const qR = matToQuat(M); const ls = Math.log(scale);
  const keep = new Uint8Array(count); let n = 0;
  const out = new Float32Array(count * stride);
  for (let i = 0; i < count; i++) {
    const b = i * stride; const d = [data[b+idx.x]-origin[0], data[b+idx.y]-origin[1], data[b+idx.z]-origin[2]];
    const x = (M[0][0]*d[0]+M[0][1]*d[1]+M[0][2]*d[2]) * scale, y = (M[1][0]*d[0]+M[1][1]*d[1]+M[1][2]*d[2]) * scale, z = (M[2][0]*d[0]+M[2][1]*d[1]+M[2][2]*d[2]) * scale;
    if (filter && !filter(x, y, z, data, b, idx)) continue;
    const o = n * stride; out.set(data.subarray(b, b + stride), o);
    out[o+idx.x] = x; out[o+idx.y] = y; out[o+idx.z] = z;
    let q = [data[b+idx.rot_0], data[b+idx.rot_1], data[b+idx.rot_2], data[b+idx.rot_3]]; const ql = Math.hypot(...q) || 1; q = q.map(v => v/ql);
    const qn = qmul(qR, q); out[o+idx.rot_0] = qn[0]; out[o+idx.rot_1] = qn[1]; out[o+idx.rot_2] = qn[2]; out[o+idx.rot_3] = qn[3];
    out[o+idx.scale_0] += ls; out[o+idx.scale_1] += ls; out[o+idx.scale_2] += ls;
    n++;
  }
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${n}\n` + p.names.map(nm => `property float ${nm}`).join('\n') + '\nend_header\n';
  fs.writeFileSync(outPath, Buffer.concat([Buffer.from(header, 'latin1'), Buffer.from(out.buffer, 0, n * stride * 4)]));
  return n;
}
if (process.argv[1].endsWith('level.mjs')) {
  const cfg = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
  const n = transformPly(process.argv[2], process.argv[3], cfg); console.log('wrote', n);
}
