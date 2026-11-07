// Minimal binary little-endian PLY reader for 3DGS files (float properties only).
import fs from 'node:fs';
export function readPly(path) {
  const buf = fs.readFileSync(path);
  const headerEnd = buf.indexOf('end_header\n') + 'end_header\n'.length;
  const header = buf.subarray(0, headerEnd).toString('latin1');
  const count = +header.match(/element vertex (\d+)/)[1];
  const props = [...header.matchAll(/property (\w+) (\w+)/g)].map(m => ({ type: m[1], name: m[2] }));
  if (props.some(p => p.type !== 'float')) throw new Error('only float props supported');
  const stride = props.length;
  const data = new Float32Array(buf.buffer.slice(buf.byteOffset + headerEnd, buf.byteOffset + headerEnd + count * stride * 4));
  const idx = Object.fromEntries(props.map((p, i) => [p.name, i]));
  return { count, stride, data, idx, names: props.map(p => p.name) };
}
