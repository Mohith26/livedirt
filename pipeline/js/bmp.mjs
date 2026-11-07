import fs from 'node:fs';
export function writeBmp(path, w, h, rgb) { // rgb: Uint8Array w*h*3, row 0 = top
  const rowSize = Math.ceil(w * 3 / 4) * 4; const size = 54 + rowSize * h; const b = Buffer.alloc(size);
  b.write('BM', 0); b.writeUInt32LE(size, 2); b.writeUInt32LE(54, 10); b.writeUInt32LE(40, 14); b.writeInt32LE(w, 18); b.writeInt32LE(h, 22);
  b.writeUInt16LE(1, 26); b.writeUInt16LE(24, 28); b.writeUInt32LE(rowSize * h, 34);
  for (let y = 0; y < h; y++) { const row = 54 + (h - 1 - y) * rowSize; for (let x = 0; x < w; x++) { const i = (y * w + x) * 3; b[row + x * 3] = rgb[i + 2]; b[row + x * 3 + 1] = rgb[i + 1]; b[row + x * 3 + 2] = rgb[i]; } }
  fs.writeFileSync(path, b);
}
