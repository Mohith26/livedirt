import { readPly } from './ply.mjs';
const p = readPly(process.argv[2]); const { count, stride, data, idx } = p;
// height profile under excavator: find splats in region
const pts = [];
for (let i = 0; i < count; i++) { const b = i*stride; const x = data[b+idx.x], y = data[b+idx.y], z = data[b+idx.z];
  if (x > -7 && x < 0 && z > -2.5 && z < 4.5 && 1/(1+Math.exp(-data[b+idx.opacity])) > 0.5) pts.push([x, y, z, data[b+idx.f_dc_0], data[b+idx.f_dc_1], data[b+idx.f_dc_2]]); }
console.log('n', pts.length);
// height histogram
const H = {}; for (const q of pts) { const k = (Math.floor(q[1]*10)/10).toFixed(1); H[k] = (H[k]||0)+1; } console.log(Object.entries(H).sort((a,b)=>+a[0]-+b[0]).map(e=>e.join(':')).join(' '));
// for a band of heights, scan angles to find the direction minimizing projected width (track direction = long axis of track band)
for (const [lo, hi] of [[2.0,2.4],[2.2,2.6],[2.4,2.8]]) {
  const band = pts.filter(q => q[1] > lo && q[1] < hi);
  let best = null;
  for (let a = 0; a < 180; a += 1) { const t = a*Math.PI/180, u = [Math.cos(t), Math.sin(t)];
    const proj = band.map(q => q[0]*u[0] + q[2]*u[1]).sort((a,b)=>a-b); const perp = band.map(q => -q[0]*u[1] + q[2]*u[0]).sort((a,b)=>a-b);
    const L = proj[Math.floor(proj.length*0.99)] - proj[Math.floor(proj.length*0.01)], Wd = perp[Math.floor(perp.length*0.99)] - perp[Math.floor(perp.length*0.01)];
    if (!best || L*Wd < best.area) best = { a, L, Wd, area: L*Wd };
  }
  console.log('band', lo, hi, 'n', band.length, 'minAreaRect angle', best.a, 'len', best.L.toFixed(2), 'wid', best.Wd.toFixed(2));
}
