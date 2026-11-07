/** Scene products built by pipeline/js/build-scene.mjs + machine pose measured from the capture. */
export interface PaletteStat { n: number; mean: [number, number, number]; sd: [number, number, number]; lumP10: number; lumP50: number; lumP90: number }

export interface TerrainMeta {
  version: number;
  N: number;
  cell: number;
  half: number;
  layout: string[];
  metricScale: number;
  scaleSource: string;
  siteRadius: number;
  splats: number;
  palette: { flat: PaletteStat; gentle: PaletteStat; steep: PaletteStat; pile: PaletteStat };
  source: { title: string; author: string; url: string; license: string; footage: string; reconstruction: string };
}

export interface MachineMeta {
  note: string;
  center: { x: number; z: number };
  groundY: number;
  trackYawDeg: number;
  houseYawDeg: number;
  spec: {
    model: string; source: string; trackLength: number; undercarriageWidth: number; trackGauge: number; shoeWidth: number; trackHeight: number;
    boom: number; arm: number; bucketTipRadius: number; bucketCapacityM3: number; heightOverCab: number; houseWidth: number;
    counterweightClearance: number; tailSwingRadius: number; maxDigDepth: number; maxReachGround: number; swingSpeedRpm: number;
  };
}

export interface SceneData {
  meta: TerrainMeta;
  machine: MachineMeta;
  H0: Float32Array;
  T: Float32Array;
  M: Uint8Array;
  albedo: Uint8Array; // rgb8, N*N*3
  /** object code per cell: 0 ground, 1 our excavator, 2 other object */
  objectCode: Uint8Array;
}

export async function loadSceneData(base: string, onProgress?: (f: number, label: string) => void): Promise<SceneData> {
  onProgress?.(0, 'terrain');
  const [meta, machine, binBuf] = await Promise.all([
    fetch(`${base}/terrain.json`).then((r) => r.json() as Promise<TerrainMeta>),
    fetch(`${base}/machine.json`).then((r) => r.json() as Promise<MachineMeta>),
    fetch(`${base}/terrain.bin`).then((r) => r.arrayBuffer()),
  ]);
  const N = meta.N, n = N * N;
  const expect = n * 4 * 2 + n + n * 3;
  if (binBuf.byteLength !== expect) throw new Error(`terrain.bin size ${binBuf.byteLength} != expected ${expect}`);
  const H0 = new Float32Array(binBuf, 0, n);
  const T = new Float32Array(binBuf, n * 4, n);
  const M = new Uint8Array(binBuf, n * 8, n);
  const albedo = new Uint8Array(binBuf, n * 9, n * 3);
  const objectCode = labelObjects(M, N, meta.cell, meta.half, machine.center.x, machine.center.z);
  onProgress?.(1, 'terrain');
  return { meta, machine, H0, T, M, albedo, objectCode };
}

/** connected components of the object mask; the component nearest the machine centre is "ours" */
export function labelObjects(M: Uint8Array, N: number, cell: number, half: number, mx: number, mz: number): Uint8Array {
  const code = new Uint8Array(N * N);
  const comp = new Int32Array(N * N).fill(-1);
  let best = -1, bestD = Infinity, id = 0;
  const stack: number[] = [];
  for (let s = 0; s < N * N; s++) {
    if (!M[s] || comp[s] >= 0) continue;
    let sx = 0, sz = 0, cnt = 0;
    stack.push(s);
    comp[s] = id;
    while (stack.length) {
      const k = stack.pop()!;
      const x = k % N, z = (k / N) | 0;
      sx += (x + 0.5) * cell - half; sz += (z + 0.5) * cell - half; cnt++;
      if (x > 0 && M[k - 1] && comp[k - 1] < 0) { comp[k - 1] = id; stack.push(k - 1); }
      if (x < N - 1 && M[k + 1] && comp[k + 1] < 0) { comp[k + 1] = id; stack.push(k + 1); }
      if (z > 0 && M[k - N] && comp[k - N] < 0) { comp[k - N] = id; stack.push(k - N); }
      if (z < N - 1 && M[k + N] && comp[k + N] < 0) { comp[k + N] = id; stack.push(k + N); }
    }
    const d = Math.hypot(sx / cnt - mx, sz / cnt - mz);
    if (d < bestD) { bestD = d; best = id; }
    id++;
  }
  for (let k = 0; k < N * N; k++) code[k] = comp[k] < 0 ? 0 : comp[k] === best ? 1 : 2;
  return code;
}
