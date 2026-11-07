import * as THREE from 'three';
import { SplatMesh } from '@sparkjsdev/spark';
import { Heightfield, type Rect } from '../sim/heightfield';
import { HeightRaycaster } from '../sim/heightRay';
import { soilById, type SoilType } from '../sim/soil';
import { makeTerrainDeformer, makeGradeOnly, type DeformUniforms } from '../render/deform';
import { GeneratedSoil, type SoilPalette } from '../render/generatedSoil';
import type { SceneData } from '../data/scene';

/**
 * Owns the earthwork simulation and everything that mirrors it on the GPU:
 * float textures sampled by the splat deformer, the captured splat tiles, and the generated-soil layer.
 */
export class TerrainSystem {
  readonly data: SceneData;
  readonly hf: Heightfield;
  /** raycaster over the current terrain surface H (for picking) */
  readonly ray: HeightRaycaster;
  /** raycaster over terrain + obstacles (for sensors) */
  readonly sensorRay: HeightRaycaster;
  readonly tStatic: THREE.DataTexture;
  readonly tDyn: THREE.DataTexture;
  readonly uniforms: DeformUniforms;
  readonly tiles: SplatMesh[] = [];
  readonly generated: GeneratedSoil;
  palette!: SoilPalette;
  private staticData: Float32Array;
  private dynData: Float32Array;
  private genTimer = 0;
  private genDirty = false;
  private rayTimer = 0;
  private rayDirty = false;
  /** splats loaded so far */
  splatCount = 0;
  onChange: (() => void) | null = null;

  constructor(data: SceneData, opts: { zoneRadius: number }) {
    this.data = data;
    const { N, cell, half } = data.meta;
    this.hf = new Heightfield({ N, cell, half, H0: data.H0, mask: data.M, zoneRadius: opts.zoneRadius });
    this.hf.maxDigDepth = data.machine.spec.maxDigDepth;
    this.ray = new HeightRaycaster(N, cell, half, Float32Array.from(data.H0));
    const s = new Float32Array(N * N);
    for (let k = 0; k < N * N; k++) s[k] = data.objectCode[k] === 2 ? Math.max(data.T[k], data.H0[k]) : data.H0[k];
    this.sensorRay = new HeightRaycaster(N, cell, half, s);

    this.staticData = new Float32Array(N * N * 4);
    for (let k = 0; k < N * N; k++) {
      this.staticData[k * 4] = data.H0[k];
      this.staticData[k * 4 + 1] = data.objectCode[k];
      const a = data.albedo;
      this.staticData[k * 4 + 2] = (0.299 * a[k * 3] + 0.587 * a[k * 3 + 1] + 0.114 * a[k * 3 + 2]) / 255;
    }
    this.tStatic = this.mkTex(this.staticData, N);
    this.dynData = new Float32Array(N * N * 4);
    for (let k = 0; k < N * N; k++) this.dynData[k * 4 + 2] = 1e4;
    this.tDyn = this.mkTex(this.dynData, N);

    this.uniforms = {
      grid: new THREE.Vector4(half, cell, N, 0),
      flags: new THREE.Vector4(0, 0, 0, 0),
      grade: new THREE.Vector4(0.03, data.meta.palette.flat.lumP50, 0.12, 0.45),
      cSub: new THREE.Vector3(),
      cSpoil: new THREE.Vector3(),
      grading: new THREE.Vector4(1, 1, 0, 1),
      tint: new THREE.Vector3(1, 1, 1),
    };
    this.setSoil(this.hf.soil);
    this.generated = new GeneratedSoil(400_000, makeGradeOnly(this.uniforms));
  }

  private mkTex(arr: Float32Array, N: number) {
    const t = new THREE.DataTexture(arr, N, N, THREE.RGBAFormat, THREE.FloatType);
    t.minFilter = THREE.NearestFilter;
    t.magFilter = THREE.NearestFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  }

  setSoil(soil: SoilType | string) {
    const s = typeof soil === 'string' ? soilById(soil) : soil;
    this.hf.soil = s;
    const p = this.data.meta.palette;
    const steep = p.steep ?? p.gentle;
    const pile = p.pile ?? p.flat;
    const sub: [number, number, number] = [steep.mean[0] * s.subsoilTint[0], steep.mean[1] * s.subsoilTint[1], steep.mean[2] * s.subsoilTint[2]];
    const spoil: [number, number, number] = [pile.mean[0] * s.spoilTint[0], pile.mean[1] * s.spoilTint[1], pile.mean[2] * s.spoilTint[2]];
    this.uniforms.cSub.set(...sub);
    this.uniforms.cSpoil.set(...spoil);
    this.palette = { sub, spoil, lumLo: steep.lumP10 / steep.lumP50, lumHi: steep.lumP90 / steep.lumP50 };
    this.genDirty = true;
    this.tiles.forEach((t) => t.updateVersion());
  }

  /** load captured splat tiles (inner first for a fast first frame) */
  async loadTiles(base: string, onProgress: (f: number, label: string) => void): Promise<void> {
    const files = [
      { url: `${base}/inner.sog`, label: 'dig zone (1.3M splats)', w: 0.68 },
      { url: `${base}/outer.sog`, label: 'surroundings (0.6M splats)', w: 0.32 },
    ];
    let done = 0;
    for (const f of files) {
      const mesh = new SplatMesh({
        url: f.url,
        worldModifier: makeTerrainDeformer(this.tStatic, this.tDyn, this.uniforms),
        raycastable: false,
        editable: false,
        onProgress: (e: ProgressEvent) => {
          if (e.lengthComputable) onProgress(done + f.w * (e.loaded / e.total) * 0.92, f.label);
        },
      });
      mesh.name = f.url;
      this.tiles.push(mesh);
      await mesh.initialized;
      this.splatCount += mesh.packedSplats?.numSplats ?? 0;
      done += f.w;
      onProgress(done, f.label);
      this.onTileReady?.(mesh);
    }
  }
  onTileReady: ((m: SplatMesh) => void) | null = null;

  /** write simulation state into the dynamic texture for a rect */
  private upload(r: Rect) {
    const hf = this.hf, N = hf.N, d = this.dynData;
    for (let j = r.j0; j <= r.j1; j++)
      for (let i = r.i0; i <= r.i1; i++) {
        const k = j * N + i;
        d[k * 4] = hf.B[k] + hf.L[k] - hf.H0[k];
        d[k * 4 + 1] = hf.L[k];
        const D = hf.design[k];
        d[k * 4 + 2] = Number.isNaN(D) ? 1e4 : D - hf.H0[k];
      }
    this.tDyn.needsUpdate = true;
    this.tiles.forEach((t) => t.updateVersion());
  }

  /** call when the design surface changes */
  designChanged() {
    this.upload({ i0: 0, j0: 0, i1: this.hf.N - 1, j1: this.hf.N - 1 });
  }

  refreshLook() {
    this.tiles.forEach((t) => t.updateVersion());
    this.generated.mesh.updateVersion();
  }

  setGeneratedOverlay(on: boolean) {
    this.uniforms.flags.x = on ? 1 : 0;
    this.generated.overlay = on;
    this.genDirty = true;
    this.tiles.forEach((t) => t.updateVersion());
  }

  update(dt: number) {
    const hf = this.hf;
    if (hf.isSettling()) hf.relax(3);
    const r = hf.consumeDirty();
    if (r) {
      this.upload(r);
      this.generated.include(r);
      this.genDirty = true;
      this.rayDirty = true;
      this.onChange?.();
    }
    this.genTimer -= dt;
    if (this.genDirty && this.genTimer <= 0) {
      this.generated.rebuild(hf, this.palette);
      this.genDirty = false;
      this.genTimer = 0.08;
    }
    this.rayTimer -= dt;
    if (this.rayDirty && this.rayTimer <= 0) {
      const N = hf.N;
      for (let k = 0; k < N * N; k++) this.ray.surf[k] = hf.B[k] + hf.L[k];
      this.ray.rebuild();
      const s = this.sensorRay.surf, code = this.data.objectCode, T = this.data.T;
      for (let k = 0; k < N * N; k++) s[k] = code[k] === 2 ? Math.max(T[k], hf.B[k] + hf.L[k]) : hf.B[k] + hf.L[k];
      this.sensorRay.rebuild();
      this.rayDirty = false;
      this.rayTimer = 0.1;
    }
  }

  /** in twin mode, cover the ground under the photographed machine with generated soil */
  setMachineFill(on: boolean) {
    const g = this.generated, code = this.data.objectCode, N = this.hf.N;
    if (on) {
      const m = new Uint8Array(N * N);
      let i0 = N, j0 = N, i1 = 0, j1 = 0;
      for (let k = 0; k < N * N; k++) if (code[k] === 1) { m[k] = 1; const i = k % N, j = (k / N) | 0; i0 = Math.min(i0, i); j0 = Math.min(j0, j); i1 = Math.max(i1, i); j1 = Math.max(j1, j); }
      g.fillMask = m;
      g.include({ i0, j0, i1, j1 });
    } else g.fillMask = null;
    this.genDirty = true;
  }

  reset() {
    this.hf.reset();
    this.generated.clear();
  }
}
