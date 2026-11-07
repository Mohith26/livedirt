import * as THREE from 'three';
import { PackedSplats, SplatMesh, utils, type GsplatModifier } from '@sparkjsdev/spark';
import type { Heightfield } from '../sim/heightfield';
import { rng } from '../sensors/lidarModel';

/**
 * Procedural splats for surfaces the camera never saw: trench walls, pile flanks, and loose spoil clumps.
 *
 * The capture only contains the original surface. When the heightfield steepens, terrain-bound splats move
 * vertically and leave the added surface area uncovered. For every modified cell we add splats in
 * proportion to the extra area (sqrt(1 + |grad H|^2) - 1), oriented to the local normal, coloured from the
 * palette learned from the scene's own cut faces (luminance distribution preserved). Loose deposits also get
 * clumps. Generation is deterministic per cell so regeneration while digging does not flicker.
 *
 * This is explicitly NOT learned: it is the stand-in for the generative (diffusion) fill described in the
 * project writeup, and it is tinted purple when the "generated" overlay is enabled.
 */
export interface SoilPalette {
  sub: [number, number, number];
  spoil: [number, number, number];
  lumLo: number; // p10 / p50
  lumHi: number; // p90 / p50
}

export class GeneratedSoil {
  readonly packed: PackedSplats;
  readonly mesh: SplatMesh;
  count = 0;
  readonly capacity: number;
  overlay = false;
  densityWall = 650; // splats / m^2 of added area
  densitySpoil = 160; // clumps / m^2 of loose surface
  private region: { i0: number; j0: number; i1: number; j1: number } | null = null;
  /** cells to cover even if unchanged (ground hidden under the photographed machine, shown in twin mode) */
  fillMask: Uint8Array | null = null;

  constructor(capacity = 400_000, modifier?: GsplatModifier) {
    this.capacity = capacity;
    this.packed = new PackedSplats({ maxSplats: capacity });
    this.packed.numSplats = 0;
    this.mesh = new SplatMesh({ packedSplats: this.packed, raycastable: false, editable: false, worldModifier: modifier });
    this.mesh.name = 'generated-soil';
  }

  /** grow the region of interest by a dirty rect from the simulation */
  include(r: { i0: number; j0: number; i1: number; j1: number } | null) {
    if (!r) return;
    const m = 2;
    const g = { i0: r.i0 - m, j0: r.j0 - m, i1: r.i1 + m, j1: r.j1 + m };
    this.region = this.region
      ? { i0: Math.min(this.region.i0, g.i0), j0: Math.min(this.region.j0, g.j0), i1: Math.max(this.region.i1, g.i1), j1: Math.max(this.region.j1, g.j1) }
      : g;
  }

  clear() {
    this.region = null;
    this.count = 0;
    this.packed.numSplats = 0;
    this.packed.needsUpdate = true;
    this.mesh.updateVersion();
  }

  /**
   * Material model per modified cell:
   *   base   - dark crevice fill (continuous coverage)
   *   clods  - lit top splat + offset shadow splat pairs; size/contrast depend on surface type
   *            (loose spoil: big clumps; cut floor: crumbs; steep cut wall: vertical bucket-tooth streaks)
   * Lighting is an overcast-sky approximation: sky visibility from the normal and from depth below the rim.
   */
  rebuild(hf: Heightfield, pal: SoilPalette) {
    if (!this.region) return;
    const N = hf.N, cell = hf.cell;
    const r = this.region;
    const i0 = Math.max(3, r.i0), j0 = Math.max(3, r.j0), i1 = Math.min(N - 4, r.i1), j1 = Math.min(N - 4, r.j1);
    const arr = this.packed.ensureSplats(this.capacity);
    let n = 0;
    const cap = this.capacity;
    const purple: [number, number, number] = [0.66, 0.32, 1.0];
    const H = (k: number) => hf.B[k] + hf.L[k];
    const overlay = this.overlay;
    const put = (x: number, y: number, z: number, sx: number, sy: number, sz: number, q: number[], a: number, c: [number, number, number]) => {
      if (n >= cap) return;
      let cr = c[0], cg = c[1], cb = c[2];
      if (overlay) { cr = cr * 0.2 + purple[0] * 0.8; cg = cg * 0.2 + purple[1] * 0.8; cb = cb * 0.2 + purple[2] * 0.8; }
      utils.setPackedSplat(arr, n++, x, y, z, sx, sy, sz, q[0], q[1], q[2], q[3], a, Math.min(1, Math.max(0, cr)), Math.min(1, Math.max(0, cg)), Math.min(1, Math.max(0, cb)));
    };
    const q = [0, 0, 0, 1];
    // quaternion with local Z -> n and local X -> t (tangent)
    const frame = (nx: number, ny: number, nz: number, tx: number, ty: number, tz: number) => {
      // b = n x t
      const bx = ny * tz - nz * ty, by = nz * tx - nx * tz, bz = nx * ty - ny * tx;
      // rotation matrix columns: t, b, n
      const m00 = tx, m01 = bx, m02 = nx, m10 = ty, m11 = by, m12 = ny, m20 = tz, m21 = bz, m22 = nz;
      const tr = m00 + m11 + m22;
      if (tr > 0) { const S = Math.sqrt(tr + 1) * 2; q[3] = 0.25 * S; q[0] = (m21 - m12) / S; q[1] = (m02 - m20) / S; q[2] = (m10 - m01) / S; }
      else if (m00 > m11 && m00 > m22) { const S = Math.sqrt(1 + m00 - m11 - m22) * 2; q[3] = (m21 - m12) / S; q[0] = 0.25 * S; q[1] = (m01 + m10) / S; q[2] = (m02 + m20) / S; }
      else if (m11 > m22) { const S = Math.sqrt(1 + m11 - m00 - m22) * 2; q[3] = (m02 - m20) / S; q[0] = (m01 + m10) / S; q[1] = 0.25 * S; q[2] = (m12 + m21) / S; }
      else { const S = Math.sqrt(1 + m22 - m00 - m11) * 2; q[3] = (m10 - m01) / S; q[0] = (m02 + m20) / S; q[1] = (m12 + m21) / S; q[2] = 0.25 * S; }
      return q;
    };
    for (let j = j0; j <= j1 && n < cap; j++)
      for (let i = i0; i <= i1 && n < cap; i++) {
        const k = j * N + i;
        const fill = this.fillMask ? this.fillMask[k] : 0;
        if (hf.mask[k] && !fill) continue;
        const dH = H(k) - hf.H0[k];
        const loose = hf.L[k];
        const changed = fill ? 1 : Math.max(Math.abs(dH), loose);
        if (changed < 0.03) continue;
        const gx = (H(k + 1) - H(k - 1)) / (2 * cell);
        const gz = (H(k + N) - H(k - N)) / (2 * cell);
        const g2 = gx * gx + gz * gz;
        const areaF = Math.sqrt(1 + g2);
        const slopeDeg = (Math.atan(Math.sqrt(g2)) * 180) / Math.PI;
        const cover = Math.min(1, (changed - 0.03) / 0.1);
        let rim = -Infinity;
        for (let dj = -3; dj <= 3; dj += 3) for (let di = -3; di <= 3; di += 3) rim = Math.max(rim, H(k + dj * N + di));
        const below = Math.max(0, rim - H(k));
        const occl = 1 - 0.42 * Math.min(1, below / 1.4);
        const isLoose = loose > 0.02;
        const isWall = !isLoose && slopeDeg > 42;
        const R = rng(k * 2654435761 + 1013904223);
        const base = isLoose ? pal.spoil : pal.sub;
        const moist = isLoose ? 1 : 1 - 0.15 * Math.min(1, Math.max(0, -dH) / 2);
        const A = cell * cell * areaF * cover;
        const nBase = Math.floor(A * 260 + R());
        const nClod = Math.floor(A * (isLoose ? 240 : isWall ? 300 : 200) + R());
        const nCrumb = Math.floor(A * (isLoose ? 950 : isWall ? 260 : 650) + R());
        // bucket-spoil chunks: wet clay leaves the bucket in big clods (10-25 cm)
        const nChunk = isLoose ? Math.floor(A * 28 * Math.min(1, loose / 0.15) + R()) : 0;
        const [cx, cz] = hf.center(i, j);
        // fall-line tangent (direction of steepest descent projected on the surface) for wall streaks
        const gl = Math.sqrt(g2) || 1;
        for (let s = 0; s < nBase + nClod + nCrumb + nChunk && n < cap; s++) {
          const px = cx + (R() - 0.5) * cell, pz = cz + (R() - 0.5) * cell;
          const py = hf.heightAt(px, pz);
          const e = cell * 0.5;
          const sx = (hf.heightAt(px + e, pz) - hf.heightAt(px - e, pz)) / (2 * e);
          const sz = (hf.heightAt(px, pz + e) - hf.heightAt(px, pz - e)) / (2 * e);
          let nx = -sx, ny = 1, nz = -sz;
          const nl = Math.hypot(nx, ny, nz); nx /= nl; ny /= nl; nz /= nl;
          // tangent: along fall line for walls (streaks run downhill), random otherwise
          let tx: number, ty: number, tz: number;
          if (isWall) { tx = -gx / gl; tz = -gz / gl; ty = -gl; } else { const a = R() * Math.PI * 2; tx = Math.cos(a); ty = 0; tz = Math.sin(a); }
          const dp = tx * nx + ty * ny + tz * nz; tx -= dp * nx; ty -= dp * ny; tz -= dp * nz;
          const tl = Math.hypot(tx, ty, tz) || 1; tx /= tl; ty /= tl; tz /= tl;
          const sky = (0.7 + 0.3 * ny) * occl;
          const hue = (R() - 0.5) * 0.05;
          const col = (m: number): [number, number, number] => [base[0] * m * (1 + hue), base[1] * m, base[2] * m * (1 - hue)];
          const qq = frame(nx, ny, nz, tx, ty, tz);
          if (s < nBase) {
            const rad = 0.045 + 0.03 * R();
            put(px, py, pz, rad, rad * (0.7 + 0.3 * R()), rad * 0.1, qq, 0.9, col(0.6 * sky * moist * (0.85 + 0.3 * R())));
            continue;
          }
          if (s >= nBase + nClod + nCrumb) {
            // chunk: shadowed underside + lit, slightly rounded top
            const size = 0.07 + 0.13 * Math.pow(R(), 1.5);
            const lumTop = 1.0 + (pal.lumHi - 1) * 0.7 * R();
            put(px - nx * 0.01, py - ny * 0.01 + 0.005, pz - nz * 0.01, size * 1.2, size * 1.05, size * 0.35, qq, 0.9, col(0.42 * sky));
            put(px + nx * size * 0.4, py + ny * size * 0.4, pz + nz * size * 0.4, size, size * (0.75 + 0.25 * R()), size * 0.55, qq, 0.96, col(lumTop * sky));
            continue;
          }
          if (s >= nBase + nClod) {
            // crumbs: tiny, high-contrast grains drawn from the full luminance spread of the captured soil
            const u = R();
            const lf = u < 0.5 ? pal.lumLo + (1 - pal.lumLo) * (u * 2) : 1 + (pal.lumHi - 1) * (u - 0.5) * 2;
            const rad = 0.009 + 0.02 * R();
            put(px + nx * rad * 0.5, py + ny * rad * 0.5, pz + nz * rad * 0.5, rad, rad * (0.6 + 0.4 * R()), rad * 0.5, qq, 0.95, col(lf * sky * moist));
            continue;
          }
          // clod pair: shadow then lit top
          let size: number, elong: number, contrast: number;
          if (isLoose) { size = 0.03 + 0.06 * Math.pow(R(), 1.8); elong = 0.75 + 0.3 * R(); contrast = 1.0; }
          else if (isWall) { size = 0.02 + 0.03 * R(); elong = 2.2 + 1.8 * R(); contrast = 0.55; }
          else { size = 0.018 + 0.035 * Math.pow(R(), 1.4); elong = 0.8 + 0.4 * R(); contrast = 0.8; }
          const lumTop = 1.05 + (pal.lumHi - 1) * 0.8 * contrast * R();
          const lumShadow = 0.45 + 0.2 * R();
          // shadow: slightly larger, offset "down-light" along the surface (overcast => mostly toward the fall line)
          const off = size * 0.35;
          const dx = isWall ? tx : -0.5 * tx, dz = isWall ? tz : -0.5 * tz, dy = isWall ? ty : -0.5 * ty;
          put(px + dx * off + nx * 0.002, py + dy * off + ny * 0.002, pz + dz * off + nz * 0.002, size * 1.15 * elong, size * 1.1, size * 0.25, qq, 0.85, col(lumShadow * sky * moist));
          const lift = size * (isLoose ? 0.45 : 0.25);
          put(px + nx * lift, py + ny * lift, pz + nz * lift, size * elong, size * (0.8 + 0.25 * R()), size * 0.4, qq, 0.95, col(lumTop * sky * moist));
        }
      }
    this.count = n;
    this.packed.numSplats = n;
    this.packed.needsUpdate = true;
    this.mesh.updateVersion();
  }
}

export const toVec3 = (c: [number, number, number]) => new THREE.Vector3(c[0], c[1], c[2]);
