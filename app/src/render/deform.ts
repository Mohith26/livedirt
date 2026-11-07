import * as THREE from 'three';
import { dyno, type GsplatModifier } from '@sparkjsdev/spark';

/**
 * GPU splat deformation driven by the earthwork heightfield.
 *
 * Every splat is evaluated against two float textures on the same grid as the simulation:
 *   tStatic (RGBA32F): r = original terrain H0, g = object code (0 ground, 1 our machine, 2 other object)
 *   tDyn    (RGBA32F): r = dH = H - H0, g = loose thickness, b = design grade relative to H0 (1e4 = none),
 *                      a = reserved
 * Splats near the original surface (and not on objects) are "terrain-bound": they translate vertically by dH
 * and are re-tinted toward the learned subsoil palette where cut, or toward loose spoil where material
 * was deposited. Per-splat brightness variation is preserved so texture survives the re-tint.
 */
export interface DeformUniforms {
  grid: THREE.Vector4; // half, cell, N, time
  flags: THREE.Vector4; // generatedOverlay, gradeOverlay, hideOwnMachine, wetness
  grade: THREE.Vector4; // tol, surfaceLum, bindLow, bindHigh
  cSub: THREE.Vector3;
  cSpoil: THREE.Vector3;
  grading: THREE.Vector4; // exposure, saturation, warmth, contrast (global look)
  tint: THREE.Vector3;
}

const GLSL_HELPERS = dyno.unindent(`
  vec4 ldFetch(sampler2D t, ivec2 c) {
    ivec2 sz = textureSize(t, 0);
    c = clamp(c, ivec2(0), sz - ivec2(1));
    return texelFetch(t, c, 0);
  }
  vec4 ldBilinear(sampler2D t, vec2 xz, vec4 grid) {
    vec2 f = (xz + grid.x) / grid.y - 0.5;
    vec2 i0 = floor(f);
    vec2 w = f - i0;
    ivec2 c = ivec2(i0);
    vec4 a = ldFetch(t, c);
    vec4 b = ldFetch(t, c + ivec2(1, 0));
    vec4 cc = ldFetch(t, c + ivec2(0, 1));
    vec4 d = ldFetch(t, c + ivec2(1, 1));
    return mix(mix(a, b, w.x), mix(cc, d, w.x), w.y);
  }
  float ldHash(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
  vec3 ldGrade(vec3 c, vec4 g, vec3 tint) {
    c *= g.x;
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(l), c, g.y);
    c = (c - 0.5) * g.w + 0.5;
    c *= mix(vec3(1.0), vec3(1.06, 1.0, 0.9), g.z);
    return clamp(c * tint, 0.0, 1.0);
  }
`);

export function makeTerrainDeformer(tStatic: THREE.DataTexture, tDyn: THREE.DataTexture, u: DeformUniforms): GsplatModifier {
  const uStatic = dyno.dynoSampler2D(tStatic, 'ldStatic');
  const uDyn = dyno.dynoSampler2D(tDyn, 'ldDyn');
  const uGrid = dyno.dynoVec4(u.grid, 'ldGrid');
  const uFlags = dyno.dynoVec4(u.flags, 'ldFlags');
  const uGrade = dyno.dynoVec4(u.grade, 'ldGradeP');
  const uSub = dyno.dynoVec3(u.cSub, 'ldSub');
  const uSpoil = dyno.dynoVec3(u.cSpoil, 'ldSpoil');
  const uLook = dyno.dynoVec4(u.grading, 'ldLook');
  const uTint = dyno.dynoVec3(u.tint, 'ldTint');

  return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
    if (!gsplat) throw new Error('deformer: missing gsplat input');
    const block = dyno.dyno({
      inTypes: { gsplat: dyno.Gsplat, tS: 'sampler2D', tD: 'sampler2D', grid: 'vec4', flags: 'vec4', gp: 'vec4', sub: 'vec3', spoil: 'vec3', look: 'vec4', tint: 'vec3' },
      outTypes: { gsplat: dyno.Gsplat },
      inputs: { gsplat, tS: uStatic, tD: uDyn, grid: uGrid, flags: uFlags, gp: uGrade, sub: uSub, spoil: uSpoil, look: uLook, tint: uTint },
      globals: () => [GLSL_HELPERS],
      statements: ({ inputs, outputs }) => {
        const I = inputs as Record<string, string>;
        const O = outputs.gsplat!;
        return dyno.unindentLines(`
          ${O} = ${I.gsplat};
          {
            vec3 c = ${O}.center;
            vec4 st = ldBilinear(${I.tS}, c.xz, ${I.grid});
            float code = ldFetch(${I.tS}, ivec2(floor((c.xz + ${I.grid}.x) / ${I.grid}.y))).g;
            vec4 dh = ldBilinear(${I.tD}, c.xz, ${I.grid});
            float rel = c.y - st.r;
            float onObject = step(0.5, code);
            float bind = (1.0 - smoothstep(${I.gp}.z, ${I.gp}.w, rel)) * (1.0 - onObject);
            if (${I.flags}.z > 0.5 && code > 0.5 && code < 1.5 && rel > 0.02) {
              ${O}.rgba.a = 0.0;
            }
            ${O}.center.y += bind * dh.r;
            vec3 col = ${O}.rgba.rgb;
            float lum = dot(col, vec3(0.299, 0.587, 0.114));
            float rl = clamp(lum / max(${I.gp}.y, 1e-3), 0.25, 2.2);
            float h = ldHash(float(${O}.index));
            float dug = bind * smoothstep(0.03, 0.4, -dh.r);
            float loose = bind * smoothstep(0.005, 0.12, dh.g);
            vec3 subC = ${I.sub} * rl * (0.92 + 0.16 * h);
            vec3 spC = ${I.spoil} * rl * (0.85 + 0.3 * h);
            col = mix(col, subC, dug * 0.88);
            col = mix(col, spC, loose * 0.8);
            // hand-off: where the ground has been changed beyond a few cm, the captured surface fades out and
            // the generated soil layer (rebuilt on the CPU from the same heightfield) takes over
            float changed = max(-dh.r, max(dh.r, dh.g));
            ${O}.rgba.a *= 1.0 - bind * smoothstep(0.05, 0.16, changed);
            col *= 1.0 - ${I.flags}.w * 0.38 * (1.0 - onObject * 0.6);
            if (${I.flags}.x > 0.5) col = mix(col, vec3(0.66, 0.32, 1.0), max(dug, loose) * 0.85);
            if (${I.flags}.y > 0.5 && dh.b < 1000.0 && bind > 0.5) {
              float e = dh.r - dh.b;
              float tol = ${I.gp}.x;
              vec3 gc = e > tol ? mix(vec3(0.25, 0.6, 1.0), vec3(0.08, 0.2, 0.95), clamp((e - tol) / 0.6, 0.0, 1.0))
                       : (e < -tol ? vec3(1.0, 0.22, 0.16) : vec3(0.2, 0.95, 0.45));
              col = mix(col, gc, 0.6);
            }
            ${O}.rgba.rgb = ldGrade(col, ${I.look}, ${I.tint});
          }
        `);
      },
    });
    return { gsplat: block.outputs.gsplat };
  });
}

/** global grade only (for meshes that should not deform, e.g. the generated soil layer handles its own) */
export function makeGradeOnly(u: DeformUniforms): GsplatModifier {
  const uLook = dyno.dynoVec4(u.grading, 'ldLook');
  const uTint = dyno.dynoVec3(u.tint, 'ldTint');
  return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
    if (!gsplat) throw new Error('grade: missing gsplat input');
    const block = dyno.dyno({
      inTypes: { gsplat: dyno.Gsplat, look: 'vec4', tint: 'vec3' },
      outTypes: { gsplat: dyno.Gsplat },
      inputs: { gsplat, look: uLook, tint: uTint },
      globals: () => [GLSL_HELPERS],
      statements: ({ inputs, outputs }) => {
        const I = inputs as Record<string, string>;
        const O = outputs.gsplat!;
        return [`${O} = ${I.gsplat};`, `${O}.rgba.rgb = ldGrade(${O}.rgba.rgb, ${I.look}, ${I.tint});`];
      },
    });
    return { gsplat: block.outputs.gsplat };
  });
}
