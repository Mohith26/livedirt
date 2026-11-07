import * as THREE from 'three';

/**
 * Time-of-day and weather. These are honest approximations: the captured splats have baked overcast
 * lighting, so "relighting" here is a colour grade on splat colours + sky + particles, not physically
 * based relighting. Lidar effects (rain/dust) are modelled separately in the sensor.
 */
export interface Look { exposure: number; saturation: number; warmth: number; contrast: number; tint: [number, number, number]; sky: [string, string, string] }

export const LOOKS: Record<string, Look & { label: string }> = {
  captured: { label: 'As captured (overcast)', exposure: 1, saturation: 1, warmth: 0, contrast: 1, tint: [1, 1, 1], sky: ['#cfd8de', '#eef1f2', '#d9d2c6'] },
  noon: { label: 'Bright noon', exposure: 1.1, saturation: 1.08, warmth: 0.15, contrast: 1.14, tint: [1.02, 1.0, 0.97], sky: ['#7fa7d6', '#d8e6f2', '#d6cdbf'] },
  golden: { label: 'Golden hour', exposure: 0.96, saturation: 1.12, warmth: 0.9, contrast: 1.07, tint: [1.08, 0.95, 0.8], sky: ['#8ea4c3', '#f4c08c', '#b88e6c'] },
  dusk: { label: 'Dusk', exposure: 0.55, saturation: 0.75, warmth: 0.2, contrast: 1.05, tint: [0.82, 0.86, 1.06], sky: ['#1a2232', '#5b586c', '#363131'] },
};

export class Environment {
  look = 'captured';
  rain = 0; // 0..1
  dust = 0; // 0..1
  private rainLines: THREE.LineSegments;
  private rainPos: Float32Array;
  private dustPts: THREE.Points;
  private dustPos: Float32Array;
  private dustVel: Float32Array;
  readonly group = new THREE.Group();
  dustCenter = new THREE.Vector3();

  constructor() {
    const NR = 3000;
    this.rainPos = new Float32Array(NR * 6);
    const rg = new THREE.BufferGeometry();
    rg.setAttribute('position', new THREE.BufferAttribute(this.rainPos, 3));
    this.rainLines = new THREE.LineSegments(rg, new THREE.LineBasicMaterial({ color: 0xc8d6e6, transparent: true, opacity: 0.35, depthWrite: false }));
    this.rainLines.frustumCulled = false;
    this.rainLines.visible = false;
    for (let i = 0; i < NR; i++) this.respawnRain(i, true, new THREE.Vector3());
    const ND = 2200;
    this.dustPos = new Float32Array(ND * 3);
    this.dustVel = new Float32Array(ND * 3);
    const dg = new THREE.BufferGeometry();
    dg.setAttribute('position', new THREE.BufferAttribute(this.dustPos, 3));
    const sprite = this.softSprite();
    this.dustPts = new THREE.Points(dg, new THREE.PointsMaterial({ map: sprite, size: 1.6, color: 0xb59a74, transparent: true, opacity: 0.0, depthWrite: false, sizeAttenuation: true }));
    this.dustPts.frustumCulled = false;
    this.dustPts.visible = false;
    for (let i = 0; i < ND; i++) this.respawnDust(i);
    this.group.add(this.rainLines, this.dustPts);
  }

  private softSprite() {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d')!;
    const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, 'rgba(255,255,255,0.55)'); grd.addColorStop(0.5, 'rgba(255,255,255,0.18)'); grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
    const t = new THREE.CanvasTexture(c); return t;
  }

  private respawnRain(i: number, init: boolean, cam: THREE.Vector3) {
    const x = cam.x + (Math.random() - 0.5) * 40, z = cam.z + (Math.random() - 0.5) * 40;
    const y = init ? cam.y + (Math.random() - 0.5) * 20 : cam.y + 12 + Math.random() * 4;
    const o = i * 6;
    this.rainPos[o] = x; this.rainPos[o + 1] = y; this.rainPos[o + 2] = z;
    this.rainPos[o + 3] = x + 0.03; this.rainPos[o + 4] = y - 0.45; this.rainPos[o + 5] = z + 0.02;
  }
  private respawnDust(i: number) {
    const a = Math.random() * Math.PI * 2, r = Math.sqrt(Math.random()) * 9;
    this.dustPos[i * 3] = this.dustCenter.x + Math.cos(a) * r;
    this.dustPos[i * 3 + 1] = this.dustCenter.y + Math.random() * 4.5;
    this.dustPos[i * 3 + 2] = this.dustCenter.z + Math.sin(a) * r;
    this.dustVel[i * 3] = 0.4 + Math.random() * 0.4; this.dustVel[i * 3 + 1] = (Math.random() - 0.3) * 0.15; this.dustVel[i * 3 + 2] = (Math.random() - 0.5) * 0.3;
  }

  /** grading uniforms for the splat shader */
  grading(): { look: THREE.Vector4; tint: THREE.Vector3 } {
    const L = LOOKS[this.look] ?? LOOKS.captured;
    let exposure = L.exposure, sat = L.saturation, contrast = L.contrast;
    const tint: [number, number, number] = [...L.tint];
    if (this.rain > 0) { exposure *= 1 - 0.18 * this.rain; sat *= 1 - 0.2 * this.rain; contrast *= 1 - 0.05 * this.rain; tint[2] *= 1 + 0.04 * this.rain; }
    if (this.dust > 0) { sat *= 1 - 0.3 * this.dust; contrast *= 1 - 0.18 * this.dust; tint[0] *= 1 + 0.06 * this.dust; tint[2] *= 1 - 0.08 * this.dust; exposure *= 1 + 0.05 * this.dust; }
    return { look: new THREE.Vector4(exposure, sat, L.warmth, contrast), tint: new THREE.Vector3(...tint) };
  }

  sky(): [string, string, string] {
    const L = LOOKS[this.look] ?? LOOKS.captured;
    if (this.rain > 0.3) return ['#7d858c', '#a7adb1', '#8b8580'];
    if (this.dust > 0.4) return ['#b9a68a', '#d7c6a8', '#a8916f'];
    return L.sky;
  }

  tick(dt: number, cam: THREE.Vector3) {
    this.rainLines.visible = this.rain > 0.01;
    if (this.rainLines.visible) {
      (this.rainLines.material as THREE.LineBasicMaterial).opacity = 0.15 + 0.35 * this.rain;
      const n = this.rainPos.length / 6;
      const active = Math.floor(n * this.rain);
      for (let i = 0; i < n; i++) {
        const o = i * 6;
        if (i >= active) { this.rainPos[o + 1] = -999; this.rainPos[o + 4] = -999; continue; }
        const v = 14 * dt;
        this.rainPos[o + 1] -= v; this.rainPos[o + 4] -= v;
        if (this.rainPos[o + 1] < cam.y - 10 || Math.abs(this.rainPos[o] - cam.x) > 22 || Math.abs(this.rainPos[o + 2] - cam.z) > 22) this.respawnRain(i, false, cam);
      }
      (this.rainLines.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    }
    this.dustPts.visible = this.dust > 0.01;
    if (this.dustPts.visible) {
      (this.dustPts.material as THREE.PointsMaterial).opacity = 0.55 * this.dust;
      const n = this.dustPos.length / 3;
      for (let i = 0; i < n; i++) {
        this.dustPos[i * 3] += this.dustVel[i * 3] * dt; this.dustPos[i * 3 + 1] += this.dustVel[i * 3 + 1] * dt; this.dustPos[i * 3 + 2] += this.dustVel[i * 3 + 2] * dt;
        const dx = this.dustPos[i * 3] - this.dustCenter.x, dz = this.dustPos[i * 3 + 2] - this.dustCenter.z;
        if (dx * dx + dz * dz > 110 || this.dustPos[i * 3 + 1] > this.dustCenter.y + 6) this.respawnDust(i);
      }
      (this.dustPts.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    }
  }
}
