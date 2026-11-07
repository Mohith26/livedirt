import * as THREE from 'three';
import type { Heightfield, DesignTrench } from '../sim/heightfield';

/** terrain-conforming helper geometry: brush ring, design outline, spoil marker, swing-zone ring */
export class Overlays {
  readonly group = new THREE.Group();
  private brush: THREE.LineLoop;
  private design: THREE.LineLoop;
  private designCenter: THREE.Line;
  private zone: THREE.LineLoop;
  readonly spoil: THREE.Group;
  private hf: Heightfield;

  constructor(hf: Heightfield) {
    this.hf = hf;
    const mk = (color: number, opacity: number, n: number) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
      const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: false, depthWrite: false });
      const l = new THREE.LineLoop(g, m);
      l.frustumCulled = false; l.renderOrder = 20; l.visible = false;
      return l;
    };
    this.brush = mk(0xffffff, 0.9, 64);
    this.design = mk(0x3ddc84, 0.95, 200);
    this.zone = mk(0xff4d4d, 0.45, 96);
    const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(2 * 3), 3));
    this.designCenter = new THREE.Line(cg, new THREE.LineDashedMaterial({ color: 0x3ddc84, dashSize: 0.3, gapSize: 0.2, depthTest: false, transparent: true, opacity: 0.8 }));
    this.designCenter.renderOrder = 20; this.designCenter.visible = false; this.designCenter.frustumCulled = false;
    this.spoil = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 1.4, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    pole.position.y = 0.7;
    const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.45, 0.28), new THREE.MeshBasicMaterial({ color: 0xff7a1a, side: THREE.DoubleSide }));
    flag.position.set(0.23, 1.25, 0);
    this.spoil.add(pole, flag);
    this.spoil.visible = false;
    this.group.add(this.brush, this.design, this.designCenter, this.zone, this.spoil);
  }

  private ring(line: THREE.LineLoop, cx: number, cz: number, r: number, lift = 0.04) {
    const pos = line.geometry.attributes.position as THREE.BufferAttribute;
    const n = pos.count;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const x = cx + Math.cos(a) * r, z = cz + Math.sin(a) * r;
      pos.setXYZ(i, x, this.hf.heightAt(x, z) + lift, z);
    }
    pos.needsUpdate = true;
    line.visible = true;
  }

  showBrush(x: number, z: number, r: number) { this.ring(this.brush, x, z, r); }
  hideBrush() { this.brush.visible = false; }
  showZone(x: number, z: number, r: number) { this.ring(this.zone, x, z, r, 0.06); }
  hideZone() { this.zone.visible = false; }

  setDesign(spec: DesignTrench | null) {
    if (!spec) { this.design.visible = false; this.designCenter.visible = false; return; }
    const [x0, z0] = spec.p0, [x1, z1] = spec.p1;
    const len = Math.hypot(x1 - x0, z1 - z0) || 1;
    const ux = (x1 - x0) / len, uz = (z1 - z0) / len, vx = -uz, vz = ux;
    const topHalf = spec.bottomWidth / 2 + spec.depth * spec.sideSlope;
    const ext = spec.depth * spec.sideSlope;
    const corners: [number, number][] = [
      [x0 - ux * ext + vx * topHalf, z0 - uz * ext + vz * topHalf],
      [x1 + ux * ext + vx * topHalf, z1 + uz * ext + vz * topHalf],
      [x1 + ux * ext - vx * topHalf, z1 + uz * ext - vz * topHalf],
      [x0 - ux * ext - vx * topHalf, z0 - uz * ext - vz * topHalf],
    ];
    const pos = this.design.geometry.attributes.position as THREE.BufferAttribute;
    const n = pos.count, per = n / 4;
    for (let e = 0; e < 4; e++) {
      const a = corners[e], b = corners[(e + 1) % 4];
      for (let s = 0; s < per; s++) {
        const t = s / per;
        const x = a[0] + (b[0] - a[0]) * t, z = a[1] + (b[1] - a[1]) * t;
        pos.setXYZ(e * per + s, x, this.hf.heightAt(x, z, 'H0') + 0.05, z);
      }
    }
    pos.needsUpdate = true;
    this.design.visible = true;
    const cp = this.designCenter.geometry.attributes.position as THREE.BufferAttribute;
    cp.setXYZ(0, x0, this.hf.heightAt(x0, z0, 'H0') + 0.05, z0);
    cp.setXYZ(1, x1, this.hf.heightAt(x1, z1, 'H0') + 0.05, z1);
    cp.needsUpdate = true;
    this.designCenter.computeLineDistances();
    this.designCenter.visible = true;
  }

  setSpoil(p: THREE.Vector3 | null) {
    if (!p) { this.spoil.visible = false; return; }
    this.spoil.position.set(p.x, this.hf.heightAt(p.x, p.z), p.z);
    this.spoil.visible = true;
  }
}
