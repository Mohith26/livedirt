import * as THREE from 'three';
import type { ScanResult } from './lidarClient';
import { PointLabel } from './lidarModel';

/** turbo-like colormap for range/height */
export function ramp(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)));
  return [r, g, b];
}

const LABEL_COLORS: Record<number, [number, number, number]> = {
  [PointLabel.Person]: [1, 0.15, 0.55],
  [PointLabel.Dust]: [0.85, 0.72, 0.45],
  [PointLabel.Rain]: [0.55, 0.75, 1],
};

/** in-scene point cloud of the latest scan */
export class LidarPoints {
  readonly points: THREE.Points;
  private geo = new THREE.BufferGeometry();
  private pos: Float32Array;
  private col: Float32Array;
  colorBy: 'height' | 'label' | 'intensity' = 'height';

  constructor(max = 70000) {
    this.pos = new Float32Array(max * 3);
    this.col = new Float32Array(max * 3);
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    this.geo.setDrawRange(0, 0);
    const mat = new THREE.PointsMaterial({ size: 0.045, vertexColors: true, sizeAttenuation: true, depthWrite: false, transparent: true, opacity: 0.95 });
    this.points = new THREE.Points(this.geo, mat);
    this.points.frustumCulled = false;
    this.points.renderOrder = 10;
  }

  update(s: ScanResult, groundY: number) {
    const n = Math.min(s.count, this.pos.length / 3);
    for (let i = 0; i < n; i++) {
      this.pos[i * 3] = s.points[i * 4];
      this.pos[i * 3 + 1] = s.points[i * 4 + 1];
      this.pos[i * 3 + 2] = s.points[i * 4 + 2];
      const lab = s.label[i];
      let c: [number, number, number];
      if (LABEL_COLORS[lab] && this.colorBy !== 'intensity') c = LABEL_COLORS[lab];
      else if (this.colorBy === 'intensity') { const v = s.points[i * 4 + 3]; c = [v, v, v]; }
      else c = ramp((s.points[i * 4 + 1] - groundY + 3) / 6);
      this.col[i * 3] = c[0]; this.col[i * 3 + 1] = c[1]; this.col[i * 3 + 2] = c[2];
    }
    this.geo.setDrawRange(0, n);
    (this.geo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.geo.attributes.color as THREE.BufferAttribute).needsUpdate = true;
  }
}

/** 2D panel: bird's-eye view (sensor-centred, heading up) and a 64-row range image */
export class LidarPanel {
  readonly bev: HTMLCanvasElement;
  readonly range: HTMLCanvasElement;
  private bctx: CanvasRenderingContext2D;
  private rctx: CanvasRenderingContext2D;
  private rangeImg: ImageData | null = null;
  bevRadius = 18;

  constructor(bev: HTMLCanvasElement, range: HTMLCanvasElement) {
    this.bev = bev;
    this.range = range;
    this.bctx = bev.getContext('2d')!;
    this.rctx = range.getContext('2d')!;
  }

  draw(s: ScanResult, heading: number, groundY: number) {
    const ctx = this.bctx, W = this.bev.width, H = this.bev.height;
    ctx.fillStyle = '#0b0a09';
    ctx.fillRect(0, 0, W, H);
    const sc = Math.min(W, H) / 2 / this.bevRadius;
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let r = 5; r <= this.bevRadius; r += 5) { ctx.beginPath(); ctx.arc(W / 2, H / 2, r * sc, 0, Math.PI * 2); ctx.stroke(); }
    const ch = Math.cos(-heading - Math.PI / 2), sh = Math.sin(-heading - Math.PI / 2);
    const img = ctx.getImageData(0, 0, W, H);
    const d = img.data;
    for (let i = 0; i < s.count; i++) {
      const dx = s.points[i * 4] - s.pose.x, dz = s.points[i * 4 + 2] - s.pose.z;
      const rx = dx * ch - dz * sh, rz = dx * sh + dz * ch;
      const px = Math.round(W / 2 + rx * sc), py = Math.round(H / 2 + rz * sc);
      if (px < 0 || py < 0 || px >= W || py >= H) continue;
      const lab = s.label[i];
      const c = LABEL_COLORS[lab] ?? ramp((s.points[i * 4 + 1] - groundY + 3) / 6);
      const o = (py * W + px) * 4;
      d[o] = c[0] * 255; d[o + 1] = c[1] * 255; d[o + 2] = c[2] * 255; d[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    // ego marker
    ctx.fillStyle = '#ff7a1a';
    ctx.beginPath(); ctx.moveTo(W / 2, H / 2 - 7); ctx.lineTo(W / 2 - 5, H / 2 + 5); ctx.lineTo(W / 2 + 5, H / 2 + 5); ctx.closePath(); ctx.fill();

    // range image: 64 rows x hRes columns, coloured by range
    const RW = s.hRes, RH = 64;
    if (!this.rangeImg || this.rangeImg.width !== RW) this.rangeImg = new ImageData(RW, RH);
    const rd = this.rangeImg.data;
    rd.fill(0);
    for (let i = 3; i < rd.length; i += 4) rd[i] = 255;
    for (let i = 0; i < s.count; i++) {
      const dx = s.points[i * 4] - s.pose.x, dy = s.points[i * 4 + 1] - s.pose.y, dz = s.points[i * 4 + 2] - s.pose.z;
      const r = Math.hypot(dx, dy, dz);
      const lab = s.label[i];
      const c = LABEL_COLORS[lab] ?? ramp(1 - Math.min(1, r / 40));
      const o = (s.ring[i] * RW + s.col[i]) * 4;
      rd[o] = c[0] * 255; rd[o + 1] = c[1] * 255; rd[o + 2] = c[2] * 255;
    }
    // draw scaled
    const tmp = document.createElement('canvas');
    tmp.width = RW; tmp.height = RH;
    tmp.getContext('2d')!.putImageData(this.rangeImg, 0, 0);
    this.rctx.imageSmoothingEnabled = false;
    this.rctx.drawImage(tmp, 0, 0, this.range.width, this.range.height);
  }
}
