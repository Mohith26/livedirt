import { McapWriter, type IWritable } from '@mcap/core';
import type { ScanResult } from '../sensors/lidarClient';
import type { SceneData } from '../data/scene';

/**
 * Records a LiveDirt session to MCAP using Foxglove's well-known schemas (JSON encoding):
 *   /lidar/points    foxglove.PointCloud       (x, y, z, intensity, label, ring) per scan
 *   /camera/roof     foxglove.CompressedImage  (JPEG) at ~5 Hz
 *   /tf              foxglove.FrameTransforms  world -> base_link -> lidar / camera
 *   /terrain/grid    foxglove.Grid             elevation of the dig zone at 1 Hz
 *   /machine/joints  livedirt.JointState       swing/boom/arm/bucket + autonomy phase
 *   /earthwork       livedirt.Earthwork        volume ledger + grade stats
 * Coordinates are converted from three.js (Y-up) to ROS/Foxglove (Z-up): X = x, Y = -z, Z = y.
 */
class MemWritable implements IWritable {
  chunks: Uint8Array[] = [];
  size = 0n;
  async write(b: Uint8Array) { this.chunks.push(b.slice()); this.size += BigInt(b.byteLength); }
  position() { return this.size; }
  bytes() { const out = new Uint8Array(Number(this.size)); let o = 0; for (const c of this.chunks) { out.set(c, o); o += c.byteLength; } return out; }
}

const enc = new TextEncoder();
const b64 = (u8: Uint8Array) => { let s = ''; const CH = 0x8000; for (let i = 0; i < u8.length; i += CH) s += String.fromCharCode(...u8.subarray(i, i + CH)); return btoa(s); };
const ts = (t: number) => ({ sec: Math.floor(t), nsec: Math.floor((t % 1) * 1e9) });
const toRos = (x: number, y: number, z: number) => ({ x, y: -z, z: y });
// rotation about the up axis by yaw (three convention atan2(z,x)) expressed in ROS frame: yaw_ros = -yaw ... see note below
const quatYawRos = (yaw: number) => {
  // a three.js direction (cos y, 0, sin y) maps to ROS (cos y, -sin y, 0) => ROS yaw = -y
  const h = -yaw / 2;
  return { x: 0, y: 0, z: Math.sin(h), w: Math.cos(h) };
};

const SCHEMAS: Record<string, object> = {
  'foxglove.PointCloud': { type: 'object', properties: { timestamp: { type: 'object' }, frame_id: { type: 'string' }, pose: { type: 'object' }, point_stride: { type: 'integer' }, fields: { type: 'array' }, data: { type: 'string', contentEncoding: 'base64' } } },
  'foxglove.CompressedImage': { type: 'object', properties: { timestamp: { type: 'object' }, frame_id: { type: 'string' }, data: { type: 'string', contentEncoding: 'base64' }, format: { type: 'string' } } },
  'foxglove.FrameTransforms': { type: 'object', properties: { transforms: { type: 'array' } } },
  'foxglove.Grid': { type: 'object', properties: { timestamp: { type: 'object' }, frame_id: { type: 'string' }, pose: { type: 'object' }, column_count: { type: 'integer' }, cell_size: { type: 'object' }, row_stride: { type: 'integer' }, cell_stride: { type: 'integer' }, fields: { type: 'array' }, data: { type: 'string', contentEncoding: 'base64' } } },
  'livedirt.JointState': { type: 'object', properties: { swing: { type: 'number' }, boom: { type: 'number' }, arm: { type: 'number' }, bucket: { type: 'number' }, phase: { type: 'string' } } },
  'livedirt.Earthwork': { type: 'object', properties: { bankCut: { type: 'number' }, sloughed: { type: 'number' }, dumped: { type: 'number' }, onGradePct: { type: 'number' }, remainingCut: { type: 'number' }, overDig: { type: 'number' } } },
};

export interface RecorderHost {
  twin: { joints: { swing: number; boom: number; arm: number; bucket: number } };
  sensorPose(which: 'lidar' | 'camera'): [number, number, number];
  autonomy: { phase: string };
  terrain: { hf: { N: number; cell: number; half: number; B: Float32Array; L: Float32Array; ledger: { bankCut: number; sloughed: number; dumped: number }; designSpec: unknown; gradeStats(t?: number): { onGradePct: number; remainingCut: number; overDig: number } } };
  machine: { cx: number; cz: number; groundY: number };
  cab: { readPixels(): Promise<Uint8Array>; width: number; height: number };
}

export class McapRecorder {
  recording = false;
  private w: MemWritable | null = null;
  private writer: McapWriter | null = null;
  private ch: Record<string, number> = {};
  private seq = 0;
  private chain: Promise<unknown> = Promise.resolve();
  private t0 = 0;
  private lastCam = -1;
  private lastGrid = -1;
  private lastTf = -1;
  private counts: Record<string, number> = {};
  private camBusy = false;
  private host: RecorderHost | null = null;
  private wallStart = 0;

  start(data: SceneData) {
    this.w = new MemWritable();
    this.writer = new McapWriter({ writable: this.w, useStatistics: true, useChunks: true, useMessageIndex: true, useSummaryOffsets: true, useChunkIndex: true });
    this.recording = true;
    this.counts = {};
    this.seq = 0;
    this.t0 = -1;
    this.lastCam = this.lastGrid = this.lastTf = -1;
    this.wallStart = Date.now() / 1000;
    const topics: [string, string][] = [
      ['/lidar/roof/points', 'foxglove.PointCloud'], ['/lidar/front/points', 'foxglove.PointCloud'], ['/camera/roof', 'foxglove.CompressedImage'], ['/tf', 'foxglove.FrameTransforms'],
      ['/terrain/grid', 'foxglove.Grid'], ['/machine/joints', 'livedirt.JointState'], ['/earthwork', 'livedirt.Earthwork'],
    ];
    this.chain = this.chain.then(async () => {
      await this.writer!.start({ profile: '', library: 'livedirt' });
      await this.writer!.addMetadata({ name: 'livedirt', metadata: new Map([['scene', data.meta.source.title], ['license', data.meta.source.license], ['metricScale', String(data.meta.metricScale)], ['frame_convention', 'ROS Z-up (X=x, Y=-z, Z=y from three.js)']]) });
      for (const [topic, schema] of topics) {
        const sid = await this.writer!.registerSchema({ name: schema, encoding: 'jsonschema', data: enc.encode(JSON.stringify(SCHEMAS[schema])) });
        this.ch[topic] = await this.writer!.registerChannel({ topic, schemaId: sid, messageEncoding: 'json', metadata: new Map() });
      }
    });
  }

  private put(topic: string, t: number, msg: object) {
    if (!this.writer) return;
    const channelId = this.ch[topic];
    const time = BigInt(Math.round((this.wallStart + t) * 1e9));
    const data = enc.encode(JSON.stringify(msg));
    this.counts[topic] = (this.counts[topic] ?? 0) + 1;
    this.chain = this.chain.then(() => this.writer!.addMessage({ channelId: this.ch[topic] ?? channelId, sequence: this.seq++, logTime: time, publishTime: time, data }));
  }

  private rel(t: number) { if (this.t0 < 0) this.t0 = t; return t - this.t0; }

  private scanTick: Record<string, number> = {};
  addScan(s: ScanResult, src: 'roof' | 'front' = 'roof') {
    if (!this.recording) return;
    // record each lidar at 5 Hz (half the sensor rate) to keep files a manageable size
    this.scanTick[src] = (this.scanTick[src] ?? 0) + 1;
    if (this.scanTick[src] % 2 === 0) return;
    const t = this.rel(s.t);
    // point layout: x y z intensity (float32) label ring (uint8) + 2 pad = 20 bytes
    const stride = 20;
    const buf = new ArrayBuffer(s.count * stride);
    const dv = new DataView(buf);
    for (let i = 0; i < s.count; i++) {
      const o = i * stride;
      const p = toRos(s.points[i * 4], s.points[i * 4 + 1], s.points[i * 4 + 2]);
      dv.setFloat32(o, p.x, true); dv.setFloat32(o + 4, p.y, true); dv.setFloat32(o + 8, p.z, true); dv.setFloat32(o + 12, s.points[i * 4 + 3], true);
      dv.setUint8(o + 16, s.label[i]); dv.setUint8(o + 17, s.ring[i]);
    }
    this.put(`/lidar/${src}/points`, t, {
      timestamp: ts(this.wallStart + t), frame_id: 'world', pose: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } }, point_stride: stride,
      fields: [{ name: 'x', offset: 0, type: 7 }, { name: 'y', offset: 4, type: 7 }, { name: 'z', offset: 8, type: 7 }, { name: 'intensity', offset: 12, type: 7 }, { name: 'label', offset: 16, type: 1 }, { name: 'ring', offset: 17, type: 1 }],
      data: b64(new Uint8Array(buf)),
    });
  }

  tick(tAbs: number, host: RecorderHost) {
    if (!this.recording) return;
    this.host = host;
    const t = this.rel(tAbs);
    if (t - this.lastTf >= 0.1) {
      this.lastTf = t;
      const j = host.twin.joints;
      const lp = host.sensorPose('lidar'), cp = host.sensorPose('camera');
      const base = toRos(host.machine.cx, host.machine.groundY, host.machine.cz);
      const stamp = ts(this.wallStart + t);
      const tfs = [
        { timestamp: stamp, parent_frame_id: 'world', child_frame_id: 'base_link', translation: base, rotation: quatYawRos(j.swing) },
        { timestamp: stamp, parent_frame_id: 'world', child_frame_id: 'lidar', translation: toRos(lp[0], lp[1], lp[2]), rotation: quatYawRos(j.swing) },
        { timestamp: stamp, parent_frame_id: 'world', child_frame_id: 'camera', translation: toRos(cp[0], cp[1], cp[2]), rotation: quatYawRos(j.swing) },
      ];
      this.put('/tf', t, { transforms: tfs });
      this.put('/machine/joints', t, { swing: j.swing, boom: j.boom, arm: j.arm, bucket: j.bucket, phase: host.autonomy.phase });
      const L = host.terrain.hf.ledger;
      const g = host.terrain.hf.designSpec ? host.terrain.hf.gradeStats() : { onGradePct: 0, remainingCut: 0, overDig: 0 };
      this.put('/earthwork', t, { bankCut: L.bankCut, sloughed: L.sloughed, dumped: L.dumped, onGradePct: g.onGradePct, remainingCut: g.remainingCut, overDig: g.overDig });
    }
    if (t - this.lastGrid >= 1) {
      this.lastGrid = t;
      const hf = host.terrain.hf;
      // dig-zone crop: 24 m x 24 m around the machine at the sim resolution
      const span = 24, n = Math.round(span / hf.cell);
      const ci0 = Math.floor((host.machine.cx - span / 2 + hf.half) / hf.cell), cj0 = Math.floor((host.machine.cz - span / 2 + hf.half) / hf.cell);
      const buf = new Float32Array(n * n);
      // Grid rows run along ROS +Y (= three -z): row r corresponds to three z decreasing
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        const i = ci0 + c, j = cj0 + (n - 1 - r);
        const k = Math.min(hf.N - 1, Math.max(0, j)) * hf.N + Math.min(hf.N - 1, Math.max(0, i));
        buf[r * n + c] = hf.B[k] + hf.L[k];
      }
      const ox = (ci0) * hf.cell - hf.half, oz = (cj0 + n) * hf.cell - hf.half;
      this.put('/terrain/grid', t, {
        timestamp: ts(this.wallStart + t), frame_id: 'world', pose: { position: { x: ox, y: -oz, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
        column_count: n, cell_size: { x: hf.cell, y: hf.cell }, row_stride: n * 4, cell_stride: 4, fields: [{ name: 'elevation', offset: 0, type: 7 }], data: b64(new Uint8Array(buf.buffer)),
      });
    }
    if (t - this.lastCam >= 0.2 && !this.camBusy) {
      this.lastCam = t;
      this.camBusy = true;
      const tt = t;
      void this.grabJpeg(host).then((jpeg) => {
        this.camBusy = false;
        if (jpeg && this.recording) this.put('/camera/roof', tt, { timestamp: ts(this.wallStart + tt), frame_id: 'camera', data: b64(jpeg), format: 'jpeg' });
      }).catch(() => { this.camBusy = false; });
    }
  }

  private async grabJpeg(host: RecorderHost): Promise<Uint8Array | null> {
    const px = await host.cab.readPixels();
    const c = document.createElement('canvas');
    c.width = host.cab.width; c.height = host.cab.height;
    c.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(px), c.width, c.height), 0, 0);
    const blob = await new Promise<Blob | null>((res) => c.toBlob(res, 'image/jpeg', 0.82));
    return blob ? new Uint8Array(await blob.arrayBuffer()) : null;
  }

  async stopAsync(): Promise<{ bytes: Uint8Array; summary: string }> {
    this.recording = false;
    await this.chain;
    await this.writer!.end();
    const bytes = this.w!.bytes();
    const total = Object.values(this.counts).reduce((a, b) => a + b, 0);
    const summary = `${(bytes.byteLength / 1e6).toFixed(1)} MB MCAP · ${total} messages · ${(this.counts['/lidar/roof/points'] ?? 0) + (this.counts['/lidar/front/points'] ?? 0)} lidar scans · ${this.counts['/camera/roof'] ?? 0} frames`;
    this.writer = null;
    void this.host;
    return { bytes, summary };
  }
}
