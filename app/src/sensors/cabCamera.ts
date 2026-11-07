import * as THREE from 'three';
import { SparkRenderer } from '@sparkjsdev/spark';

/**
 * Roof-mounted forward camera rendered from the splat scene by a secondary SparkRenderer (its own sort
 * order and offscreen target), composited into the main canvas as picture-in-picture, and read back to
 * the CPU on demand (detector input, MCAP export).
 */
export class CabCamera {
  readonly camera: THREE.PerspectiveCamera;
  readonly spark: SparkRenderer;
  readonly width: number;
  readonly height: number;
  private quadScene = new THREE.Scene();
  private quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private busy = false;
  enabled = true;
  rateHz = 15;
  private acc = 0;
  frames = 0;
  /** device pixel rect of the PiP inside the canvas (css px), set by the UI */
  rect = { x: 0, y: 0, w: 360, h: 203 };
  lastRenderMs = 0;

  constructor(renderer: THREE.WebGLRenderer, width = 640, height = 360, fovDeg = 72) {
    this.width = width;
    this.height = height;
    this.camera = new THREE.PerspectiveCamera(fovDeg, width / height, 0.1, 400);
    this.spark = new SparkRenderer({ renderer, target: { width, height }, maxStdDev: Math.sqrt(8) });
    const mat = new THREE.ShaderMaterial({
      uniforms: { map: { value: this.spark.target!.texture } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: 'uniform sampler2D map; varying vec2 vUv; void main(){ gl_FragColor = vec4(texture2D(map, vUv).rgb, 1.0); }',
      depthTest: false, depthWrite: false,
    });
    this.quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
  }

  /** set the camera pose from a world position and a look direction (yaw atan2(z,x) convention + pitch) */
  setPose(pos: THREE.Vector3, yaw: number, pitch: number) {
    this.camera.position.copy(pos);
    const d = new THREE.Vector3(Math.cos(yaw) * Math.cos(pitch), Math.sin(pitch), Math.sin(yaw) * Math.cos(pitch));
    this.camera.lookAt(pos.clone().add(d));
    this.camera.updateMatrixWorld();
  }

  /** when true the periodic tick is suspended (e.g. a perception sweep drives renders explicitly) */
  paused = false;

  /** render a frame right now and return its pixels */
  async renderNow(scene: THREE.Scene): Promise<Uint8Array> {
    while (this.busy) await new Promise((r) => setTimeout(r, 5));
    this.busy = true;
    try {
      await this.spark.update({ scene, camera: this.camera });
      this.spark.renderTarget({ scene, camera: this.camera });
      this.frames++;
    } finally { this.busy = false; }
    return this.readPixels();
  }

  async tick(dt: number, scene: THREE.Scene) {
    if (!this.enabled || this.paused) return;
    this.acc += dt;
    if (this.busy || this.acc < 1 / this.rateHz) return;
    this.acc = 0;
    this.busy = true;
    try {
      const t0 = performance.now();
      await this.spark.update({ scene, camera: this.camera });
      this.spark.renderTarget({ scene, camera: this.camera });
      this.lastRenderMs = performance.now() - t0;
      this.frames++;
    } finally {
      this.busy = false;
    }
  }

  /** draw the latest target into the PiP rect of the main canvas */
  composite(renderer: THREE.WebGLRenderer) {
    if (!this.enabled || this.frames === 0) return;
    const c = renderer.domElement;
    const H = c.clientHeight;
    const { x, y, w, h } = this.rect;
    renderer.setScissorTest(true);
    renderer.setViewport(x, H - y - h, w, h);
    renderer.setScissor(x, H - y - h, w, h);
    renderer.render(this.quadScene, this.quadCam);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, c.clientWidth, c.clientHeight);
  }

  /** RGBA pixels of the last frame (top row first) */
  async readPixels(): Promise<Uint8Array> {
    const px = await this.spark.readTarget();
    // WebGL render targets are bottom-up: flip rows
    const w = this.width, h = this.height, out = new Uint8Array(px.length);
    for (let y = 0; y < h; y++) out.set(px.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
    return out;
  }
}
