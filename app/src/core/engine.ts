import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SparkRenderer } from '@sparkjsdev/spark';

export type FrameHook = (dt: number, t: number) => void;

/** Renderer, camera, controls, sky and the frame loop. */
export class Engine {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly spark: SparkRenderer;
  readonly canvas: HTMLCanvasElement;
  private hooks: FrameHook[] = [];
  private postHooks: FrameHook[] = [];
  private last = performance.now();
  private t0 = performance.now();
  fps = 60;
  private fpsAcc = 0;
  private fpsN = 0;
  frameMs = 16;
  private skyTex: THREE.CanvasTexture;

  constructor(container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance', preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.canvas = this.renderer.domElement;
    this.canvas.className = 'gl';
    container.appendChild(this.canvas);

    this.camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.05, 600);
    this.camera.position.set(9, 9.5, 12);

    this.spark = new SparkRenderer({ renderer: this.renderer, maxStdDev: Math.sqrt(8) });
    this.scene.add(this.spark);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 1.5, 0);
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 1.5;
    this.controls.maxDistance = 80;
    this.controls.screenSpacePanning = false;

    this.skyTex = this.makeSky('#cfd8de', '#eef1f2', '#d9d2c6');
    this.scene.background = this.skyTex;

    const ro = new ResizeObserver(() => this.resize(container));
    ro.observe(container);
  }

  makeSky(top: string, horizon: string, ground: string) {
    const c = document.createElement('canvas');
    c.width = 4; c.height = 256;
    const g = c.getContext('2d')!;
    const grd = g.createLinearGradient(0, 0, 0, 256);
    grd.addColorStop(0, top);
    grd.addColorStop(0.5, horizon);
    grd.addColorStop(1, ground);
    g.fillStyle = grd;
    g.fillRect(0, 0, 4, 256);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.mapping = THREE.EquirectangularReflectionMapping;
    return t;
  }

  setSky(top: string, horizon: string, ground: string) {
    const old = this.skyTex;
    this.skyTex = this.makeSky(top, horizon, ground);
    this.scene.background = this.skyTex;
    old.dispose();
  }

  resize(container: HTMLElement) {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  onFrame(h: FrameHook) { this.hooks.push(h); }
  /** hooks that render extra passes after the main render (e.g. picture-in-picture) */
  onAfterRender(h: FrameHook) { this.postHooks.push(h); }

  start() {
    this.renderer.setAnimationLoop(() => this.tick());
  }

  private tick() {
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    const t = (now - this.t0) / 1000;
    this.controls.update();
    for (const h of this.hooks) h(dt, t);
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
    this.renderer.render(this.scene, this.camera);
    for (const h of this.postHooks) h(dt, t);
    this.frameMs = performance.now() - now;
    this.fpsAcc += dt; this.fpsN++;
    if (this.fpsAcc > 0.5) { this.fps = this.fpsN / this.fpsAcc; this.fpsAcc = 0; this.fpsN = 0; }
  }
}
