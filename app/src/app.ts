import * as THREE from 'three';
import { Engine } from './core/engine';
import { TerrainSystem } from './core/terrainSystem';
import type { SceneData } from './data/scene';
import { ExcavatorModel } from './machine/excavatorModel';
import { DX225_GEOMETRY, solveIk, forwardPlanar, type JointState } from './machine/kinematics';
import { DigAutonomy } from './machine/autonomy';
import { LidarClient, type ScanResult } from './sensors/lidarClient';
import { LidarPoints, LidarPanel } from './sensors/lidarViews';
import { CabCamera } from './sensors/cabCamera';
import { PointLabel, type Capsule } from './sensors/lidarModel';
import { Environment, LOOKS } from './scenario/environment';
import { Worker as SiteWorker } from './scenario/actors';
import { Overlays } from './render/overlays';
import { SOILS } from './sim/soil';
import type { DesignTrench } from './sim/heightfield';
import { $, $$, setText, fmt, toast, icons } from './ui/dom';
import { McapRecorder } from './export/mcap';
import { YoloxDetector, type Detection } from './perception/detector';
import { runSweep, CONDITIONS, DISTANCES, type SweepResult, type SweepHost, type Condition } from './perception/sweep';

type Mode = 'explore' | 'dig' | 'design' | 'scenario' | 'auto';
const DEG = Math.PI / 180;
export const REPO_URL = 'https://github.com/Mohith26/livedirt';

const PHYS_DT = 0.05;

export class App {
  private physAcc = 0;
  readonly engine: Engine;
  readonly data: SceneData;
  readonly terrain: TerrainSystem;
  readonly twin: ExcavatorModel;
  readonly autonomy: DigAutonomy;
  readonly lidar: LidarClient;
  readonly lidarFront: LidarClient;
  lidarView: 'roof' | 'front' = 'roof';
  private lastFront: ScanResult | null = null;
  readonly lidarPts = new LidarPoints();
  lidarPanel!: LidarPanel;
  readonly cab: CabCamera;
  readonly env = new Environment();
  readonly worker = new SiteWorker();
  readonly overlays: Overlays;
  readonly mcap: McapRecorder;
  readonly detector = new YoloxDetector(416);
  detectorOn = false;
  private detBusy = false;
  private detTimer = 0;
  private lastDets: Detection[] = [];
  sweep: SweepResult | null = null;
  sweeping = false;
  private cabYawOverride: number | null = null;
  mode: Mode = 'explore';
  // tool state
  brush = 0.7;
  depthRate = 0.8; // m/s at the brush centre
  gradeAssist = true;
  spoilMode: 'auto' | 'pick' = 'auto';
  spoilPoint: THREE.Vector3 | null = null;
  designParams = { bottomWidth: 0.9, depth: 1.0, sideSlope: 1, tol: 0.03 };
  designDraft: THREE.Vector3 | null = null;
  gradeOverlay = true;
  showLidarInScene = false;
  twinVisible = false;
  placingWorker = false;
  // runtime
  private cursor: THREE.Vector3 | null = null;
  private digging = false;
  private strokeDir = new THREE.Vector2(1, 0);
  private lastDigPos: THREE.Vector2 | null = null;
  private ndc = new THREE.Vector2();
  private rc = new THREE.Raycaster();
  private surfaceTimer = 0;
  private personReturns = 0;
  private personTruthInZone = false;
  private personDetectedInZone = false;
  private zoneR = 0;
  private lastScan: ScanResult | null = null;
  private telemetryTimer = 0;
  private hasDug = false;
  readonly machine: { cx: number; cz: number; groundY: number; trackYaw: number; houseYaw: number };
  private restJoints: JointState;

  constructor(engine: Engine, data: SceneData, terrain: TerrainSystem) {
    this.engine = engine;
    this.data = data;
    this.terrain = terrain;
    const m = data.machine;
    this.machine = { cx: m.center.x, cz: m.center.z, groundY: m.groundY, trackYaw: m.trackYawDeg * DEG, houseYaw: m.houseYawDeg * DEG };

    // lights for meshes (the splats carry baked lighting)
    const hemi = new THREE.HemisphereLight(0xe4ebef, 0x5a534c, 1.25);
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(-12, 25, 8);
    engine.scene.add(hemi, sun);

    // twin placed at the photographed machine's measured pose, joints posed to match the capture
    this.twin = new ExcavatorModel(DX225_GEOMETRY);
    this.twin.place(this.machine.cx, this.machine.groundY, this.machine.cz, this.machine.trackYaw);
    const dir = new THREE.Vector2(Math.cos(this.machine.houseYaw), Math.sin(this.machine.houseYaw));
    const tipXZ = new THREE.Vector2(this.machine.cx, this.machine.cz).addScaledVector(dir, 6.2);
    const tipY = terrain.hf.heightAt(tipXZ.x, tipXZ.y) + 0.25;
    const ik = solveIk(DX225_GEOMETRY, this.machine.cx, this.machine.groundY, this.machine.cz, [tipXZ.x, tipY, tipXZ.y], -100 * DEG, this.machine.houseYaw);
    this.restJoints = { ...ik.joints };
    this.twin.setJoints(this.restJoints);
    this.twin.root.visible = false;
    engine.scene.add(this.twin.root);

    this.autonomy = new DigAutonomy(terrain.hf, DX225_GEOMETRY, this.machine.cx, this.machine.cz, this.machine.groundY, this.restJoints);
    // grade survey merges both lidars (the level roof unit cannot see the near field in front of the machine)
    this.autonomy.requestScan = () => {
      const parts = [this.lastFront, this.lastScan].filter((x): x is ScanResult => !!x);
      if (!parts.length) return null;
      const n = parts.reduce((a, p) => a + p.count, 0);
      const out = new Float32Array(n * 4);
      let o = 0;
      for (const p of parts) { out.set(p.points.subarray(0, p.count * 4), o); o += p.count * 4; }
      return out;
    };

    // sensors
    const N = data.meta.N;
    const albedo = new Float32Array(N * N);
    for (let k = 0; k < N * N; k++) albedo[k] = (0.299 * data.albedo[k * 3] + 0.587 * data.albedo[k * 3 + 1] + 0.114 * data.albedo[k * 3 + 2]) / 255;
    const obstacle = new Uint8Array(N * N);
    for (let k = 0; k < N * N; k++) obstacle[k] = data.objectCode[k] === 2 ? 1 : 0;
    this.lidar = new LidarClient(N, data.meta.cell, data.meta.half, terrain.sensorRay.surf, obstacle, albedo);
    this.lidar.onScan = (s) => this.onScan(s, 'roof');
    this.lidarFront = new LidarClient(N, data.meta.cell, data.meta.half, terrain.sensorRay.surf, obstacle, albedo);
    this.lidarFront.hRes = 512;
    this.lidarFront.onScan = (s) => this.onScan(s, 'front');
    this.lidarPts.points.visible = false;
    engine.scene.add(this.lidarPts.points);
    this.cab = new CabCamera(engine.renderer, 640, 360, 74);

    engine.scene.add(this.env.group, this.worker.group);
    this.overlays = new Overlays(terrain.hf);
    engine.scene.add(this.overlays.group);
    this.mcap = new McapRecorder();

    terrain.uniforms.grade.x = this.designParams.tol;
    this.buildUI();
    this.bindInput();
    engine.onFrame((dt, t) => this.frame(dt, t));
    engine.onAfterRender(() => this.cab.composite(engine.renderer));
  }

  // ------------------------------------------------------------------ UI
  private buildUI() {
    const ui = document.createElement('div');
    ui.innerHTML = `
      <header class="topbar">
        <div class="brand glass"><div class="mark">${LOGO}</div><div><b>LiveDirt</b><span>A neural job site you can dig into</span></div></div>
        <nav class="modes glass">
          ${(['explore', 'dig', 'design', 'scenario', 'auto'] as Mode[]).map((m, i) => `<button data-mode="${m}">${icons[m === 'auto' ? 'auto' : m]}<span>${MODE_LABEL[m]}</span><kbd>${i + 1}</kbd></button>`).join('')}
        </nav>
        <div class="actions">
          <button class="iconbtn glass" id="btnMcap" title="Record sensor data to an MCAP file (opens in Foxglove)">${icons.download}<span id="mcapLabel">Record MCAP</span></button>
          <button class="iconbtn glass" id="btnAbout">${icons.info}<span>What's real</span></button>
          <a class="iconbtn glass" href="${REPO_URL}" target="_blank" rel="noopener" title="Source on GitHub">${icons.github}</a>
        </div>
      </header>
      <aside class="toolpanel glass" id="toolpanel"></aside>
      <aside class="telemetry" id="telemetry">
        <section class="card glass" id="cardEarth">
          <div class="head"><h3>Earthwork</h3><span class="pill" id="soilPill">Wet clay</span></div>
          <div class="big" id="tBank">0.00<small>m³ bank cut</small></div>
          <div class="row"><span>Loose produced <small>(× swell)</small></span><b id="tLoose">0.00</b></div>
          <div class="row"><span>Sloughed from walls</span><b id="tSlough">0.00</b></div>
          <div class="row"><span>Spoil on site</span><b id="tField">0.00</b></div>
          <div class="row"><span>Mass moved</span><b id="tMass">0.0 t</b></div>
        </section>
        <section class="card glass" id="cardGrade" hidden>
          <div class="head"><h3>Grade control</h3><span class="pill" id="gradePill">no design</span></div>
          <div class="big" id="gOn">0%<small>on grade</small></div>
          <div class="bar"><i id="gBar"></i></div>
          <div class="row"><span>Remaining cut</span><b id="gRem">0.00</b></div>
          <div class="row"><span>Over-dig</span><b id="gOver">0.00</b></div>
          <div class="row"><span>Cut outside design</span><b id="gUnpl">0.00</b></div>
          <div class="row"><span>Mean |error|</span><b id="gMae">0.0 cm</b></div>
          <div class="legend"><span><i style="background:#3f8cff"></i>above grade</span><span><i style="background:#33f075"></i>on grade ±<b id="gTol">3</b> cm</span><span><i style="background:#ff3a2a"></i>over-dug</span></div>
        </section>
        <section class="card glass" id="cardSafety">
          <div class="head"><h3>Safety monitor</h3><span class="pill ok" id="safePill"><i></i>Clear</span></div>
          <div class="kv">
            <span>Swing exclusion radius</span><b id="sZone">0.0 m</b>
            <span>Nearest person</span><b id="sDist">none</b>
            <span>Lidar returns on person</span><b id="sRet">0</b>
          </div>
        </section>
        <section class="card glass" id="cardSensors">
          <div class="head"><h3>Sensors</h3><span class="pill" id="sensPill">2 × 64-beam lidar · 10 Hz</span></div>
          <div class="kv">
            <span>Points / scan</span><b id="lPts">0</b>
            <span>Dropout (of hits)</span><b id="lDrop">0%</b>
            <span>Dust / rain returns</span><b id="lAtm">0</b>
            <span>Scan compute</span><b id="lMs">0 ms</b>
            <span>Cab cam render</span><b id="cMs">0 ms</b>
          </div>
        </section>
      </aside>
      <section class="dock" id="dock">
        <div class="pip glass" id="pipCam"><div class="ph" id="camPh"></div><div class="cap">${icons.camera}ROOF CAM <em>· splat render</em></div><div class="rec"></div><canvas class="boxes" id="camBoxes" width="352" height="198"></canvas></div>
        <div class="pip glass lidarbox">
          <div class="cap">${icons.lidar}LIDAR <em>·</em><span class="lsw"><button data-v="roof" class="on">Roof 360°</button><button data-v="front">Front, tilted 35°</button></span></div>
          <div class="lidarwrap"><canvas class="bev" id="lidarBev" width="396" height="396"></canvas>
            <div class="lidarmeta"><div><span>terrain</span><b id="lmT">0</b></div><div><span>obstacle</span><b id="lmO">0</b></div><div><span>person</span><b id="lmP">0</b></div><div><span>dust</span><b id="lmD">0</b></div><div><span>rain</span><b id="lmR">0</b></div><div style="margin-top:8px"><span>range</span><b>90 m</b></div><div><span>v-FOV</span><b>±22.5°</b></div></div></div>
          <canvas class="range" id="lidarRange" width="1024" height="64"></canvas>
        </div>
      </section>
      <div class="status glass" id="status"><span><b id="stFps">0</b> fps</span><span><b id="stSplats">0</b> captured splats</span><span><b id="stGen">0</b> generated</span><span id="stSim">settled</span></div>
      <div class="coach glass" id="coach"></div>
      <div id="toasts"></div>
      <div class="scrim" id="scrim"></div>
      <aside class="drawer" id="about">${aboutHtml(this.data)}</aside>
    `;
    document.body.appendChild(ui);
    this.lidarPanel = new LidarPanel($('#lidarBev') as HTMLCanvasElement, $('#lidarRange') as HTMLCanvasElement);
    $$('.modes button').forEach((b) => b.addEventListener('click', () => this.setMode(b.dataset.mode as Mode)));
    $$('.lsw button').forEach((b) => b.addEventListener('click', () => { this.lidarView = b.dataset.v as 'roof' | 'front'; $$('.lsw button').forEach((x) => x.classList.toggle('on', x === b)); }));
    $('#btnAbout').addEventListener('click', () => this.toggleAbout(true));
    $('#scrim').addEventListener('click', () => this.toggleAbout(false));
    $('#about').addEventListener('click', (e) => { if ((e.target as HTMLElement).closest('.x')) this.toggleAbout(false); });
    $('#btnMcap').addEventListener('click', () => this.toggleMcap());
    this.setMode('explore');
    this.coach('<b>Drag</b> to orbit · press <b>2</b> to dig');
  }

  private toggleAbout(on: boolean) {
    $('#about').classList.toggle('open', on);
    $('#scrim').classList.toggle('on', on);
  }

  coach(html: string, ms = 5200) {
    const c = $('#coach');
    c.innerHTML = html;
    c.classList.add('show');
    clearTimeout((c as any)._t);
    (c as any)._t = setTimeout(() => c.classList.remove('show'), ms);
  }

  setMode(m: Mode) {
    const prev = this.mode;
    this.mode = m;
    $$('.modes button').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
    const stage = $('#stage');
    stage.classList.toggle('dig', m === 'dig');
    stage.classList.toggle('design', m === 'design');
    const c = this.engine.controls;
    c.mouseButtons = m === 'dig' || m === 'design'
      ? { LEFT: null as unknown as THREE.MOUSE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE }
      : { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    c.touches = m === 'dig' || m === 'design' ? { ONE: null as unknown as THREE.TOUCH, TWO: THREE.TOUCH.DOLLY_ROTATE } : { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    if (m !== 'dig') this.overlays.hideBrush();
    if (m === 'auto') this.setTwin(true);
    else if (prev === 'auto' && this.autonomy.phase !== 'idle' && this.autonomy.phase !== 'done') { /* keep running in background */ }
    this.placingWorker = false;
    const tp = $('#toolpanel');
    tp.classList.add('swap');
    setTimeout(() => { this.renderPanel(); tp.classList.remove('swap'); }, 120);
    if (m === 'dig') this.coach('<b>Drag on the soil</b> to dig · right-drag to orbit · <b>Z</b> to undo');
    if (m === 'design') this.coach('<b>Click twice</b> to set the trench centreline');
    if (m === 'auto' && !this.terrain.hf.designSpec) this.coach('Autonomy needs a design trench: use the <b>example trench</b> or draw one');
  }

  setTwin(on: boolean) {
    this.twinVisible = on;
    this.terrain.setMachineFill(on);
    this.twin.root.visible = on;
    this.terrain.uniforms.flags.z = on ? 1 : 0;
    this.terrain.tiles.forEach((t) => t.updateVersion());
  }

  private slider(id: string, label: string, min: number, max: number, step: number, value: number, unit: string, fmtv: (v: number) => string = (v) => v.toFixed(2)) {
    const p = ((value - min) / (max - min)) * 100;
    return `<div class="field"><label>${label}<b id="${id}V">${fmtv(value)}${unit}</b></label><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}" style="--p:${p}%"></div>`;
  }
  private bindSlider(id: string, unit: string, fmtv: (v: number) => string, cb: (v: number) => void) {
    const el = $(`#${id}`) as HTMLInputElement;
    if (!el) return;
    const upd = () => {
      const v = parseFloat(el.value);
      el.style.setProperty('--p', `${((v - +el.min) / (+el.max - +el.min)) * 100}%`);
      setText($(`#${id}V`), fmtv(v) + unit);
      cb(v);
    };
    el.addEventListener('input', upd);
  }
  private toggleHtml(id: string, label: string, on: boolean, small = '') {
    return `<div class="toggle ${on ? 'on' : ''}" id="${id}"><div class="lbl">${label}${small ? `<small>${small}</small>` : ''}</div><div class="sw"></div></div>`;
  }
  private bindToggle(id: string, cb: (on: boolean) => void) {
    const el = $(`#${id}`);
    if (!el) return;
    el.addEventListener('click', () => { el.classList.toggle('on'); cb(el.classList.contains('on')); });
  }

  private renderPanel() {
    const tp = $('#toolpanel');
    const hf = this.terrain.hf;
    const m = this.mode;
    if (m === 'explore') {
      tp.innerHTML = `
        <h3>Explore the capture</h3>
        <div class="sub">A real UK construction site, reconstructed from drone video as 1.9M 3D Gaussians and scaled to metres.</div>
        ${this.toggleHtml('tGen', 'Show generated soil', this.terrain.uniforms.flags.x > 0.5, 'Tints everything the camera never saw')}
        ${this.toggleHtml('tLid', 'Lidar points in scene', this.showLidarInScene, 'Latest scan, coloured by height')}
        ${this.toggleHtml('tTwin', 'Machine twin', this.twinVisible, 'Swap the photographed excavator for the articulated model')}
        <div class="btns"><button class="btn grow" id="vOver">Overview</button><button class="btn grow" id="vCab">Operator view</button><button class="btn grow" id="vLow">Ground level</button></div>
        <div class="hint"><b>Left-drag</b> orbit · <b>right-drag</b> pan · <b>scroll</b> zoom · <b>double-click</b> to focus</div>`;
      this.bindToggle('tGen', (on) => this.terrain.setGeneratedOverlay(on));
      this.bindToggle('tLid', (on) => { this.showLidarInScene = on; this.lidarPts.points.visible = on; });
      this.bindToggle('tTwin', (on) => this.setTwin(on));
      $('#vOver').onclick = () => this.flyTo('overview');
      $('#vCab').onclick = () => this.flyTo('operator');
      $('#vLow').onclick = () => this.flyTo('ground');
    } else if (m === 'dig') {
      tp.innerHTML = `
        <h3>Dig</h3>
        <div class="sub">Earthwork physics on the captured terrain: swell, repose, wall sloughing, volume ledger.</div>
        ${this.slider('sBrush', 'Bucket footprint', 0.3, 1.6, 0.05, this.brush, ' m')}
        ${this.slider('sRate', 'Cut rate', 0.2, 2, 0.05, this.depthRate, ' m/s')}
        <div class="field"><label>Soil</label><select id="sSoil">${SOILS.map((s) => `<option value="${s.id}" ${s.id === hf.soil.id ? 'selected' : ''}>${s.label}</option>`).join('')}</select>
        <div class="note" id="soilNote">${hf.soil.note} Swell ×${hf.soil.swell}, repose ${hf.soil.reposeDeg}°, walls stand to ${hf.soil.bankDeg}°.</div></div>
        <div class="field"><label>Spoil placement</label><div class="seg" id="segSpoil"><button data-v="auto" class="${this.spoilMode === 'auto' ? 'on' : ''}">Beside the cut</button><button data-v="pick" class="${this.spoilMode === 'pick' ? 'on' : ''}">Pick a spot</button></div></div>
        ${this.toggleHtml('tAssist', 'Grade assist', this.gradeAssist, 'Never cut below the design grade (if one is set)')}
        <div class="btns"><button class="btn grow" id="bUndo">${icons.undo}Undo</button><button class="btn grow danger" id="bReset">${icons.reset}Reset site</button></div>
        <div class="hint"><b>Left-drag</b> dig · <b>right-drag</b> orbit · <b>[ ]</b> footprint · <b>Z</b> undo</div>`;
      this.bindSlider('sBrush', ' m', (v) => v.toFixed(2), (v) => (this.brush = v));
      this.bindSlider('sRate', ' m/s', (v) => v.toFixed(2), (v) => (this.depthRate = v));
      ($('#sSoil') as HTMLSelectElement).onchange = (e) => { this.terrain.setSoil((e.target as HTMLSelectElement).value); this.renderPanel(); };
      $$('#segSpoil button').forEach((b) => b.addEventListener('click', () => { this.spoilMode = b.dataset.v as 'auto' | 'pick'; $$('#segSpoil button').forEach((x) => x.classList.toggle('on', x === b)); if (this.spoilMode === 'pick') this.coach('<b>Shift-click</b> where the spoil should go'); else { this.spoilPoint = null; this.overlays.setSpoil(null); } }));
      this.bindToggle('tAssist', (on) => (this.gradeAssist = on));
      $('#bUndo').onclick = () => this.undo();
      $('#bReset').onclick = () => this.resetSite();
    } else if (m === 'design') {
      const d = this.designParams;
      tp.innerHTML = `
        <h3>Design trench</h3>
        <div class="sub">Set a target grade like a machine-control model; the soil is scored against it live.</div>
        ${this.slider('dW', 'Bottom width', 0.6, 3, 0.05, d.bottomWidth, ' m')}
        ${this.slider('dD', 'Depth below grade', 0.3, 3, 0.05, d.depth, ' m')}
        <div class="field"><label>Side slope (run : rise)</label><div class="seg" id="segSlope">${[[0, 'Vertical'], [0.5, '½ : 1'], [1, '1 : 1'], [1.5, '1½ : 1']].map(([v, l]) => `<button data-v="${v}" class="${d.sideSlope === v ? 'on' : ''}">${l}</button>`).join('')}</div></div>
        ${this.slider('dT', 'Tolerance', 0.01, 0.1, 0.005, d.tol, ' cm', (v) => (v * 100).toFixed(1))}
        ${this.toggleHtml('tHeat', 'Grade heatmap on soil', this.gradeOverlay)}
        <div class="btns"><button class="btn primary grow" id="bExample">Example trench</button><button class="btn grow" id="bClear">Clear</button></div>
        <div class="hint"><b>Click</b> start, <b>click</b> end. Vertical walls need shoring in practice; in this soil model they may slough.</div>`;
      this.bindSlider('dW', ' m', (v) => v.toFixed(2), (v) => { d.bottomWidth = v; this.reapplyDesign(); });
      this.bindSlider('dD', ' m', (v) => v.toFixed(2), (v) => { d.depth = v; this.reapplyDesign(); });
      this.bindSlider('dT', ' cm', (v) => (v * 100).toFixed(1), (v) => { d.tol = v; this.terrain.uniforms.grade.x = v; this.terrain.refreshLook(); });
      $$('#segSlope button').forEach((b) => b.addEventListener('click', () => { d.sideSlope = parseFloat(b.dataset.v!); $$('#segSlope button').forEach((x) => x.classList.toggle('on', x === b)); this.reapplyDesign(); }));
      this.bindToggle('tHeat', (on) => { this.gradeOverlay = on; this.terrain.uniforms.flags.y = on && hf.designSpec ? 1 : 0; this.terrain.refreshLook(); });
      $('#bExample').onclick = () => this.exampleTrench();
      $('#bClear').onclick = () => this.setDesign(null);
    } else if (m === 'scenario') {
      tp.innerHTML = `
        <h3>Scenario</h3>
        <div class="sub">Stress the sensors. One capture, many conditions.</div>
        <div class="field"><label>Lighting</label><div class="chips" id="chLook">${Object.entries(LOOKS).map(([k, v]) => `<button data-v="${k}" class="${this.env.look === k ? 'on' : ''}">${v.label}</button>`).join('')}</div></div>
        ${this.slider('scRain', 'Rain', 0, 1, 0.05, this.env.rain, '', (v) => (v * 100).toFixed(0) + '%')}
        ${this.slider('scDust', 'Airborne dust', 0, 1, 0.05, this.env.dust, '', (v) => (v * 100).toFixed(0) + '%')}
        <div class="field"><label>Ground worker</label><div class="btns" style="margin-top:0"><button class="btn grow" id="bPlace">${icons.person}${this.worker.group.visible ? 'Move worker' : 'Place worker'}</button><button class="btn" id="bRemove" ${this.worker.group.visible ? '' : 'disabled'}>Remove</button></div></div>
        <div class="btns"><button class="btn grow" id="bBlind">Blind-spot test</button></div>
        ${this.toggleHtml('tGen2', 'Show generated soil', this.terrain.uniforms.flags.x > 0.5)}
        <div class="field" style="margin-top:16px"><label>Perception</label>
          ${this.toggleHtml('tDet', 'Person detector on roof cam', this.detectorOn, 'YOLOX-Nano (COCO), in-browser WASM')}
          <div class="btns" style="margin-top:4px"><button class="btn grow" id="bSweep" ${this.sweeping ? 'disabled' : ''}>Run perception sweep</button></div>
          <div class="hint" id="sweepMsg">${this.sweep ? '' : '4 conditions × 8 distances × 4 bearings in twin mode: camera vs lidar recall.'}</div>
          <div id="sweepOut">${this.sweep ? sweepTable(this.sweep) : ''}</div>
        </div>
        <div class="note">Lighting is a colour grade on baked splat colours, not relighting. Rain and dust also change the lidar: extinction, early returns and dropouts.</div>`;
      $$('#chLook button').forEach((b) => b.addEventListener('click', () => { this.env.look = b.dataset.v!; $$('#chLook button').forEach((x) => x.classList.toggle('on', x === b)); this.applyEnv(); }));
      this.bindSlider('scRain', '', (v) => (v * 100).toFixed(0) + '%', (v) => { this.env.rain = v; this.terrain.uniforms.flags.w = v; this.applyEnv(); });
      this.bindSlider('scDust', '', (v) => (v * 100).toFixed(0) + '%', (v) => { this.env.dust = v; this.applyEnv(); });
      $('#bPlace').onclick = () => { this.placingWorker = true; $('#stage').classList.add('place'); this.coach('<b>Click</b> the ground to place the worker'); };
      $('#bRemove').onclick = () => { this.worker.group.visible = false; this.renderPanel(); };
      $('#bBlind').onclick = () => this.blindSpotTest();
      this.bindToggle('tGen2', (on) => this.terrain.setGeneratedOverlay(on));
      this.bindToggle('tDet', async (on) => { this.detectorOn = on; if (on) await this.ensureDetector(); if (!on) { this.lastDets = []; this.drawDets(); } });
      $('#bSweep').onclick = () => this.runPerceptionSweep();
    } else if (m === 'auto') {
      const a = this.autonomy;
      const hasDesign = !!hf.designSpec;
      tp.innerHTML = `
        <h3>Autonomous trenching</h3>
        <div class="sub">The twin digs the design trench to grade, checks its work with its own lidar, and stops for people.</div>
        ${hasDesign ? '' : `<div class="note">No design yet.</div><div class="btns"><button class="btn primary grow" id="aExample">Use example trench</button></div>`}
        <div class="btns"><button class="btn primary grow" id="aRun" ${hasDesign ? '' : 'disabled'}>${a.phase === 'idle' || a.phase === 'done' ? icons.play + 'Run dig cycle' : icons.pause + 'Pause'}</button><button class="btn" id="aReset">${icons.reset}</button></div>
        <div class="field"><label>Sim speed</label><div class="seg" id="segSpeed">${[1, 2, 4, 8].map((v) => `<button data-v="${v}" class="${a.speed === v ? 'on' : ''}">${v}×</button>`).join('')}</div></div>
        <div class="card" style="padding:10px 0 0">
          <div class="row"><span>Phase</span><b class="phase" id="aPhase">${a.phase}</b></div>
          <div class="row"><span>Bite</span><b id="aBite">0 / 0</b></div>
          <div class="row"><span>Pass</span><b id="aPass">0</b></div>
          <div class="row"><span>Bucket</span><b id="aLoad">0.00 m³</b></div>
          <div class="bar"><i id="aLoadBar"></i></div>
          <div class="row"><span>Cycles · avg time</span><b id="aCyc">0 · 0.0 s</b></div>
          <div class="row"><span>Avg bucket fill</span><b id="aFill">0%</b></div>
          <div class="row"><span>Productivity</span><b id="aProd">0 m³/h</b></div>
          <div class="row"><span>Lidar vs truth (grade)</span><b id="aSurv">n/a</b></div>
          <div class="row"><span>Soil drag on bucket</span><b id="aForce">${a.neural ? '0 kN' : 'kinematic cut'}</b></div>
        </div>
        ${this.toggleHtml('tNeural', 'Learned soil (beta)', !!a.neural, 'U-Net trained on 3D MPM dig cycles; predicts terrain and bucket force. Arm slows above the DX225 arm force (112.8 kN).')}
        <div class="log" id="aLog"></div>`;
      if (!hasDesign) $('#aExample').onclick = () => { this.exampleTrench(); this.renderPanel(); };
      $('#aRun').onclick = () => this.toggleAutonomy();
      $('#aReset').onclick = () => { a.stop(); a.joints = { ...this.restJoints }; this.twin.setJoints(a.joints); this.renderPanel(); };
      $$('#segSpeed button').forEach((b) => b.addEventListener('click', () => { a.speed = parseFloat(b.dataset.v!); $$('#segSpeed button').forEach((x) => x.classList.toggle('on', x === b)); }));
      this.bindToggle('tNeural', async (on) => { await this.setNeuralSoil(on); this.renderPanel(); });
    }
  }

  // ------------------------------------------------------------------ actions
  private toggleAutonomy() {
    const a = this.autonomy;
    if (a.phase === 'idle' || a.phase === 'done') {
      if (!this.terrain.hf.designSpec) return;
      a.plan(this.terrain.hf.designSpec);
      this.overlays.setSpoil(new THREE.Vector3(a.spoil[0], 0, a.spoil[1]));
      this.terrain.hf.beginStroke();
      a.start();
      toast('Dig cycle started', 'ok');
    } else {
      a.stop();
    }
    this.renderPanel();
  }

  private exampleTrench() {
    const { cx, cz, houseYaw } = this.machine;
    const ux = Math.cos(houseYaw), uz = Math.sin(houseYaw);
    // along the boom heading, in front of the machine, the way excavators dig toward themselves
    const p0: [number, number] = [cx + ux * 4.2, cz + uz * 4.2];
    const p1: [number, number] = [cx + ux * 7.8, cz + uz * 7.8];
    // 1:1 side slopes (45 deg) stand in this soil model's wet clay (critical angle 62 deg); 1/2:1 (63 deg) would slough
    this.designParams = { ...this.designParams, bottomWidth: 0.9, depth: 1.0, sideSlope: 1 };
    this.setDesign({ p0, p1, bottomWidth: 0.9, depth: 1.0, sideSlope: 1 });
    if (this.mode === 'design') this.renderPanel();
    this.flyTo('trench');
  }

  setDesign(spec: DesignTrench | null) {
    const hf = this.terrain.hf;
    hf.setDesign(spec);
    this.overlays.setDesign(spec);
    this.terrain.uniforms.flags.y = spec && this.gradeOverlay ? 1 : 0;
    this.terrain.designChanged();
    ($('#cardGrade') as HTMLElement).hidden = !spec;
    if (spec) toast(`Design: ${Math.hypot(spec.p1[0] - spec.p0[0], spec.p1[1] - spec.p0[1]).toFixed(1)} m trench, ${spec.depth.toFixed(2)} m deep`, 'ok');
  }
  private reapplyDesign() {
    const s = this.terrain.hf.designSpec;
    if (!s) return;
    const d = this.designParams;
    this.setDesign({ ...s, bottomWidth: d.bottomWidth, depth: d.depth, sideSlope: d.sideSlope });
  }

  private undo() {
    if (this.terrain.hf.undo()) { this.terrain.generated.include({ i0: 0, j0: 0, i1: this.terrain.hf.N - 1, j1: this.terrain.hf.N - 1 }); toast('Undone'); }
    else toast('Nothing to undo', 'warn');
  }
  private resetSite() {
    this.autonomy.stop();
    this.terrain.reset();
    this.spoilPoint = null;
    this.overlays.setSpoil(null);
    toast('Site restored to the capture', 'ok');
  }

  applyEnv() {
    const g = this.env.grading();
    this.terrain.uniforms.grading.copy(g.look);
    this.terrain.uniforms.tint.copy(g.tint);
    this.terrain.refreshLook();
    const [a, b, c] = this.env.sky();
    this.engine.setSky(a, b, c);
  }

  private blindSpotTest() {
    // put a worker 2.4 m from the cab side, inside the swing radius but under the roof lidar's lowest beam
    const { cx, cz, houseYaw, groundY } = this.machine;
    const a = houseYaw + 100 * DEG;
    const x = cx + Math.cos(a) * 3.4, z = cz + Math.sin(a) * 3.4;
    this.worker.place(x, this.terrain.hf.heightAt(x, z), z, a + Math.PI);
    this.worker.group.visible = true;
    void groundY;
    this.renderPanel();
    this.flyTo('blind');
    toast('Worker placed 3.4 m from the swing centre: watch the lidar returns', 'warn', 4200);
  }

  flyTo(which: 'overview' | 'operator' | 'ground' | 'trench' | 'blind') {
    const { cx, cz, groundY, houseYaw } = this.machine;
    const ux = Math.cos(houseYaw), uz = Math.sin(houseYaw);
    let target: THREE.Vector3, pos: THREE.Vector3;
    if (which === 'overview') { target = new THREE.Vector3(cx + 4, groundY - 0.6, cz - 2); pos = new THREE.Vector3(cx + 13, groundY + 8, cz + 9); }
    else if (which === 'operator') { target = new THREE.Vector3(cx + ux * 7, groundY - 1, cz + uz * 7); pos = new THREE.Vector3(cx + ux * 0.6 - uz * 1.4, groundY + 3.4, cz + uz * 0.6 + ux * 1.4); }
    else if (which === 'ground') { target = new THREE.Vector3(cx + ux * 5, groundY - 0.8, cz + uz * 5); pos = new THREE.Vector3(cx + ux * 11 + uz * 4, groundY + 0.4, cz + uz * 11 - ux * 4); }
    else if (which === 'blind') { target = new THREE.Vector3(cx, groundY + 1, cz); pos = new THREE.Vector3(cx - ux * 6 - uz * 7, groundY + 6, cz - uz * 6 + ux * 7); }
    else { const s = this.terrain.hf.designSpec; const mx = s ? (s.p0[0] + s.p1[0]) / 2 : cx, mz = s ? (s.p0[1] + s.p1[1]) / 2 : cz; target = new THREE.Vector3(mx, this.terrain.hf.heightAt(mx, mz) - 0.5, mz); pos = new THREE.Vector3(mx - uz * 8 + ux * 3, target.y + 6.5, mz + ux * 8 + uz * 3); }
    this.tweenCamera(pos, target);
  }

  private tween: { p0: THREE.Vector3; p1: THREE.Vector3; t0: THREE.Vector3; t1: THREE.Vector3; k: number } | null = null;
  private tweenCamera(pos: THREE.Vector3, target: THREE.Vector3) {
    this.tween = { p0: this.engine.camera.position.clone(), p1: pos, t0: this.engine.controls.target.clone(), t1: target, k: 0 };
  }

  sensorPose(which: 'lidar' | 'camera'): [number, number, number] {
    const v = new THREE.Vector3();
    (which === 'lidar' ? this.twin.lidarMount : this.twin.cameraMount).getWorldPosition(v);
    return [v.x, v.y, v.z];
  }

  private async toggleMcap() {
    if (!this.mcap.recording) {
      this.mcap.start(this.data);
      setText($('#mcapLabel'), 'Stop & save');
      $('#btnMcap').classList.add('rec');
      toast('Recording lidar, roof cam, terrain and machine state', 'ok');
    } else {
      setText($('#mcapLabel'), 'Saving…');
      const { bytes, summary } = await this.mcap.stopAsync();
      setText($('#mcapLabel'), 'Record MCAP');
      $('#btnMcap').classList.remove('rec');
      const blob = new Blob([bytes as BlobPart], { type: 'application/octet-stream' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `livedirt-${new Date().toISOString().replace(/[:.]/g, '-')}.mcap`;
      a.click();
      toast(`Saved ${summary}`, 'ok', 4000);
    }
  }

  private neuralModel: import('./sim/neuralSoil').NeuralModel | null = null;
  async setNeuralSoil(on: boolean) {
    const a = this.autonomy;
    if (!on) { a.neural = null; return; }
    if (!this.neuralModel) {
      toast('Loading learned soil model (4.4 MB)…');
      const ort = await import('onnxruntime-web/wasm');
      ort.env.wasm.wasmPaths = import.meta.env.DEV ? 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/' : '/ort/';
      const { loadNeuralModel } = await import('./sim/neuralSoil');
      this.neuralModel = await loadNeuralModel(ort, '/models/soil-v1.onnx', { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      toast('Learned soil ready', 'ok');
    }
    // friction from the selected soil's repose angle (the model was trained on dry sand, mu 0.3-0.7)
    const mu = Math.min(0.7, Math.max(0.3, Math.tan((this.terrain.hf.soil.reposeDeg * Math.PI) / 180)));
    a.neural = { model: this.neuralModel, mu, fMax: 112800 };
  }

  async ensureDetector() {
    if (this.detector.ready) return;
    toast('Loading YOLOX-Nano (3.7 MB)…');
    // production self-hosts the ORT runtime; Vite's dev server will not import modules from /public
    await this.detector.load('/models/yolox_nano.onnx', import.meta.env.DEV ? 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/' : '/ort/');
    toast('Detector ready', 'ok');
  }

  private drawDets(gt?: { x: number; y: number; w: number; h: number } | null) {
    const c = $('#camBoxes') as HTMLCanvasElement;
    if (!c) return;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    const sx = c.width / this.cab.width, sy = c.height / this.cab.height;
    if (gt) { ctx.setLineDash([4, 3]); ctx.strokeStyle = '#3ddc84'; ctx.lineWidth = 1.5; ctx.strokeRect(gt.x * sx, gt.y * sy, gt.w * sx, gt.h * sy); ctx.setLineDash([]); }
    for (const d of this.lastDets) {
      ctx.strokeStyle = '#ff2d7a'; ctx.lineWidth = 2;
      ctx.strokeRect(d.x * sx, d.y * sy, d.w * sx, d.h * sy);
      ctx.fillStyle = '#ff2d7a'; ctx.font = '600 10px Inter, sans-serif';
      const label = `person ${(d.score * 100).toFixed(0)}%`;
      ctx.fillRect(d.x * sx, d.y * sy - 13, ctx.measureText(label).width + 8, 13);
      ctx.fillStyle = '#fff'; ctx.fillText(label, d.x * sx + 4, d.y * sy - 3);
    }
  }

  async runPerceptionSweep() {
    if (this.sweeping) return;
    await this.ensureDetector();
    this.sweeping = true;
    this.cab.paused = true;
    const wasVisible = this.worker.group.visible;
    const wasTwin = this.twinVisible;
    // twin mode: the machine is a mesh, so camera occlusion by boom/arm/bucket can be ray-tested exactly
    this.setTwin(true);
    await new Promise((r) => setTimeout(r, 400));
    const saved = { look: this.env.look, rain: this.env.rain, dust: this.env.dust };
    this.renderPanel();
    const app = this;
    const host: SweepHost = {
      scene: this.engine.scene,
      cabCamera: this.cab.camera,
      cabSize: { w: this.cab.width, h: this.cab.height },
      async renderCab() { for (let i = 0; i < 6; i++) app.env.tick(0.05, app.cab.camera.position); return app.cab.renderNow(app.engine.scene); },
      setCabPose(dyaw: number) {
        const cp = new THREE.Vector3(); app.twin.cameraMount.getWorldPosition(cp);
        app.cab.setPose(cp, app.twin.joints.swing + (Number.isNaN(dyaw) ? 0 : dyaw), -24 * DEG);
      },
      placeWorker(x, z, face) { app.worker.place(x, app.terrain.hf.heightAt(x, z), z, face); app.worker.group.visible = true; },
      hideWorker() { app.worker.group.visible = false; },
      workerCapsule: () => this.worker.capsule(),
      setCondition: (c: Condition) => { this.env.look = c.look; this.env.rain = c.rain; this.env.dust = c.dust; this.terrain.uniforms.flags.w = c.rain; this.env.dustCenter.set(this.machine.cx + Math.cos(this.machine.houseYaw) * 8, this.machine.groundY - 1, this.machine.cz + Math.sin(this.machine.houseYaw) * 8); this.applyEnv(); },
      sensorRay: this.terrain.sensorRay,
      lidarPoses: () => ({ roof: this.sensorPose('lidar'), front: (() => { const v = new THREE.Vector3(); this.twin.frontLidarMount.getWorldPosition(v); return [v.x, v.y, v.z] as [number, number, number]; })(), yaw: this.twin.joints.swing }),
      heightAt: (x, z) => this.terrain.hf.heightAt(x, z),
      occluders: [this.twin.root],
    };
    try {
      this.sweep = await runSweep(host, this.detector, 'YOLOX-Nano (COCO, 416, ONNX, WASM)', (f, msg) => setText($('#sweepMsg'), `Running ${(f * 100).toFixed(0)}% · ${msg}`));
      (window as unknown as { __sweep: SweepResult }).__sweep = this.sweep;
      toast('Perception sweep complete', 'ok');
    } catch (e) {
      console.error(e);
      toast('Sweep failed: ' + (e as Error).message, 'warn', 5000);
    } finally {
      this.sweeping = false;
      this.cab.paused = false;
      Object.assign(this.env, saved);
      this.terrain.uniforms.flags.w = saved.rain;
      this.applyEnv();
      this.worker.group.visible = wasVisible;
      this.setTwin(wasTwin);
      this.renderPanel();
    }
  }

  // ------------------------------------------------------------------ input
  private pick(clientX: number, clientY: number) {
    const r = this.engine.canvas.getBoundingClientRect();
    this.ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.rc.setFromCamera(this.ndc, this.engine.camera);
    const o = this.rc.ray.origin, d = this.rc.ray.direction;
    const h = this.terrain.ray.cast(o.x, o.y, o.z, d.x, d.y, d.z, 250);
    return h ? new THREE.Vector3(h.x, h.y, h.z) : null;
  }

  private bindInput() {
    const cv = this.engine.canvas;
    cv.addEventListener('pointermove', (e) => {
      this.cursor = this.pick(e.clientX, e.clientY);
      if (this.mode === 'dig' && this.cursor) this.overlays.showBrush(this.cursor.x, this.cursor.z, this.brush);
      if (this.mode === 'design' && this.designDraft && this.cursor) {
        this.overlays.setDesign({ p0: [this.designDraft.x, this.designDraft.z], p1: [this.cursor.x, this.cursor.z], bottomWidth: this.designParams.bottomWidth, depth: this.designParams.depth, sideSlope: this.designParams.sideSlope });
      }
    });
    cv.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const p = this.pick(e.clientX, e.clientY);
      if (!p) return;
      if (this.placingWorker) {
        this.worker.place(p.x, p.y, p.z, Math.atan2(this.machine.cz - p.z, this.machine.cx - p.x));
        this.worker.group.visible = true;
        this.placingWorker = false;
        $('#stage').classList.remove('place');
        this.renderPanel();
        return;
      }
      if (this.mode === 'dig') {
        if (e.shiftKey || this.spoilMode === 'pick' && e.altKey) { this.spoilPoint = p; this.spoilMode = 'pick'; this.overlays.setSpoil(p); toast('Spoil point set'); return; }
        this.digging = true;
        this.lastDigPos = new THREE.Vector2(p.x, p.z);
        this.terrain.hf.beginStroke();
        cv.setPointerCapture(e.pointerId);
      } else if (this.mode === 'design') {
        if (!this.designDraft) { this.designDraft = p; toast('Now click the trench end'); }
        else {
          const spec: DesignTrench = { p0: [this.designDraft.x, this.designDraft.z], p1: [p.x, p.z], ...this.designParams };
          if (Math.hypot(spec.p1[0] - spec.p0[0], spec.p1[1] - spec.p0[1]) < 1) { toast('Trench too short', 'warn'); return; }
          this.designDraft = null;
          this.setDesign(spec);
        }
      }
    });
    cv.addEventListener('dblclick', (e) => {
      const p = this.pick(e.clientX, e.clientY);
      if (p && this.mode !== 'design') this.tweenCamera(this.engine.camera.position.clone().sub(this.engine.controls.target).multiplyScalar(0.6).add(p), p);
    });
    window.addEventListener('pointerup', () => { this.digging = false; this.lastDigPos = null; });
    window.addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'SELECT') return;
      const modes: Mode[] = ['explore', 'dig', 'design', 'scenario', 'auto'];
      if (e.key >= '1' && e.key <= '5') this.setMode(modes[+e.key - 1]);
      if (e.key === 'z' || e.key === 'Z') this.undo();
      if (e.key === '[') { this.brush = Math.max(0.3, this.brush - 0.1); if (this.mode === 'dig') this.renderPanel(); }
      if (e.key === ']') { this.brush = Math.min(1.6, this.brush + 0.1); if (this.mode === 'dig') this.renderPanel(); }
      if (e.key === 'Escape') { this.designDraft = null; this.overlays.setDesign(this.terrain.hf.designSpec); this.toggleAbout(false); }
      if (e.key === 'g') this.terrain.setGeneratedOverlay(this.terrain.uniforms.flags.x < 0.5);
    });
  }

  // ------------------------------------------------------------------ frame
  private frame(dt: number, t: number) {
    const hf = this.terrain.hf;
    // camera tween
    if (this.tween) {
      const tw = this.tween;
      tw.k = Math.min(1, tw.k + dt / 1.1);
      const e = tw.k < 0.5 ? 4 * tw.k ** 3 : 1 - Math.pow(-2 * tw.k + 2, 3) / 2;
      this.engine.camera.position.lerpVectors(tw.p0, tw.p1, e);
      this.engine.controls.target.lerpVectors(tw.t0, tw.t1, e);
      if (tw.k >= 1) this.tween = null;
    }
    // manual digging
    if (this.mode === 'dig' && this.digging && this.cursor) {
      const p2 = new THREE.Vector2(this.cursor.x, this.cursor.z);
      if (this.lastDigPos && p2.distanceTo(this.lastDigPos) > 0.05) this.strokeDir.copy(p2).sub(this.lastDigPos).normalize();
      this.lastDigPos = p2;
      const r = hf.dig(this.cursor.x, this.cursor.z, this.brush, this.depthRate * dt, this.gradeAssist && !!hf.designSpec);
      if (r.bank + r.loose > 0) this.hasDug = true;
      // spoil: auto = beside the stroke on the side away from the machine, with setback
      let sp = this.spoilPoint;
      if (this.spoilMode === 'auto' || !sp) {
        const side = new THREE.Vector2(-this.strokeDir.y, this.strokeDir.x);
        const toMachine = new THREE.Vector2(this.machine.cx - this.cursor.x, this.machine.cz - this.cursor.z);
        if (side.dot(toMachine) > 0) side.negate();
        const off = this.brush + 0.61 + 1.1;
        sp = new THREE.Vector3(this.cursor.x + side.x * off, 0, this.cursor.z + side.y * off);
      }
      hf.dump(sp.x, sp.z, hf.ledger.carried, 0.9);
      this.env.dustCenter.set(this.cursor.x, this.cursor.y, this.cursor.z);
    }
    // autonomy
    const a = this.autonomy;
    // fixed-timestep physics (same 0.05 s step as app/bench/autonomy.ts) so runs do not depend on frame rate;
    // at most 10 catch-up steps per frame
    this.physAcc = Math.min(this.physAcc + dt, 10 * PHYS_DT);
    while (this.physAcc >= PHYS_DT - 1e-9) {
      this.physAcc -= PHYS_DT;
      if (a.phase !== 'idle' && a.phase !== 'done' && a.phase !== 'unreachable') {
        const j = a.step(PHYS_DT, this.personDetectedInZone);
        this.twin.setJoints(j);
        this.env.dustCenter.copy(this.twin.tipWorld());
      }
      if (hf.isSettling()) hf.relax(3);
    }
    this.twin.tick(t);
    this.terrain.update(dt, false);
    // sensors follow the machine's sensor mounts (valid whether or not the twin is shown)
    this.twin.root.updateMatrixWorld(true);
    const lp = new THREE.Vector3();
    this.twin.lidarMount.getWorldPosition(lp);
    const capsules: Capsule[] = this.worker.group.visible ? [this.worker.capsule()] : [];
    this.surfaceTimer -= dt;
    if (this.surfaceTimer <= 0 && (hf.peekDirty() || this.hasDug || a.phase !== 'idle')) {
      this.lidar.updateSurface(this.terrain.sensorRay.surf);
      this.lidarFront.updateSurface(this.terrain.sensorRay.surf);
      this.surfaceTimer = 0.25;
      this.hasDug = false;
    }
    this.lidar.tick(dt, t, { x: lp.x, y: lp.y, z: lp.z, yaw: this.twin.joints.swing }, capsules, this.env.dust, this.env.rain);
    const fp = new THREE.Vector3();
    this.twin.frontLidarMount.getWorldPosition(fp);
    this.lidarFront.tick(dt, t, { x: fp.x, y: fp.y, z: fp.z, yaw: this.twin.joints.swing, pitch: 35 * DEG }, capsules, this.env.dust, this.env.rain);
    const cp = new THREE.Vector3();
    this.twin.cameraMount.getWorldPosition(cp);
    if (!this.sweeping) this.cab.setPose(cp, this.twin.joints.swing + (this.cabYawOverride ?? 0), -24 * DEG);
    const ph = $('#camPh');
    if (ph) { const r = ph.getBoundingClientRect(); const cr = this.engine.canvas.getBoundingClientRect(); this.cab.rect = { x: r.left - cr.left, y: r.top - cr.top, w: r.width, h: r.height }; }
    void this.cab.tick(dt, this.engine.scene);
    this.env.tick(dt, this.sweeping ? this.cab.camera.position : this.engine.camera.position);
    // live detector on the roof cam
    this.detTimer -= dt;
    if (this.detectorOn && !this.sweeping && !this.detBusy && this.detTimer <= 0 && this.detector.ready && this.cab.frames > 0) {
      this.detBusy = true; this.detTimer = 0.4;
      void this.cab.readPixels().then((px) => this.detector.detect(px, this.cab.width, this.cab.height)).then((d) => { this.lastDets = d; this.drawDets(); }).finally(() => (this.detBusy = false));
    }
    // safety monitor: swing exclusion zone = current front reach (+1 m) or tail swing (+0.6 m)
    const fk = forwardPlanar(DX225_GEOMETRY, this.twin.joints);
    this.zoneR = Math.max(fk.tip[0] + 1.0, DX225_GEOMETRY.boomFootR + 2.909 + 0.6);
    if (this.mode === 'auto' || this.worker.group.visible) this.overlays.showZone(this.machine.cx, this.machine.cz, this.zoneR); else this.overlays.hideZone();
    const dW = this.worker.group.visible ? Math.hypot(this.worker.x - this.machine.cx, this.worker.z - this.machine.cz) : Infinity;
    this.personTruthInZone = dW < this.zoneR;
    this.personDetectedInZone = this.personTruthInZone && this.personReturns >= 4;
    // MCAP
    if (this.mcap.recording) this.mcap.tick(t, this);
    this.telemetryTimer -= dt;
    if (this.telemetryTimer <= 0) { this.updateTelemetry(dW); this.telemetryTimer = 0.15; }
  }

  private personRoof = 0;
  private personFront = 0;
  private onScan(s: ScanResult, src: 'roof' | 'front') {
    if (src === 'roof') this.lastScan = s; else this.lastFront = s;
    let person = 0, terrain = 0, obstacle = 0, dust = 0, rain = 0;
    for (let i = 0; i < s.count; i++) {
      const l = s.label[i];
      if (l === PointLabel.Person) person++;
      else if (l === PointLabel.Terrain) terrain++;
      else if (l === PointLabel.Obstacle) obstacle++;
      else if (l === PointLabel.Dust) dust++;
      else rain++;
    }
    if (src === 'roof') this.personRoof = person; else this.personFront = person;
    this.personReturns = this.personRoof + this.personFront;
    if (this.mcap.recording) this.mcap.addScan(s, src);
    if (src !== this.lidarView) return;
    if (this.showLidarInScene) this.lidarPts.update(s, this.machine.groundY);
    this.lidarPanel.draw(s, this.twin.joints.swing, this.machine.groundY);
    setText($('#lmT'), fmt.int(terrain)); setText($('#lmO'), fmt.int(obstacle)); setText($('#lmP'), fmt.int(person)); setText($('#lmD'), fmt.int(dust)); setText($('#lmR'), fmt.int(rain));
    setText($('#lPts'), fmt.int(s.count));
    setText($('#lDrop'), `${((s.dropped / Math.max(1, s.rays - s.noHit)) * 100).toFixed(1)}%`);
    setText($('#lAtm'), fmt.int(dust + rain));
    setText($('#lMs'), `${s.ms.toFixed(1)} ms`);
  }

  private updateTelemetry(dW: number) {
    const hf = this.terrain.hf, L = hf.ledger;
    $('#tBank').innerHTML = `${fmt.m3(L.bankCut)}<small>m³ bank cut</small>`;
    setText($('#tLoose'), `${fmt.m3(L.bankCut * hf.soil.swell + L.sloughed * hf.soil.swell)} m³`);
    setText($('#tSlough'), `${fmt.m3(L.sloughed)} m³`);
    setText($('#tField'), `${fmt.m3(L.dumped)} m³`);
    setText($('#tMass'), `${((L.bankCut + L.sloughed) * hf.soil.density).toFixed(1)} t`);
    setText($('#soilPill'), hf.soil.label);
    if (hf.designSpec) {
      const g = hf.gradeStats(this.designParams.tol);
      $('#gOn').innerHTML = `${g.onGradePct.toFixed(0)}%<small>on grade</small>`;
      ($('#gBar') as HTMLElement).style.width = `${g.onGradePct}%`;
      setText($('#gRem'), `${fmt.m3(g.remainingCut)} m³`);
      setText($('#gOver'), `${fmt.m3(g.overDig)} m³`);
      setText($('#gUnpl'), `${fmt.m3(g.unplannedCut)} m³`);
      setText($('#gMae'), fmt.cm(g.meanAbsError));
      setText($('#gTol'), (this.designParams.tol * 100).toFixed(0));
      const pill = $('#gradePill');
      pill.className = 'pill ' + (g.overDig > 0.05 ? 'warn' : g.onGradePct > 90 ? 'ok' : '');
      setText(pill, g.onGradePct > 90 ? 'to grade' : g.overDig > 0.05 ? 'over-dig' : 'cutting');
    }
    // safety
    const sp = $('#safePill');
    setText($('#sZone'), `${this.zoneR.toFixed(1)} m`);
    setText($('#sDist'), Number.isFinite(dW) ? `${dW.toFixed(1)} m` : 'none');
    setText($('#sRet'), fmt.int(this.worker.group.visible ? this.personReturns : 0));
    if (this.personDetectedInZone) { sp.className = 'pill bad'; sp.innerHTML = '<i></i>STOP · person in zone'; }
    else if (this.personTruthInZone) { sp.className = 'pill bad'; sp.innerHTML = '<i></i>Undetected person in zone'; }
    else if (this.worker.group.visible) { sp.className = 'pill ok'; sp.innerHTML = '<i></i>Clear · tracking'; }
    else { sp.className = 'pill ok'; sp.innerHTML = '<i></i>Clear'; }
    setText($('#cMs'), `${this.cab.lastRenderMs.toFixed(0)} ms`);
    // status bar
    setText($('#stFps'), this.engine.fps.toFixed(0));
    setText($('#stSplats'), `${(this.terrain.splatCount / 1e6).toFixed(2)}M`);
    setText($('#stGen'), fmt.int(this.terrain.generated.count));
    setText($('#stSim'), hf.isSettling() ? 'soil settling…' : 'settled');
    // autonomy panel
    if (this.mode === 'auto') {
      const a = this.autonomy;
      setText($('#aPhase'), a.phase.replace('-', ' '));
      setText($('#aBite'), `${Math.min(a.biteIndex, a.bitesTotal)} / ${a.bitesTotal}`);
      setText($('#aPass'), String(a.stats.passes));
      setText($('#aLoad'), `${a.bucketLoad.toFixed(2)} m³`);
      const bar = $('#aLoadBar'); if (bar) bar.style.width = `${Math.min(100, (a.bucketLoad / a.bucketCapacity) * 100)}%`;
      setText($('#aCyc'), `${a.stats.cycles} · ${a.stats.avgCycleS.toFixed(1)} s`);
      setText($('#aFill'), `${(a.stats.avgFill * 100).toFixed(0)}%`);
      setText($('#aProd'), `${a.stats.productivity.toFixed(0)} m³/h`);
      setText($('#aForce'), a.neural ? `${(a.soilForce[0] / 1000).toFixed(0)} kN${a.reliefScale < 0.99 ? ' · relief' : ''}` : 'kinematic cut');
      const sv = a.stats.surveys[a.stats.surveys.length - 1];
      setText($('#aSurv'), !sv ? 'n/a' : sv.coverage === 0 ? 'no coverage' : `${(sv.lidarMeanAbove * 100).toFixed(1)} vs ${(sv.truthMeanAbove * 100).toFixed(1)} cm · ${(sv.coverage * 100).toFixed(0)}%`);
      const log = $('#aLog'); if (log) { const txt = a.log.slice(-14).join('\n'); if (log.textContent !== txt) { log.textContent = txt; log.scrollTop = log.scrollHeight; } }
      const run = $('#aRun'); if (run) { const idle = a.phase === 'idle' || a.phase === 'done'; const want = idle ? 'Run dig cycle' : 'Pause'; if (!run.textContent?.includes(want)) run.innerHTML = (idle ? icons.play : icons.pause) + want; }
    }
  }
}

function sweepTable(r: SweepResult) {
  const cell = (v: number | null) => (v === null ? '<td class="na">n/a</td>' : `<td style="background:rgba(61,220,132,${(0.08 + v * 0.42).toFixed(2)})">${Math.round(v * 100)}</td>`);
  const rows = CONDITIONS.map((c) => {
    const cam = DISTANCES.map((d) => cell(r.summary.find((s) => s.condition === c.id && s.distance === d)!.camRecall)).join('');
    const lid = DISTANCES.map((d) => cell(r.summary.find((s) => s.condition === c.id && s.distance === d)!.lidarRecall)).join('');
    const fp = r.negatives.find((n) => n.condition === c.id)!;
    return `<tr><th rowspan="2">${c.label}<small>${fp.falsePositives} FP / ${fp.frames} empty frames</small></th><td class="k">cam</td>${cam}</tr><tr><td class="k">lidar</td>${lid}</tr>`;
  }).join('');
  return `<table class="sweep"><thead><tr><th></th><th></th>${DISTANCES.map((d) => `<th>${d}m</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table><div class="hint">Recall % (camera: YOLOX person, IoU ≥ 0.3, score ≥ ${r.scoreThreshold}; lidar: ≥ ${r.minLidarReturns} returns). Inference ${r.inferMsMean.toFixed(0)} ms/frame.</div>`;
}

const MODE_LABEL: Record<Mode, string> = { explore: 'Explore', dig: 'Dig', design: 'Design', scenario: 'Scenario', auto: 'Autonomy' };
export const LOGO = '<svg viewBox="0 0 24 24" fill="none"><path d="M3 18 9 11l3.5 3.5L16 9l5 9z" fill="#fff"/><circle cx="17" cy="5.5" r="1.8" fill="#ffe3c7"/></svg>';

function aboutHtml(d: SceneData) {
  const s = d.meta.source;
  return `
  <button class="x" aria-label="Close">${icons.close}</button>
  <h2>What's real in LiveDirt</h2>
  <p class="lead">Every neural simulator assumes the world holds still. An excavator's whole job is to change it. LiveDirt makes a real reconstructed site diggable and is explicit about which parts are learned, simulated, generated or approximated.</p>
  <h4><i style="background:#ff7a1a"></i>Learned from real footage</h4>
  <ul>
    <li><b>The site itself</b>: ${(d.meta.splats / 1e6).toFixed(2)}M 3D Gaussians reconstructed from 230 frames of 4K drone video (alignment in RealityScan, 3DGS in LichtFeld, by ${s.author}). Downsampled to 1.9M for the web.</li>
    <li><b>Terrain and obstacles</b>: the ground heightfield (10 cm grid), the obstacle mask and the soil colour palette are all extracted from those splats by the pipeline in this repo.</li>
    <li><b>Metric scale</b>: ×${d.meta.metricScale}, estimated from the excavator's track length and cab height against the Develon DX225LC-7 spec sheet (±10%).</li>
  </ul>
  <h4><i style="background:#3ddc84"></i>Simulated (first-principles models)</h4>
  <ul>
    <li><b>Earthwork</b>: two-layer bank/loose heightfield, soil swell factors, angle of repose, wall sloughing past a critical angle, OSHA-style spoil setback, a conserved volume ledger (unit-tested).</li>
    <li><b>Grade control</b>: design trench surface, per-cell error vs tolerance, remaining cut, over-dig.</li>
    <li><b>Machine</b>: DX225LC-7 linkage kinematics (boom 5.7 m, arm 2.9 m), analytic IK, rate-limited joints, a trench-to-grade dig cycle that surveys its work with its own lidar.</li>
    <li><b>Lidar</b>: 64-beam, ±22.5° spinning sensor ray-cast against the evolving terrain with range noise, incidence/range/albedo dropout, dust and rain early returns, per-point ground-truth labels.</li>
    <li><b>Roof camera</b>: a second splat renderer with its own sort order, from the machine's sensor mount.</li>
  </ul>
  <h4><i style="background:#a752ff"></i>Generated, not learned</h4>
  <ul>
    <li><b>Soil the camera never saw</b> (trench walls, floors, spoil) is procedural: clods, crumbs and tooth streaks coloured from the scene's own palette. Toggle "generated soil" to see exactly where. A research version would replace this with a learned (diffusion) prior.</li>
  </ul>
  <h4><i style="background:#ffb020"></i>Approximated</h4>
  <ul>
    <li><b>Lighting and weather</b>: colour grading of baked splat colours plus particles. Not relighting.</li>
    <li><b>The articulated machine</b> is a mesh twin, not splats: the "splat world + mesh machine" hybrid.</li>
    <li><b>Obstacle detection</b> in the pipeline is a colour + height heuristic.</li>
  </ul>
  <h4><i style="background:#6f6861"></i>Not modelled</h4>
  <ul>
    <li>Soil continuum mechanics (no MPM/DEM), water, bucket-soil forces, hydraulics dynamics, lidar motion distortion, the machine's self-occlusion in the lidar, view-dependent splat colour (SH0 only).</li>
  </ul>
  <h4>Credits</h4>
  <table>
    <tr><td>3DGS scene</td><td><a href="${s.url}" target="_blank" rel="noopener">${s.title}</a> by ${s.author}, ${s.license}</td></tr>
    <tr><td>Source footage</td><td>${s.footage}</td></tr>
    <tr><td>Renderer</td><td><a href="https://sparkjs.dev" target="_blank" rel="noopener">Spark</a> (World Labs) on three.js</td></tr>
    <tr><td>Machine dimensions</td><td>Develon DX225LC-7 spec sheet</td></tr>
    <tr><td>Built by</td><td>Mohith Gajjela · <a href="${REPO_URL}" target="_blank" rel="noopener">source &amp; writeup</a></td></tr>
  </table>`;
}
