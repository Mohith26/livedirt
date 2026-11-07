import './ui/styles.css';
import { Engine } from './core/engine';
import { loadSceneData } from './data/scene';
import { TerrainSystem } from './core/terrainSystem';
import { App, LOGO } from './app';

const root = document.getElementById('app')!;
root.innerHTML = `<div class="stage" id="stage"></div>
  <div class="loader" id="loader"><div class="inner">
    <div class="mark">${LOGO}</div>
    <h1>LiveDirt</h1>
    <p>A real construction site, reconstructed from drone footage as 3D Gaussians, made diggable. Terrain physics, grade control, simulated lidar and a roof camera, in your browser.</p>
    <div class="track"><i id="ldBar"></i></div>
    <div class="lab" id="ldLab">Preparing…</div>
    <div class="facts"><div><b>230</b>4K drone frames</div><div><b>1.9M</b>Gaussians</div><div><b>10 cm</b>sim grid</div><div><b>64</b>lidar beams</div></div>
    <div class="err" id="ldErr"></div>
  </div></div>`;

const bar = document.getElementById('ldBar')!;
const lab = document.getElementById('ldLab')!;
const setP = (f: number, l: string) => { bar.style.width = `${Math.round(f * 100)}%`; lab.textContent = `${l} · ${Math.round(f * 100)}%`; };

function webglOk() {
  try { const c = document.createElement('canvas'); return !!c.getContext('webgl2'); } catch { return false; }
}

async function boot() {
  if (!webglOk()) throw new Error('This demo needs WebGL 2. Try a recent Chrome, Edge, Firefox or Safari.');
  const engine = new Engine(document.getElementById('stage')!);
  setP(0.02, 'Loading terrain model');
  const data = await loadSceneData('/scene');
  const terrain = new TerrainSystem(data, { zoneRadius: 11.5 });
  engine.scene.add(terrain.generated.mesh);
  terrain.onTileReady = (m) => engine.scene.add(m);
  const app = new App(engine, data, terrain);
  (window as unknown as { __ld: unknown }).__ld = { engine, terrain, app };
  app.flyTo('overview');
  engine.start();
  await terrain.loadTiles('/scene', (f, l) => setP(0.05 + f * 0.95, `Streaming ${l}`));
  setP(1, 'Ready');
  setTimeout(() => document.getElementById('loader')!.classList.add('done'), 250);
}

boot().catch((e) => {
  console.error(e);
  document.getElementById('ldErr')!.textContent = e instanceof Error ? e.message : String(e);
});
