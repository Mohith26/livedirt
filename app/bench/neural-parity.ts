/**
 * Parity check: the browser port (src/sim/neuralSoil.ts + onnxruntime) must reproduce the PyTorch surrogate.
 *   npx tsx bench/neural-parity.ts <model.onnx> <fixture.json>
 */
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-node';
import { buildFeatures, loadNeuralModel, NX, NY } from '../src/sim/neuralSoil';

const [modelPath, fixPath] = process.argv.slice(2);
const fx = JSON.parse(readFileSync(fixPath, 'utf8'));
if (fx.nx !== NX || fx.ny !== NY) throw new Error(`grid mismatch ${fx.nx}x${fx.ny} vs ${NX}x${NY}`);
const nan = (a: (number | null)[]) => Float32Array.from(a.map((v) => (v === null ? NaN : v)));
const feats = buildFeatures(Float32Array.from(fx.H), nan(fx.T0), nan(fx.T1), fx.mu);
let fErr = 0;
for (let k = 0; k < feats.length; k++) fErr = Math.max(fErr, Math.abs(feats[k] - fx.x[k]));
const model = await loadNeuralModel(ort, modelPath);
const t0 = performance.now();
const { dh, force } = await model.step(feats);
const ms = performance.now() - t0;
let dErr = 0, dMax = 0;
for (let k = 0; k < dh.length; k++) { dErr = Math.max(dErr, Math.abs(dh[k] - fx.dh[k])); dMax = Math.max(dMax, Math.abs(fx.dh[k])); }
const fRel = Math.max(...force.map((f, i) => Math.abs(f - fx.force[i]) / (Math.abs(fx.force[i]) + 1)));
console.log(JSON.stringify({ feature_max_abs_err: fErr, dh_max_abs_err_m: dErr, dh_max_m: dMax, force_max_rel_err: fRel, ms_first_step: ms }));
if (fErr > 1e-5 || dErr > 1e-5 || fRel > 1e-4) { console.error('PARITY FAIL'); process.exit(1); }
console.log('PARITY OK');
