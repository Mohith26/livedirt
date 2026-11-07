# LiveDirt

**A neural job site you can dig into.** Live demo: **[livedirt.mohithgajjela.com](https://livedirt.mohithgajjela.com)**

Neural reconstruction (NeRF, 3D Gaussian splatting) is becoming the default way to build photoreal simulation from real logs. Almost every neural simulator assumes the world holds still: cars and people move, the ground does not. An excavator's entire job is to change the ground.

LiveDirt takes a real construction site, reconstructed from drone video as 1.9M 3D Gaussians, and makes it diggable. Under the hood is a small but complete autonomy stack running in the browser:

- **Captured world.** A real UK site (drone footage, 3DGS by dok11, CC BY 4.0), levelled, scaled to metres and split for streaming. Rendered with [Spark](https://sparkjs.dev) on three.js.
- **Earthwork physics on that world.** Two-layer bank/loose heightfield on a 10 cm grid, soil swell factors, angle of repose, wall sloughing past a critical angle, OSHA-style spoil setback, a conserved volume ledger. Splats are deformed on the GPU by the sim every frame.
- **Ground the camera never saw.** Trench walls, floors and spoil are filled with generated splats coloured from the scene's own soil palette. One toggle highlights everything that is generated rather than captured.
- **Grade control.** Draw a design trench; every cell is scored against it live (on-grade %, remaining cut, over-dig).
- **A machine twin and an autonomy loop.** DX225LC-7-dimensioned linkage, analytic IK, rate-limited joints and a trench-to-grade dig cycle that surveys its own work with its own simulated lidar before it calls the job done.
- **Sensors.** Two 64-beam lidars (360° roof unit, front unit tilted 35°) with noise, dropout, dust and rain returns and per-point labels, plus a roof camera rendered from the splats by a second renderer.
- **Perception and safety.** YOLOX person detection on the roof-cam splat renders, a lidar swing-zone safety monitor, a blind-spot test, and a sweep that measures camera vs lidar recall across distance and weather.
- **Data out.** Record a session to **MCAP** with Foxglove schemas (point clouds, JPEG frames, TF, terrain grid, joints, earthwork ledger).

## Results (all measured, reproducible)

Full tables and method: [docs/RESULTS.md](docs/RESULTS.md). Raw outputs: [`app/bench/results/`](app/bench/results).

**Autonomous trenching on the captured terrain** (headless benchmark, `npm run bench`; 3.6 m trench, 0.9 m bottom, 1.0 m deep):

| Soil / side slope | On grade (±3 cm, truth) | Mean abs error | Over-dig | Lidar verdict | Lidar vs truth (mean grade) |
|---|---|---|---|---|---|
| Wet clay, 1:1 (stable) | **99.1%** | 0.07 cm | 0.000 m³ | verified, 1 pass | 0.06 vs 0.03 cm |
| Wet clay, ½:1 (63°, past the 62° critical angle) | 94.2% | 0.62 cm | 0.017 m³ | not verified (74% in tol) | -0.40 vs -0.94 cm |
| Sandy gravel, 1:1 (45°, past the 40° critical angle) | 34.5% | 5.34 cm | 0.331 m³ | not verified (32% in tol) | -6.28 vs -6.57 cm |

Cycle times average 19 s, consistent with a 22 t excavator. Designing a slope steeper than the soil can hold makes the walls slough and the controller refuses to sign off, which is the behaviour you want.

**Perception sweep** (in-app, 4 conditions × 8 distances × 4 bearings, twin mode with exact occlusion):

- Camera (YOLOX-Nano on splat renders): 0% recall at 4 m in every condition (a person seen almost top-down from a 5 m mast), peaking at 100% around 10 to 12 m in clear conditions, falling off below ~24 px of person height. Heavy dust takes camera recall to 0%.
- Lidar (two units, ≥5 person returns): 100% at every tested distance and condition. The level roof unit alone is blind inside ~6 m; the tilted front unit covers it.
- Blind-spot test: a worker 3.4 m from the swing centre, beside the cab, gets **0** lidar returns from either unit. The monitor reports "undetected person in zone".

**Own reconstruction from the raw footage** (`pipeline/train/`): from the original 4K drone video, pycolmap registered 200 / 200 frames (0.46 px reprojection error); Brush trained 7000 steps in 337 s on a MacBook. On 25 held-out frames: **PSNR 23.5 dB, SSIM 0.723, LPIPS 0.330**. The articulated dump truck driving through the footage renders as a smeared ghost, a live example of the static-scene assumption breaking on a real job site.

## What's real

| | |
|---|---|
| **Learned from footage** | site appearance and geometry (3DGS), the terrain heightfield, obstacle mask and soil palette extracted from it |
| **Simulated** | earthwork, grade, kinematics, dig cycle, lidar, roof camera, safety monitor |
| **Generated, not learned** | soil on surfaces the camera never saw (purple in the overlay) |
| **Approximated** | lighting/weather (colour grade + particles, not relighting); the machine is a mesh twin; metric scale ±10% |
| **Not modelled** | soil continuum mechanics, water, bucket forces, hydraulics, lidar motion distortion, machine self-occlusion in the lidar |

## Run it

```bash
cd app
npm install
npm run dev        # http://127.0.0.1:5199
npm test           # earthwork, raycaster, lidar, kinematics
npm run bench -- wet-clay 1    # headless autonomy benchmark on the captured terrain
npm run build
```

Rebuild the scene from the public source capture: `pipeline/build.sh` (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)). Training a splat from the raw footage with held-out evaluation: [`pipeline/train/`](pipeline/train) (`bash pipeline/train/run_all.sh`).

## Controls

`1`-`5` modes · left-drag orbit (or dig in Dig mode) · right-drag orbit while digging · `[` `]` bucket size · `Z` undo · `G` generated-soil overlay

## Credits

- 3DGS scene: [Muddy Excavator Site](https://superspl.at/scene/ded12920) by dok11, CC BY 4.0, from footage by David Pickup (Pexels).
- Renderer: [Spark](https://sparkjs.dev) (World Labs), three.js.
- Detector: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Megvii), Apache-2.0, via onnxruntime-web.
- Machine dimensions: Develon DX225LC-7 spec sheet.

Code: MIT. Scene data: CC BY 4.0 (see LICENSE).
