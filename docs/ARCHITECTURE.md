# Architecture

```
pipeline/ (Node, offline)                       app/ (TypeScript, browser)
  build.sh                                         core/engine.ts          renderer, camera, controls, frame loop
   ├ download SOG (CC BY scene)                    core/terrainSystem.ts   sim <-> GPU sync, splat tiles, generated soil
   ├ splat-transform -> PLY                        sim/heightfield.ts      earthwork model (bank/loose), grade, ledger, undo
   ├ js/build-scene.mjs                            sim/heightRay.ts        ray vs heightfield (max-mip pyramid)
   │   level (RANSAC), scale, crop, clean          render/deform.ts        Spark dyno GLSL: per-splat deformation + recolour
   │   js/terrain.mjs: H0, T, mask, palette        render/generatedSoil.ts procedural splats for never-seen surfaces
   │   albedo bake                                 machine/kinematics.ts   DX225LC-7 linkage FK/IK
   ├ js/split.mjs: inner/outer tiles               machine/excavatorModel  mesh twin + sensor mounts
   └ splat-transform -> inner.sog, outer.sog       machine/autonomy.ts     trench-to-grade dig cycle + lidar survey
  js/view*.mjs, measure.mjs: QA renders            sensors/lidarModel.ts   64-beam lidar model (worker: lidar.worker.ts)
  train/: own reconstruction (pycolmap+Brush)      sensors/cabCamera.ts    second SparkRenderer -> offscreen target
                                                   perception/             YOLOX (onnxruntime-web) + perception sweep
                                                   export/mcap.ts          MCAP writer, Foxglove schemas
                                                   app.ts                  modes, tools, UI, safety monitor, telemetry
```

## Scene pipeline

1. **Decode.** The scene is distributed as PlayCanvas SOG (WebP-packed means/quats/scales/colour codebooks). `splat-transform` decodes it to a float PLY; SH bands above 0 are dropped (the capture is overcast and view-dependence adds little).
2. **Level.** RANSAC over opaque, small splats finds the dominant ground plane (normal ≈ -Y, as usual for COLMAP-style frames); least-squares refines it. Splats are rotated so the plane normal is +Y and translated so the plane is y = 0.
3. **Scale.** No metric scale ships with a monocular capture. The excavator's track length and height over cab are measured in orthographic renders of the levelled cloud and compared with the Develon DX225LC-7 spec sheet (track 4.445 m, height 2.98 m), giving 0.87 m per unit, about ±10%.
4. **Terrain products** (10 cm grid, 48 m square):
   - `T`: per-cell 85th-percentile height of opaque splats (top surface, includes objects).
   - Object mask: cells with "machine evidence" (saturated orange, bright panels, blue) raised above an opening-filtered ground, closed, grown 0.5 m, small blobs removed. A pure geometric approach flagged 45% of the site, including the whole spoil mound; colour evidence brought it to ~10% and kept the mound diggable.
   - `H0`: `T` outside objects, inpainted underneath them.
   - Soil palette: colour statistics of terrain-bound splats bucketed by slope and height.
   - Albedo: top-down colour map (used for lidar intensity/dropout).
5. **Clean.** Splats more than 35 cm below `H0` outside objects are reconstruction noise that would show through when digging; they are removed.
6. **Tile.** Inner dig zone (r < 13 m, 1.31M splats, full detail) and an outer ring (adaptive decimation to ~590k). Both are re-encoded as SOG and stay under the 25 MiB static-asset file limit; the inner tile loads first.
7. **Machine pose.** Swing centre, track heading and house heading are hand-measured from orthographic renders (`js/view-raised.mjs`) and stored in `machine.json` with their uncertainty.

## Earthwork model

`H = B + L`: bank (in-situ) surface plus loose thickness. The tool removes loose first, then bank; bank volume becomes loose volume times the soil's swell factor. Each step relaxes the active region on an 8-neighbourhood with true diagonal distances (4-neighbour flow produced square piles):

- loose material flows wherever the slope exceeds the angle of repose;
- bank faces slough into loose material when they exceed the soil's critical angle and are steeper than the scan itself was there (faces that existed in the capture demonstrably stood, so they are treated as stable).

Invariants are unit-tested: loose on field + carried = swell × (bank cut + sloughed); masked cells never change; nothing goes below max dig depth; in machine-control mode nothing is cut below design inside the footprint or below original grade outside it.

## Splat deformation on the GPU

Spark builds each frame's splats with a generator graph (`dyno`). LiveDirt injects a world-space modifier written in GLSL that, per splat:

1. samples `H0` and the object code (static texture) and `dH = H - H0`, loose thickness and design (dynamic texture) with manual bilinear filtering;
2. computes a binding weight (near the original surface and not on an object);
3. translates bound splats by `dH` and re-tints them toward the learned subsoil/spoil palette, keeping per-splat brightness so texture survives;
4. fades bound splats out where the ground changed by more than ~10 cm, where the generated-soil layer takes over;
5. applies the grade heatmap, generated-soil overlay, rain wetness, the global colour grade, and hides the photographed machine in twin mode.

The sim writes only the dirty rectangle into the dynamic texture; the GPU regenerates splats when it changes.

## Generated soil

For every changed cell the CPU emits splats proportional to surface area: a dark crevice base, clod pairs (a lit top splat and an offset shadow splat, which reads as relief under overcast light), fine crumbs drawn from the scene's own luminance distribution, vertical tooth streaks on steep cut walls, and 10 to 25 cm chunks on loose spoil. Sky visibility from the normal and depth below the rim darkens trench floors. Generation is deterministic per cell, so regenerating while digging does not flicker. This is a procedural stand-in for a learned generative prior and is labelled as such everywhere.

## Machine and autonomy

Linkage dimensions come from the DX225LC-7 spec sheet (boom 5.7 m, arm 2.9 m); boom-foot position and joint limits are typical values for the class. A FK -> IK round trip is exact across the envelope, and reach at ground and dig depth land near the spec values (tests).

The controller splits the design into lanes across the full top width (so side slopes get trimmed) and bites along each lane, far to near, extending through the end slopes. Each bite: approach, penetrate, drag at cut depth (bounded by design and a per-bite maximum), curl, lift, swing to the spoil point (beside the trench with a ≥ 0.61 m setback plus pile radius), dump with a dwell, swing back. If the preferred bucket attitude cannot reach a target, IK searches nearby attitudes. Joint rates are limited per joint; cycle times come out around 19 s.

After each pass the machine holds still for 1.6 s, accumulates both lidars' points into the design grid and computes coverage, the fraction of measured cells in tolerance and the mean error. It accepts at ≥ 85% in tolerance, keeps going while passes still improve the mean by ≥ 1 cm, and refuses to judge below 40% coverage. Early versions declared success with 0% coverage; the coverage gate exists because of that bug.

## Sensors

- **Lidar.** 64 channels, ±22.5°, 1024 (roof) or 512 (front) columns, 10 Hz, in a Web Worker. Rays are cast against the terrain-plus-obstacle heightfield with a max-height mip pyramid (validated against brute-force marching), plus analytic capsules for people. Range noise grows with range; dropout depends on incidence, range and albedo; dust and rain are Beer-Lambert media that produce labelled early returns and silent losses. The front unit is tilted 35° nose-down because the level roof unit cannot see the ground within ~12 m in front of a machine on a mound.
- **Roof camera.** A second `SparkRenderer` with its own sort order and an offscreen target, posed from the twin's mast mount, composited as picture-in-picture and read back for detection and MCAP.

## Perception

YOLOX-Nano ONNX (Apache-2.0) runs through onnxruntime-web's WASM backend (~70 to 120 ms per frame). The sweep places a worker at a grid of distances and bearings in four conditions, computes a ground-truth box from the capsule, tests visibility against terrain and the twin meshes, and scores camera detections (IoU ≥ 0.3) and lidar visibility (≥ 5 returns). See RESULTS.md.

## Hosting

Static build on Cloudflare Workers static assets (free tier), `livedirt.mohithgajjela.com`. Scene files are cached for a day, hashed assets immutably.
