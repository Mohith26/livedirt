# LiveDirt in Unreal Engine (local)

The simulation runs in a small local process (the **bridge**, `app/bench/unreal-bridge.ts`): captured terrain,
earthwork heightfield, adaptive dig controller, per-scoop learned soil (v3: model a16, a 5-member ensemble) or the v1
heightfield cut, lidar survey.
Unreal Editor renders it through a Python script (`LiveDirtUE/Content/Python/livedirt_ue.py`) that builds the level
(sky, sun, fog, terrain meshes, excavator) and follows the bridge every frame. No C++, so no Xcode is needed.

## Run
1. `unreal/run_bridge.sh` (keep it running)
2. Open `unreal/LiveDirtUE/LiveDirtUE.uproject` in Unreal Engine 5.4+ (Epic Games Launcher). On first open the editor
   builds the level automatically (~10 s); otherwise: Output Log -> Python -> `import livedirt_ue; livedirt_ue.start()`.
3. Control the dig at http://127.0.0.1:8790/ : run / pause / reset, soil model (learned v3 / a16 vs heightfield v1),
   controller, soil type, speed, colour mode (scan colours, grade error, cut/fill).
4. Fly around the editor viewport (right mouse + WASD, scroll for speed).

## What is real
Same simulation, model files and numbers as livedirt-v3.mohithgajjela.com (model a16, 5 U-Nets averaged, trained on
805 MPM-teacher bites incl. an active-learning DAgger loop; held-out scoops: surface RMSE 7.9-8.2 cm per member vs
17.2 cm for the cut-and-relax prior). Gate trench, final surface vs the MPM teacher: sandy gravel 11.6 / 13.7 cm
(v1 heightfield 31.3 cm), wet clay 14.5 / 11.0 cm (v1 45.1 cm). On grade: teacher 8.3% / 3.7%, learned v3 6.3% / 4.7%,
v1 57% / 33%. v3 reproduces the unfinished-trench outcome but is not yet within the teacher's own run-to-run noise in
sandy gravel. Unreal only renders it.

To swap in a different DigBench model later, copy its members to `app/public/models/scoop-<name>-s0..4.onnx` and change
`SCOOP_MODEL` at the top of `app/bench/unreal-bridge.ts`.
