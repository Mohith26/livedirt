# LiveDirt in Unreal Engine (local)

The simulation runs in a small local process (the **bridge**, `app/bench/unreal-bridge.ts`): captured terrain,
earthwork heightfield, adaptive dig controller, per-scoop learned soil (v2) or the v1 heightfield cut, lidar survey.
Unreal Editor renders it through a Python script (`LiveDirtUE/Content/Python/livedirt_ue.py`) that builds the level
(sky, sun, fog, terrain meshes, excavator) and follows the bridge every frame. No C++, so no Xcode is needed.

## Run
1. `unreal/run_bridge.sh` (keep it running)
2. Open `unreal/LiveDirtUE/LiveDirtUE.uproject` in Unreal Engine 5.4+ (Epic Games Launcher). On first open the editor
   builds the level automatically (~10 s); otherwise: Output Log -> Python -> `import livedirt_ue; livedirt_ue.start()`.
3. Control the dig at http://127.0.0.1:8790/ : run / pause / reset, soil model (learned v2 vs heightfield v1),
   controller, soil type, speed, colour mode (scan colours, grade error, cut/fill).
4. Fly around the editor viewport (right mouse + WASD, scroll for speed).

## What is real
Same simulation and numbers as livedirt-v2.mohithgajjela.com (on the gate trench: learned v2 ~8% on grade, floor ~40 cm
high, which matches the MPM teacher; v1 heightfield ~57%, floor ~1 cm). Unreal only renders it.
