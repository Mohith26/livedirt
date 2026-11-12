# LiveDirt in Tempo

[Tempo](https://github.com/tempo-sim/Tempo) is a set of Unreal Engine plugins for robotics simulation (UE 5.7/5.8,
macOS 15+ on Apple silicon, Linux, Windows). Running LiveDirt inside a Tempo project keeps everything from the plain
Unreal viewer (README.md) and adds Tempo's sensors and API: a `TempoCamera` (color, depth, labels, boxes, H.264) and a
`TempoLidar` (scene-depth lidar that also sees fog and dust) looking at the live dig, streamed to Python over gRPC.

The simulation itself is unchanged: the bridge (`app/bench/unreal-bridge.ts`, learned soil v3 = model a16) runs the dig,
and `livedirt_ue.py` renders it. While Play-In-Editor runs, the script drives the PIE copies of the actors, because
that is the world Tempo's sensors render.

## One-time setup (about 30-60 min, mostly the first C++ build)
Needs UE 5.8 (Epic Games Launcher), Xcode, git and jq (`brew install jq`).

```bash
~/livedirt-v3/unreal/make_tempo_project.sh          # creates ~/LiveDirtTempo from TempoSample
cd ~/LiveDirtTempo
./Scripts/Build.sh
```

`make_tempo_project.sh` clones TempoSample (with the Tempo submodule), renames it, enables Python / Editor Scripting /
Geometry Script in the uproject, copies `livedirt_ue.py` + `init_livedirt.py` into `Content/Python` with an editor
startup hook, and runs `Plugins/Tempo/Setup.sh`.

## Every time
1. Terminal 1: `~/livedirt-v3/unreal/run_bridge.sh` (leave it running).
2. Terminal 2: `cd ~/LiveDirtTempo && ./Scripts/Run.sh`. The editor opens and, after ~10 s, builds the LiveDirt level
   (or Output Log -> Python: `import livedirt_ue; livedirt_ue.start()`).
3. Put a sensor on the site: Content Browser -> Settings -> Show Plugin Content, then drag
   `TempoSensors Content/SensorRig/BP_SensorRig` into the level a few metres from the trench, facing it
   (its `TempoCamera` is on top; add a `TempoLidar` component to the same rig in the Details panel for lidar).
4. Press **Play** (or from Python: `tce.play_in_editor()`, below). The Output Log shows
   `[LiveDirt] following the bridge in the PIE world`.
5. Start the dig at http://127.0.0.1:8790/ (Run; soil model "Learned per-scoop (v3, a16 ensemble)").
6. Terminal 3, stream the sensors:
   ```bash
   cd ~/LiveDirtTempo && source ./TempoEnv/bin/activate
   python ./Plugins/Tempo/ExampleClients/Python/SensorPlayground.py   # list sensors, stream camera images
   python ./Plugins/Tempo/ExampleClients/Python/LidarPreview.py       # lidar returns, by distance / intensity / label
   ```

Driving it from Python instead of the editor UI:
```python
import tempo_sim.tempo_core_editor as tce, tempo_sim.tempo_world as tw
from tempo_sim.TempoWorld.WorldControl_pb2 import Value
tce.play_in_editor()
tw.spawn_actor(actor_type="BP_SensorRig")      # spawns at the origin; move it with tw.set_actor_transform(...)
tw.set_property(actor="BP_SensorRig", component="TempoCamera", property="FOVAngle", value=Value(float_value=60.0))
```

## Limits (be upfront about these)
- **Clock**: the dig runs on the bridge's wall clock. Tempo's `pause()` / `step()` / fixed-step mode freeze Unreal and
  the sensors, not the dig, so captures are not yet deterministic per step. The proper fix is a small Tempo service in
  C++ (see Tempo's Greeter plugin) that steps the bridge once per Tempo tick, so one step = one scoop update = one frame.
- **Terrain updates** arrive at most ~2x per second (one mesh rebuild per changed soil state).
- **Labels**: the terrain and machine are runtime actors, so give them semantic labels by actor tag (Tempo's label
  table `ActorTags` column) if you want label images or labelled lidar returns.
- This setup was written against Tempo's docs and checked for script syntax; the Tempo C++ build and a PIE session
  have not been run on this machine yet.
