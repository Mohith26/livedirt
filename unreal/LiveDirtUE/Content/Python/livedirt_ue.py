"""LiveDirt in Unreal Editor: renders the live LiveDirt simulation served by app/bench/unreal-bridge.ts.

The simulation (captured terrain, earthwork, adaptive dig controller, per-scoop learned soil) runs in the bridge; this
script builds the level (sky, sun, fog, terrain meshes, excavator) and, on every editor frame, polls the bridge and moves
the machine; the dig-zone terrain mesh is rebuilt whenever the soil changes (at most ~2x per second).

Usage (Unreal Editor, Output Log -> Python):
    import livedirt_ue; livedirt_ue.start()        # build the level and start following the bridge
    livedirt_ue.stop()                              # stop polling (actors stay)
Control the dig (run / pause / reset, soil model v1/v2, soil type, speed, colour mode) at http://127.0.0.1:8790/

Unreal-side requirements (enabled in LiveDirtUE.uproject): Python Editor Script Plugin, Editor Scripting Utilities,
Geometry Script. No C++ (no Xcode needed).
"""
import base64
import json
import math
import struct
import time
import urllib.request

import unreal

BRIDGE = "http://127.0.0.1:8790"
PREFIX = "LD_"
ROOT = "/Game/LiveDirt"
TERRAIN_MIN_INTERVAL = 0.5

_state = {"tick": None, "parts": {}, "zone_actor": None, "terrain_version": -1, "last_terrain": 0.0, "mats": {}, "warned": set()}


# ------------------------------------------------------------------ helpers
def _get(path, timeout=2.0):
    with urllib.request.urlopen(BRIDGE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _warn_once(key, msg):
    if key not in _state["warned"]:
        _state["warned"].add(key)
        unreal.log_warning(f"[LiveDirt] {msg}")


def _actors():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def _srgb_to_linear(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _decode_grid(g):
    ni, nj = g["ni"], g["nj"]
    h = struct.unpack(f"<{ni * nj}f", base64.b64decode(g["heights"]))
    c = base64.b64decode(g["colors"])
    return ni, nj, h, c


def grid_buffers(g, z_offset=0.0, hole=None):
    """Bridge grid payload -> (vertices, normals, triangles, colors) as plain Python tuples (testable without Unreal).
    Vertex (ii, jj) sits at Unreal (origin_x + ii*step, origin_y + jj*step, height); colours are linear RGB 0..1.
    hole = (ii0, jj0, ii1, jj1): quads fully inside this vertex-index box are left out (the live dig zone goes there)."""
    ni, nj, h, c = _decode_grid(g)
    step, (ox, oy) = g["stepCm"], g["originCm"]
    verts = [(ox + ii * step, oy + jj * step, h[jj * ni + ii] + z_offset) for jj in range(nj) for ii in range(ni)]
    lut = [_srgb_to_linear(v) for v in range(256)]
    cols = [(lut[c[3 * q]], lut[c[3 * q + 1]], lut[c[3 * q + 2]]) for q in range(ni * nj)]
    norms = []
    for jj in range(nj):
        for ii in range(ni):
            hx0 = h[jj * ni + max(ii - 1, 0)]; hx1 = h[jj * ni + min(ii + 1, ni - 1)]
            hy0 = h[max(jj - 1, 0) * ni + ii]; hy1 = h[min(jj + 1, nj - 1) * ni + ii]
            dx = (hx1 - hx0) / (step * (min(ii + 1, ni - 1) - max(ii - 1, 0)) or 1)
            dy = (hy1 - hy0) / (step * (min(jj + 1, nj - 1) - max(jj - 1, 0)) or 1)
            L = math.sqrt(dx * dx + dy * dy + 1.0)
            norms.append((-dx / L, -dy / L, 1.0 / L))
    tris = []
    for jj in range(nj - 1):
        for ii in range(ni - 1):
            if hole and hole[0] <= ii and ii + 1 <= hole[2] and hole[1] <= jj and jj + 1 <= hole[3]:
                continue
            a = jj * ni + ii; b = a + 1; d = a + ni; e = d + 1
            # winding chosen so faces point +Z in Unreal's left-handed frame; the material is two-sided anyway
            tris.append((a, d, b)); tris.append((b, d, e))
    return verts, norms, tris, cols


# ------------------------------------------------------------------ materials
def _material(name, build):
    path = f"{ROOT}/Materials/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.EditorAssetLibrary.load_asset(path)
    at = unreal.AssetToolsHelpers.get_asset_tools()
    mat = at.create_asset(name, f"{ROOT}/Materials", unreal.Material, unreal.MaterialFactoryNew())
    build(mat)
    unreal.MaterialEditingLibrary.recompile_material(mat)
    unreal.EditorAssetLibrary.save_asset(path)
    return mat


def _const(mat, value, x, y):
    e = unreal.MaterialEditingLibrary.create_material_expression(mat, unreal.MaterialExpressionConstant, x, y)
    e.set_editor_property("r", value)
    return e


def _build_vertex_color(mat):
    mel = unreal.MaterialEditingLibrary
    mat.set_editor_property("two_sided", True)
    vc = mel.create_material_expression(mat, unreal.MaterialExpressionVertexColor, -400, 0)
    mel.connect_material_property(vc, "", unreal.MaterialProperty.MP_BASE_COLOR)
    mel.connect_material_property(_const(mat, 0.93, -400, 200), "", unreal.MaterialProperty.MP_ROUGHNESS)
    mel.connect_material_property(_const(mat, 0.0, -400, 300), "", unreal.MaterialProperty.MP_SPECULAR)


def _build_solid(mat):
    mel = unreal.MaterialEditingLibrary
    col = mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -400, 0)
    col.set_editor_property("parameter_name", "Color")
    mel.connect_material_property(col, "", unreal.MaterialProperty.MP_BASE_COLOR)
    for i, (pname, prop, dv) in enumerate([("Metallic", unreal.MaterialProperty.MP_METALLIC, 0.1), ("Roughness", unreal.MaterialProperty.MP_ROUGHNESS, 0.5)]):
        p = mel.create_material_expression(mat, unreal.MaterialExpressionScalarParameter, -400, 200 + 120 * i)
        p.set_editor_property("parameter_name", pname)
        p.set_editor_property("default_value", dv)
        mel.connect_material_property(p, "", prop)


PALETTE = {  # base colour (linear), metallic, roughness
    "paint": ((0.80, 0.16, 0.012), 0.15, 0.45), "dark": ((0.018, 0.016, 0.014), 0.2, 0.85),
    "steel": ((0.25, 0.26, 0.28), 0.8, 0.35), "glass": ((0.02, 0.04, 0.06), 0.4, 0.08),
    "soil": ((0.16, 0.12, 0.08), 0.0, 0.95),
}


def _solid_instance(key):
    if key in _state["mats"]:
        return _state["mats"][key]
    parent = _material("M_LD_Solid", _build_solid)
    path = f"{ROOT}/Materials/MI_LD_{key}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        mi = unreal.EditorAssetLibrary.load_asset(path)
    else:
        at = unreal.AssetToolsHelpers.get_asset_tools()
        mi = at.create_asset(f"MI_LD_{key}", f"{ROOT}/Materials", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        unreal.MaterialEditingLibrary.set_material_instance_parent(mi, parent)
    (r, g, b), metal, rough = PALETTE.get(key, PALETTE["steel"])
    mel = unreal.MaterialEditingLibrary
    mel.set_material_instance_vector_parameter_value(mi, "Color", unreal.LinearColor(r, g, b, 1.0))
    mel.set_material_instance_scalar_parameter_value(mi, "Metallic", metal)
    mel.set_material_instance_scalar_parameter_value(mi, "Roughness", rough)
    unreal.EditorAssetLibrary.save_asset(path)
    _state["mats"][key] = mi
    return mi


# ------------------------------------------------------------------ level
def _clear():
    for a in _actors().get_all_level_actors():
        if a.get_actor_label().startswith(PREFIX):
            _actors().destroy_actor(a)
    _state["parts"].clear()
    _state["zone_actor"] = None


def _spawn(cls, label, loc=(0, 0, 0), rot=(0, 0, 0)):
    a = _actors().spawn_actor_from_class(cls, unreal.Vector(*loc), unreal.Rotator(roll=rot[0], pitch=rot[1], yaw=rot[2]))
    a.set_actor_label(PREFIX + label)
    return a


def _environment(center):
    sun = _spawn(unreal.DirectionalLight, "Sun", rot=(0, -38, 135))
    lc = sun.get_component_by_class(unreal.DirectionalLightComponent)
    lc.set_editor_property("intensity", 9.0)
    try:
        lc.set_editor_property("atmosphere_sun_light", True)
    except Exception:
        pass
    _spawn(unreal.SkyAtmosphere, "Sky")
    sky = _spawn(unreal.SkyLight, "SkyLight")
    sc = sky.get_component_by_class(unreal.SkyLightComponent)
    try:
        sc.set_editor_property("real_time_capture", True)
    except Exception:
        pass
    _spawn(unreal.ExponentialHeightFog, "Fog", loc=(center[0], center[1], 0))
    try:
        _spawn(unreal.VolumetricCloud, "Clouds")
    except Exception:
        pass
    try:
        ppv = _spawn(unreal.PostProcessVolume, "PostProcess")
        ppv.set_editor_property("unbound", True)
    except Exception:
        pass


def _mesh_actor(label, g, material, hole=None):
    a = _spawn(unreal.DynamicMeshActor, label)
    _fill_mesh(a, g, material, hole)
    return a


def _fill_mesh(actor, g, material, hole=None):
    verts, norms, tris, cols = grid_buffers(g, hole=hole)
    comp = actor.get_component_by_class(unreal.DynamicMeshComponent)
    mesh = comp.get_dynamic_mesh()
    mesh.reset()
    buf = unreal.GeometryScriptSimpleMeshBuffers()
    buf.set_editor_property("vertices", [unreal.Vector(*v) for v in verts])
    buf.set_editor_property("normals", [unreal.Vector(*n) for n in norms])
    buf.set_editor_property("vertex_colors", [unreal.LinearColor(c[0], c[1], c[2], 1.0) for c in cols])
    buf.set_editor_property("triangles", [unreal.IntVector(*t) for t in tris])
    unreal.GeometryScript_MeshBasicEditFunctions.append_buffers_to_mesh(mesh, buf)
    comp.set_material(0, material)


def _place_part(p):
    a = _state["parts"].get(p["name"])
    if a is None:
        cube = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube")
        a = _spawn(unreal.StaticMeshActor, "Machine_" + p["name"])
        sm = a.static_mesh_component
        sm.set_mobility(unreal.ComponentMobility.MOVABLE)
        sm.set_static_mesh(cube)
        sm.set_material(0, _solid_instance(p["color"]))
        _state["parts"][p["name"]] = a
    r = p["r"]
    a.set_actor_location_and_rotation(unreal.Vector(*p["c"]), unreal.Rotator(roll=r[0], pitch=r[1], yaw=r[2]), False, False)
    a.set_actor_scale3d(unreal.Vector(p["s"][0] / 100.0, p["s"][1] / 100.0, p["s"][2] / 100.0))


def build():
    scene = _get("/scene", timeout=30)
    _clear()
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    level_path = f"{ROOT}/Maps/LiveDirt"
    if unreal.EditorAssetLibrary.does_asset_exist(level_path):
        les.load_level(level_path)
        _clear()
    else:
        les.new_level(level_path)
    center = scene["machine"]["center"]
    _environment(center)
    vmat = _material("M_LD_VertexColor", _build_vertex_color)
    # whole capture at 20 cm, dropped 2 cm so the 10 cm dig-zone mesh sits on top where they overlap
    t, z = scene["terrain"], scene["zone"]
    st = round(t["stepCm"] / 10.0)  # site grid stride in 10 cm cells
    hole = (math.ceil(z["i0"] / st), math.ceil(z["j0"] / st), z["i1"] // st, z["j1"] // st)
    _mesh_actor("Terrain_Site", t, vmat, hole)
    site = _actors().get_all_level_actors()
    for a in site:
        if a.get_actor_label() == PREFIX + "Terrain_Site":
            a.set_actor_location(unreal.Vector(0, 0, -2.0), False, False)
    _state["zone_actor"] = _mesh_actor("Terrain_DigZone", _get("/terrain", timeout=10), vmat)
    _state["terrain_version"] = -1
    for p in _get("/state")["parts"]:
        _place_part(p)
    # editor camera: behind and above the machine, looking at the trench
    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    ues.set_level_viewport_camera_info(unreal.Vector(center[0] - 900, center[1] - 1300, center[2] + 900), unreal.Rotator(roll=0, pitch=-30, yaw=40))
    les.save_current_level()
    unreal.log("[LiveDirt] level built; control the dig at " + BRIDGE)


def _tick(dt):
    try:
        s = _get("/state", timeout=0.5)
    except Exception as e:  # bridge not running: keep the editor responsive, try again next frame
        _warn_once("bridge", f"bridge not reachable at {BRIDGE} ({e}); start it with: cd app && npx tsx bench/unreal-bridge.ts")
        return
    for p in s["parts"]:
        _place_part(p)
    now = time.time()
    if s["terrainVersion"] != _state["terrain_version"] and now - _state["last_terrain"] > TERRAIN_MIN_INTERVAL and _state["zone_actor"]:
        try:
            g = _get("/terrain", timeout=5)
            _fill_mesh(_state["zone_actor"], g, _material("M_LD_VertexColor", _build_vertex_color))
            _state["terrain_version"] = g["version"]
        except Exception as e:
            _warn_once("terrain", f"terrain update failed: {e}")
        _state["last_terrain"] = now


def start(rebuild=True):
    stop()
    if rebuild:
        build()
    _state["tick"] = unreal.register_slate_post_tick_callback(_tick)
    unreal.log("[LiveDirt] following the bridge; open " + BRIDGE + " to run the dig")


def stop():
    if _state["tick"] is not None:
        unreal.unregister_slate_post_tick_callback(_state["tick"])
        _state["tick"] = None
