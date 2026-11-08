"""Project Chrono baselines for DigBench (external open-source reference solver).

SCM (pychrono.vehicle.SCMTerrain, Soil Contact Model) is a Bekker-Wong-Janosi *surface* model: a terrain heightfield
whose nodes are pressed down by vertical ray hits from rigid bodies, with Mohr-Coulomb/Janosi shear and optional
bulldozing (lateral mass flow by erosion-angle relaxation). It has no volumetric soil, so it cannot carry material in a
bucket and cannot represent a fully submerged plate.

CRM (pychrono.fsi, Continuous Representation Model, SPH with elasto-plastic Drucker-Prager rheology) is volumetric and
runs on CUDA. Used for the column-collapse and bucket attempts.

Every parameter below carries its source. Nothing is fitted to the MPM results.
"""
import math
import time

import numpy as np

# ---- SCM soil parameters: dry sand --------------------------------------------------------------------------------
MU_F = 0.38  # same bulk friction as the MPM runs (run_bucket --mu-f); phi = atan(0.38) = 20.8 deg
SCM_SOIL = {
    # Wong, Theory of Ground Vehicles; "Dry sand" row (moisture 0%): n 1.1, kc 0.99 kN/m^(n+1), kphi 1528.43 kN/m^(n+2),
    # c 1.04 kPa, phi 28 deg (as tabulated in NASA/TM-20250006958 Table of typical Bekker values). Janosi K = 0.025 m
    # for dry sand (Wong's table as reproduced in a Polito thesis, Table 2.2 and a 2015 real-time terramechanics paper).
    "Bekker_Kphi": 1528.43e3,  # N/m^(n+2)
    "Bekker_Kc": 0.99e3,       # N/m^(n+1)
    "Bekker_n": 1.1,
    # Dry sand is treated as cohesionless (c = 0): the tabulated 1.04 kPa is an apparent value from plate-sinkage fits.
    # The tabulated value is run as a sensitivity case (c_pa = 1040).
    "Mohr_cohesion": 0.0,      # Pa
    "Mohr_friction": math.degrees(math.atan(MU_F)),  # deg; set equal to the task's bulk friction, NOT a Wong value
    "Janosi_shear": 0.025,     # m
    # Chrono-specific K, R (not part of original SCM): Chrono SCM demo/documentation defaults; R = 0 and large K recover
    # the original SCM. Value from Chrono's SCM wheel demos (elastic_K 2e8 Pa/m, damping_R 3e4 Pa s/m).
    "elastic_K": 2.0e8,
    "damping_R": 3.0e4,
}
# Bulldozing (lateral flow) parameters: erosion angle = soil friction angle (angle of repose of dry sand = phi), the
# remaining arguments are Chrono defaults (flow_factor 1, 3 erosion iterations, 10 propagations).
BULLDOZE = {"erosion_angle": SCM_SOIL["Mohr_friction"], "flow_factor": 1.0, "erosion_iterations": 3, "erosion_propagations": 10}

SURF = 1.2  # MPM soil depth; SCM surface sits at z = 0 so world z_scm = z_mpm - SURF
BED_X = 3.4
CELL = 0.1


def _rot_y(pitch):
    import pychrono as chrono
    # DigBench pitch is CCW in (x, z): world x = c lx - s lz, z = s lx + c lz. Chrono rotation about +y by -pitch.
    return chrono.QuatFromAngleY(-pitch)


def make_bucket_body(sys, boxes, mirror=True, material=None):
    """Rigid, kinematically driven body built from DigBench bucket_boxes (centre, half-extents, pitch) in tool frame."""
    import pychrono as chrono
    body = chrono.ChBody()
    body.SetFixed(True)
    body.EnableCollision(True)
    mat = material or chrono.ChContactMaterialNSC()
    shapes = list(boxes)
    if mirror:  # y = +0.5 side plate -> add the y = -0.5 mirror (back, floor, lid already span |y| <= 0.6)
        for c, h, a in boxes:
            if c[1] > 0.1:
                shapes.append(((c[0], -c[1], c[2]), h, a))
    for c, h, a in shapes:
        sh = chrono.ChCollisionShapeBox(mat, 2 * h[0], 2 * h[1], 2 * h[2])
        body.AddCollisionShape(sh, chrono.ChFramed(chrono.ChVector3d(*c), chrono.QUNIT))
    sys.AddBody(body)
    return body, shapes


def make_scm(sys, delta, soil=None, bulldoze=True, size=(BED_X, 2.0)):
    import pychrono as chrono
    import pychrono.vehicle as veh
    s = {**SCM_SOIL, **(soil or {})}
    t = veh.SCMTerrain(sys, False)
    t.SetReferenceFrame(chrono.ChCoordsysd(chrono.ChVector3d(size[0] / 2, 0, 0), chrono.QUNIT))
    t.SetSoilParameters(s["Bekker_Kphi"], s["Bekker_Kc"], s["Bekker_n"], s["Mohr_cohesion"], s["Mohr_friction"],
                        s["Janosi_shear"], s["elastic_K"], s["damping_R"])
    if bulldoze:
        t.EnableBulldozing(True)
        b = BULLDOZE
        t.SetBulldozingParameters(b["erosion_angle"], b["flow_factor"], b["erosion_iterations"], b["erosion_propagations"])
    t.Initialize(size[0], size[1], delta)
    return t


def heightfield(terrain, nxy=(int(BED_X / CELL), int(1.0 / CELL)), cell=CELL):
    """Terrain height (MPM convention: soil surface = SURF) at cell centres over x in [0, 3.4], y in [0, 1] (half domain)."""
    import pychrono as chrono
    H = np.zeros(nxy)
    for i in range(nxy[0]):
        for j in range(nxy[1]):
            H[i, j] = terrain.GetHeight(chrono.ChVector3d((i + 0.5) * cell, (j + 0.5) * cell, 0.0)) + SURF
    return H


def run_scm_bucket(path_fn, duration, delta=0.05, dt=1e-3, soil=None, bulldoze=True, boxes=None, sample_every=0.01,
                   active_pad=0.3):
    """Drive the bucket along path_fn(t) -> (pivot xyz in MPM frame, pitch). Returns series, final heightfield, timing."""
    import pychrono as chrono
    from ..run_bucket import bucket_boxes
    boxes = boxes or bucket_boxes(0.04)
    sys = chrono.ChSystemNSC()
    sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    body, shapes = make_bucket_body(sys, boxes)
    terrain = make_scm(sys, delta, soil, bulldoze)
    # active domain: OOBB in body frame enclosing the shell (x -0.9..0.1, y +-0.6, z -1.0..0.1) plus padding
    terrain.AddActiveDomain(body, chrono.ChVector3d(-0.45, 0, -0.45),
                            chrono.ChVector3d(1.0 + 2 * active_pad, 1.2 + 2 * active_pad, 1.1 + 2 * active_pad))
    H0 = heightfield(terrain)
    rows = []
    nsteps = int(round(duration / dt))
    force, torque = chrono.ChVector3d(), chrono.ChVector3d()
    next_s = 0.0
    t0 = time.perf_counter()
    for k in range(nsteps + 1):
        t = k * dt
        pv, pitch = path_fn(t)
        pv2, pitch2 = path_fn(t + dt)
        pos = chrono.ChVector3d(pv[0], pv[1], pv[2] - SURF)
        body.SetPos(pos)
        body.SetRot(_rot_y(pitch))
        v = (np.asarray(pv2) - np.asarray(pv)) / dt
        body.SetPosDt(chrono.ChVector3d(*v))
        body.SetAngVelParent(chrono.ChVector3d(0, -(pitch2 - pitch) / dt, 0))
        sys.DoStepDynamics(dt)
        if t >= next_s - 1e-12:
            ok = terrain.GetContactForceBody(body, force, torque)
            rows.append([t, pv[0], pv[2], pitch, force.x if ok else 0.0, force.y if ok else 0.0,
                         force.z if ok else 0.0, torque.y if ok else 0.0, terrain.GetNumContactPatches()])
            next_s += sample_every
    wall = time.perf_counter() - t0
    H1 = heightfield(terrain)
    return {"series": np.array(rows), "H0": H0, "H1": H1, "wall_s": wall, "steps": nsteps + 1, "dt": dt,
            "n_shapes": len(shapes)}


def run_scm_plate(beta_deg, gamma_deg, depth0=0.05, travel=0.05, speed=0.1, delta=0.01, dt=1e-3, soil=None,
                  bulldoze=True):
    """Structural probe: drive the 3.81 x 2.54 x 0.64 cm plate below the SCM surface. SCM ray hits come from
    vertical rays against the body, so a submerged plate sees no soil above it as load."""
    import pychrono as chrono
    sys = chrono.ChSystemNSC()
    sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    body = chrono.ChBody()
    body.SetFixed(True)
    body.EnableCollision(True)
    mat = chrono.ChContactMaterialNSC()
    body.AddCollisionShape(chrono.ChCollisionShapeBox(mat, 0.0381, 0.0254, 0.0064), chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT))
    sys.AddBody(body)
    terrain = make_scm(sys, delta, soil, bulldoze, size=(1.0, 1.0))
    terrain.AddActiveDomain(body, chrono.ChVector3d(0, 0, 0), chrono.ChVector3d(0.2, 0.2, 0.2))
    b, g = math.radians(beta_deg), math.radians(gamma_deg)
    # plate: beta rotates about y (pitch), gamma rotates about the plate's long axis; motion toward -x
    q = chrono.QuatFromAngleY(-b) * chrono.QuatFromAngleZ(g)
    rows = []
    force, torque = chrono.ChVector3d(), chrono.ChVector3d()
    n = int(travel / speed / dt)
    z = -depth0
    for k in range(n + 1):
        body.SetPos(chrono.ChVector3d(0.5 - speed * k * dt, 0, z))
        body.SetRot(q)
        body.SetPosDt(chrono.ChVector3d(-speed, 0, 0))
        sys.DoStepDynamics(dt)
        ok = terrain.GetContactForceBody(body, force, torque)
        rows.append([k * dt, force.x if ok else 0.0, force.z if ok else 0.0, terrain.GetNumContactPatches()])
    return np.array(rows)
