"""Project Chrono CRM (SPH, pychrono.fsi) runs for DigBench tasks 1 and 3. CUDA only: submit through the GPU job queue.

CRM = Continuous Representation Model: SPH discretisation of the elasto-plastic (Drucker-Prager / mu(I)) continuum
(Chrono::FSI-SPH). Material parameters are set from the task's own references, not fitted:
  task 1: bulk density 2500*0.64 = 1600 kg/m^3, mu_s = mu_2 = tan(repose 23.5 deg) (Sarlin et al. 2021, same phi as the MPM
          runs), grain diameter 5 mm, cohesion 0, E = 1 MPa, nu = 0.3 (E/nu = same MPM values; Chrono demo defaults).
  task 3: bulk density 1600 (run_bucket default), mu_s = mu_2 = 0.38 (run_bucket --mu-f default), E = 1 MPa (run_bucket).
mu_I0 = 0.04 is the Chrono demo value; with mu_2 = mu_s the rheology is rate-independent constant friction, like the MPM.
"""
import math
import os
import time

import numpy as np

DRY = bool(os.environ.get("CRM_DRYRUN"))  # construct only (no CUDA): API check on a CPU-only shell


def _setup(spacing, soil, step, max_vel, sys=None):
    import pychrono as chrono
    import pychrono.fsi as fsi
    sysm = sys or chrono.ChSystemNSC()
    sysm.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    prob = fsi.ChFsiProblemCartesian(spacing, sysm)
    prob.SetVerbose(False)
    prob.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    prob.SetStepSizeCFD(step)
    prob.SetStepsizeMBD(step)
    m = fsi.SoilProperties()
    m.density = soil["density"]
    m.Young_modulus = soil["E"]
    m.Poisson_ratio = soil["nu"]
    m.mu_I0 = 0.04
    m.mu_fric_s = soil["mu"]
    m.mu_fric_2 = soil["mu"]
    m.average_diam = soil["diam"]
    m.cohesion_coeff = 0.0
    prob.SetCrmSPH(m)
    p = fsi.SPHParameters()
    p.integration_scheme = fsi.IntegrationScheme_RK2
    p.initial_spacing = spacing
    p.d0_multiplier = 1.0
    p.free_surface_threshold = 0.8
    p.artificial_viscosity = 0.5
    p.use_consistent_gradient_discretization = False
    p.use_consistent_laplacian_discretization = False
    p.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
    p.boundary_method = fsi.BoundaryMethod_ADAMI
    p.use_variable_time_step = True
    p.max_velocity = max_vel
    prob.SetSPHParameters(p)
    return chrono, fsi, sysm, prob


def collapse_case(case, spacing, W=0.15, t_end=2.0, wall_cap=420.0, step=1e-4):
    """Column collapse in a 15 cm wide channel (Sarlin geometry). Column fills x in [0, L0], z in [0, H0] against the
    back wall; the floor is a BCE boundary. Released instantly at t = 0 (Sarlin's gate is lifted at 1.2 m/s, i.e. fast
    relative to the free-fall time, but not instantaneous: this is a stated difference)."""
    L0, H0 = case["L0"], case["H0"]
    soil = {"density": 2500.0 * 0.64, "E": 1.0e6, "nu": 0.3, "mu": math.tan(math.radians(23.5)), "diam": 0.005}
    chrono, fsi, sysm, prob = _setup(spacing, soil, step, max_vel=math.sqrt(2 * 9.81 * H0) * 1.5)
    Lc, Hc = 0.9, H0 + 0.15
    prob.AddBoxContainer(chrono.ChVector3d(Lc, W, Hc), chrono.ChVector3d(Lc / 2, 0, Hc / 2),
                         fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS)
    # Chrono's Cartesian lattice includes both box ends (n/spacing + 1 points), so shrink by one spacing to get
    # cell-centred particles filling exactly [0, L0] x [-W/2, W/2] x [0, H0]
    sp = spacing
    prob.Construct(chrono.ChVector3d(L0 - sp, W - sp, H0 - sp), chrono.ChVector3d(L0 / 2, 0, H0 / 2), fsi.BoxSide_NONE)
    n_sph = prob.GetNumSPHParticles()
    if DRY:
        return {"n_sph": n_sph, "dry": True}
    prob.Initialize()
    t0 = time.perf_counter()
    t, meta = 0.0, 5 * step
    frames, stopped = [], "t_end"
    nxt = 0.0
    while t < t_end:
        prob.DoStepDynamics(meta)
        t += meta
        if t >= nxt:
            v = np.asarray(prob.GetFluidSystemSPH().GetParticleVelocitiesNumpy())
            sp = float(np.sqrt((v ** 2).sum(1)).mean())
            frames.append((round(t, 4), sp))
            nxt += 0.05
            if t > 0.4 and sp < 0.005:
                stopped = "at_rest"
                break
        if time.perf_counter() - t0 > wall_cap:
            stopped = "wall_cap"
            break
    wall = time.perf_counter() - t0
    pos = np.asarray(prob.GetFluidSystemSPH().GetParticlePositionsNumpy())
    return {"pos": pos, "n_sph": n_sph, "t_sim": t, "wall_s": wall, "rtf": t / wall, "stopped": stopped,
            "frames": frames, "spacing": spacing, "W": W, "soil": soil, "step": step}


def bucket_case(spacing=0.04, t_end=5.5, wall_cap=900.0, step=2e-4, depth=0.6):
    """Full-scale bucket dig in CRM: same boxes/path as run_bucket (physical attitude), kinematically driven rigid body
    (huge mass, pose and velocity overwritten each exchange step). Full width bed (|y| <= 1 m)."""
    from ..attitude import physical_path
    from ..run_bucket import SURF, bucket_boxes
    soil = {"density": 1600.0, "E": 1.0e6, "nu": 0.3, "mu": 0.38, "diam": 0.001}
    chrono, fsi, sysm, prob = _setup(spacing, soil, step, max_vel=4.0)
    sysm.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))  # body is kinematic; SPH gravity set on the problem
    body = chrono.ChBody()
    body.SetMass(1e6)
    body.SetInertiaXX(chrono.ChVector3d(1e6, 1e6, 1e6))
    body.SetFixed(False)
    sysm.AddBody(body)
    geo = chrono.ChBodyGeometry()
    geo.materials.append(chrono.ChContactMaterialData())
    shapes = list(bucket_boxes(0.04))
    shapes += [((c[0], -c[1], c[2]), h, a) for c, h, a in list(shapes) if c[1] > 0.1]
    for c, h, a in shapes:
        geo.coll_boxes.append(chrono.BoxShape(chrono.ChVector3d(*c), chrono.QUNIT, chrono.ChBox(chrono.ChVector3d(2 * h[0], 2 * h[1], 2 * h[2]))))
    path_fn, T = physical_path(depth, SURF)
    p0, a0 = path_fn(0.0)
    body.SetPos(chrono.ChVector3d(*p0))
    body.SetRot(chrono.QuatFromAngleY(-a0))
    prob.AddRigidBody(body, geo, False)
    BED = (3.4, 2.0, SURF)
    prob.Construct(chrono.ChVector3d(*BED), chrono.ChVector3d(BED[0] / 2, 0, BED[2] / 2), fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS)
    n_sph = prob.GetNumSPHParticles()
    if DRY:
        return {"n_sph": n_sph, "dry": True}
    prob.Initialize()
    pos0 = np.asarray(prob.GetFluidSystemSPH().GetParticlePositionsNumpy()).copy()
    t0 = time.perf_counter()
    t, meta = 0.0, 5 * step
    rows, stopped = [], "t_end"
    nxt = 0.0
    while t < t_end:
        pv, pitch = path_fn(t)
        pv2, pitch2 = path_fn(t + meta)
        body.SetPos(chrono.ChVector3d(*pv))
        body.SetRot(chrono.QuatFromAngleY(-pitch))
        body.SetPosDt(chrono.ChVector3d(*((np.asarray(pv2) - np.asarray(pv)) / meta)))
        body.SetAngVelParent(chrono.ChVector3d(0, -(pitch2 - pitch) / meta, 0))
        prob.DoStepDynamics(meta)
        t += meta
        if t >= nxt:
            f = prob.GetFsiBodyForce(body)
            rows.append([t, pv[0], pv[2], pitch, f.x, f.y, f.z])
            nxt += 0.02
        if time.perf_counter() - t0 > wall_cap:
            stopped = "wall_cap"
            break
    wall = time.perf_counter() - t0
    pos = np.asarray(prob.GetFluidSystemSPH().GetParticlePositionsNumpy())
    pv, pitch = path_fn(min(t, T))
    return {"pos": pos, "pos0": pos0, "series": np.array(rows), "n_sph": n_sph, "t_sim": t, "wall_s": wall, "rtf": t / wall,
            "stopped": stopped, "pivot_end": pv, "pitch_end": pitch, "spacing": spacing, "soil": soil, "step": step}


def run(args, save, device_name, git_rev):
    from .. import metrics, references
    if args.cmd == "collapse-crm":
        want = {float(s) for s in args.cases.split(",")}
        cases = [c for c in references.COLLAPSE_CASES if c["a"] in want]
        rows = []
        for case in cases:
            try:
                r = collapse_case(case, args.spacing)
            except Exception as e:  # keep the job alive and record the failure
                rows.append({**case, "error": repr(e)})
                print("FAIL", case, repr(e), flush=True)
                continue
            if r.get("dry"):
                print("dry", case, r)
                continue
            pos = r.pop("pos")
            xc, h = metrics.profile_from_particles(pos[:, 0], pos[:, 2], args.spacing, args.spacing)
            dep = metrics.deposit(xc, h, case["L0"], references.COLLAPSE["grain_d"])
            sc = metrics.score(case, dep, references.collapse_runout, references.collapse_height)
            rows.append({**case, **dep, **sc, **{k: r[k] for k in ("n_sph", "t_sim", "wall_s", "rtf", "stopped", "frames", "spacing", "W", "step")},
                         "profile": {"x": xc.round(5).tolist(), "h": h.round(5).tolist()}})
            print(f"a={case['a']}: runout {sc['runout_sim']:.2f} vs {sc['runout_ref']:.2f} ({sc['runout_relerr']:+.1%}) "
                  f"height {sc['height_sim']:.2f} vs {sc['height_ref']:.2f} ({sc['height_relerr']:+.1%}) "
                  f"N={r['n_sph']} t={r['t_sim']:.2f}s wall={r['wall_s']:.0f}s rtf={r['rtf']:.4f} {r['stopped']}", flush=True)
        ok = [x for x in rows if "runout_relerr" in x]
        out = {"task": "column_collapse", "solver": "chrono-crm", "tag": args.tag, "device": device_name(True), "git": git_rev(),
               "chrono_version": "pychrono 10.0.0 (conda projectchrono)", "reference": references.COLLAPSE["source"],
               "params": {"spacing": args.spacing, "cases": args.cases, "module": __doc__},
               "summary": {"runout_mae_rel": float(np.mean([abs(x["runout_relerr"]) for x in ok])) if ok else None,
                           "height_mae_rel": float(np.mean([abs(x["height_relerr"]) for x in ok])) if ok else None,
                           "n_cases_ok": len(ok), "n_cases_requested": len(cases)},
               "cases": rows}
        save(out, f"chrono-collapse-{args.tag}.json", args.out)
        return
    # bucket
    from ..run_bucket import SURF, carried_volume
    from ..solvers.heightfield import Heightfield  # noqa: F401  (import check only)
    r = bucket_case(spacing=args.spacing if args.spacing >= 0.02 else 0.04, depth=args.depth)
    if r.get("dry"):
        print("dry", r)
        return
    pos = r.pop("pos")
    pos0 = r.pop("pos0")
    ser = r.pop("series")
    sp = r["spacing"]
    # shift to MPM-frame conventions: bed already z in [0, SURF]; bucket carried volume (full width, |y| < 0.5)
    pv, pa = r["pivot_end"], r["pitch_end"]
    rel = pos - np.array([pv[0], 0.0, pv[2]])
    ca, sa = math.cos(pa), math.sin(pa)
    lx = ca * rel[:, 0] + sa * rel[:, 2]
    lz = -sa * rel[:, 0] + ca * rel[:, 2]
    inside = (lx > -0.9) & (lx < 0.0) & (lz > -0.9) & (lz < 0.0) & (np.abs(rel[:, 1]) < 0.5)
    carried = float(inside.sum() * sp ** 3)
    on_bucket = (lx > -1.25) & (lx < 0.4) & (lz > -1.25) & (lz < 0.4) & (np.abs(rel[:, 1]) < 0.85)
    g = pos[~on_bucket]
    cell, nx, ny = 0.1, 34, 10
    def topmap(q):
        Hm = np.zeros((nx, ny))
        ix = np.clip((q[:, 0] / cell).astype(int), 0, nx - 1)
        iy = (q[:, 1] / cell).astype(int)
        m = (iy >= 0) & (iy < ny)
        np.maximum.at(Hm, (ix[m], iy[m]), q[m, 2] + 0.5 * sp)
        return Hm
    # the lattice start offset is solver-specific: report heights relative to the initial bed top (SURF)
    H0r = topmap(pos0)
    H = SURF + topmap(g) - H0r.max()
    t = ser[:, 0]
    drag = (t > 1.2) & (t < 3.0)
    fp = H[int(0.6 / cell):int(1.4 / cell), :int(0.4 / cell)]
    summary = {"drag_force_mean_kN": float(ser[drag, 4].mean() / 1e3), "drag_force_peak_kN": float(ser[:, 4].max() / 1e3),
               "vertical_force_peak_kN": float(np.abs(ser[:, 6]).max() / 1e3),
               "carried_bank_m3": carried, "bucket_geometric_m3": 0.9 * 0.9 * 1.0,
               "trench_depth_footprint_crm_m": float(SURF - fp.mean()),
               "wall_s": r["wall_s"], "rtf": r["rtf"], "t_sim": r["t_sim"], "stopped": r["stopped"], "n_sph": r["n_sph"],
               "spacing_m": sp}
    out = {"task": "bucket_dig", "solver": "chrono-crm", "tag": args.tag, "device": device_name(True), "git": git_rev(),
           "chrono_version": "pychrono 10.0.0 (conda projectchrono)", "params": {"spacing": sp, "depth": args.depth, "soil": r["soil"], "step": r["step"], "module": __doc__},
           "summary": summary, "initial_bed_top_readout_m": float(H0r.max()), "series_cols": ["t", "pivot_x", "pivot_z", "pitch", "Fx_N", "Fy_N", "Fz_N"],
           "series": ser[::2].round(4).tolist(), "H_crm": H.round(4).tolist()}
    print(summary)
    save(out, f"chrono-bucket-{args.tag}.json", args.out)
