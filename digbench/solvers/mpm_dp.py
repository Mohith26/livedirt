"""2D plane-strain MLS-MPM with Drucker-Prager sand plasticity.

Method: MLS-MPM / APIC transfer (Hu et al., SIGGRAPH 2018) with the Hencky-strain
Drucker-Prager return map of Klar et al., "Drucker-Prager elastoplasticity for sand
animation", SIGGRAPH 2016. Cohesionless: tensile states return to the stress-free apex.

Material parameters are physical inputs (friction angle, bulk density, stiffness), not
fitted to the benchmark. The friction angle defaults to the measured repose angle.
"""
import math
import time

import numpy as np
import taichi as ti


def _arch(name):
    return {"gpu": ti.gpu, "cuda": ti.cuda, "metal": ti.metal, "vulkan": ti.vulkan, "cpu": ti.cpu}[name]


def run_column_collapse(case, *, dx=0.0025, ppc_side=2, E=1.0e6, nu=0.3, phi_deg=23.5,
                        rho_bulk=1600.0, yield_fit="plane_strain", floor="sticky", floor_mu=None, gate_settle=0.15,
                        arch="gpu", max_time=None, stop_speed=0.005, seed=0, snapshot_every=None):
    """Simulate one column collapse. Returns final particle positions (wall/floor at 0) and timing."""
    H0, L0 = case["H0"], case["L0"]
    g = 9.81
    pad = 3
    Lx = L0 + 3.2 * H0 + 0.05
    Ly = 1.15 * H0 + 0.02
    nx = int(math.ceil(Lx / dx)) + 2 * pad
    ny = int(math.ceil(Ly / dx)) + 2 * pad
    x0 = pad * dx  # wall position
    y0 = pad * dx  # floor position

    ti.reset()
    ti.init(arch=_arch(arch), default_fp=ti.f32, random_seed=seed, offline_cache=True)

    # particle lattice
    sp = dx / ppc_side
    nxp = int(round(L0 / sp))
    nyp = int(round(H0 / sp))
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid((np.arange(nxp) + 0.5) * sp, (np.arange(nyp) + 0.5) * sp, indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel()], 1)
    pos += (rng.random(pos.shape) - 0.5) * 0.1 * sp
    pos[:, 0] += x0
    pos[:, 1] += y0
    n_p = len(pos)

    inv_dx = 1.0 / dx
    p_vol = sp * sp
    p_mass = rho_bulk * p_vol
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    sphi = math.sin(math.radians(phi_deg))
    if yield_fit == "plane_strain":
        # exact Mohr-Coulomb in 2D plane strain with Hencky strain: |e1-e2| <= -(1+la/mu) tr(e) sin(phi)
        alpha = sphi / math.sqrt(2.0)
    else:  # Klar et al. 2016 3D cone fit
        alpha = math.sqrt(2.0 / 3.0) * 2 * sphi / (3 - sphi)
    c_wave = math.sqrt((la + 2 * mu) / rho_bulk)
    dt = 0.4 * dx / c_wave
    sticky = floor == "sticky"
    fmu = 0.0 if floor_mu is None else float(floor_mu)

    x = ti.Vector.field(2, ti.f32, n_p)
    v = ti.Vector.field(2, ti.f32, n_p)
    C = ti.Matrix.field(2, 2, ti.f32, n_p)
    F = ti.Matrix.field(2, 2, ti.f32, n_p)
    grid_v = ti.Vector.field(2, ti.f32, (nx, ny))
    grid_m = ti.field(ti.f32, (nx, ny))
    gate_x = ti.field(ti.f32, ())
    damp = ti.field(ti.f32, ())
    vmax = ti.field(ti.f32, ())

    @ti.kernel
    def init_fields():
        for p in x:
            v[p] = ti.Vector([0.0, 0.0])
            C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            F[p] = ti.Matrix.identity(ti.f32, 2)

    @ti.kernel
    def substep():
        for i, j in grid_m:
            grid_v[i, j] = ti.Vector([0.0, 0.0])
            grid_m[i, j] = 0.0
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            Fp = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
            U, sig, V = ti.svd(Fp)
            e = ti.Vector([ti.log(ti.max(ti.abs(sig[0, 0]), 1e-4)), ti.log(ti.max(ti.abs(sig[1, 1]), 1e-4))])
            tr = e[0] + e[1]
            ehat = e - tr / 2.0
            ehn = ehat.norm() + 1e-12
            if tr >= 0.0:
                e = ti.Vector([0.0, 0.0])
            else:
                yp = ehn + (2.0 * la + 2.0 * mu) / (2.0 * mu) * tr * alpha
                if yp > 0.0:
                    e = e - (yp / ehn) * ehat
            Sig = ti.Matrix([[ti.exp(e[0]), 0.0], [0.0, ti.exp(e[1])]])
            F[p] = U @ Sig @ V.transpose()
            tr2 = e[0] + e[1]
            tau_d = ti.Matrix([[2.0 * mu * e[0] + la * tr2, 0.0], [0.0, 2.0 * mu * e[1] + la * tr2]])
            tau = U @ tau_d @ U.transpose()
            stress = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * tau
            affine = stress + p_mass * C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
        for i, j in grid_m:
            if grid_m[i, j] > 0:
                gv = grid_v[i, j] / grid_m[i, j]
                gv[1] -= dt * g
                gv *= damp[None]
                if j <= pad:  # floor
                    if ti.static(sticky):
                        gv = ti.Vector([0.0, 0.0])
                    else:
                        if gv[1] < 0:
                            vn = gv[1]
                            gv[1] = 0.0
                            vt = ti.abs(gv[0])
                            if vt > 1e-12:
                                gv[0] = gv[0] / vt * ti.max(0.0, vt + fmu * vn)
                if i <= pad and gv[0] < 0:
                    gv[0] = 0.0
                if i >= nx - pad and gv[0] > 0:
                    gv[0] = 0.0
                if j >= ny - pad and gv[1] > 0:
                    gv[1] = 0.0
                xg = i * dx
                if xg >= gate_x[None] and xg < gate_x[None] + 2 * dx and gv[0] > 0:  # gate
                    gv[0] = 0.0
                grid_v[i, j] = gv
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix.zero(ti.f32, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = offset.cast(float) - fx
                gv = grid_v[base + offset]
                weight = w[i][0] * w[j][1]
                new_v += weight * gv
                new_C += 4.0 * inv_dx * weight * gv.outer_product(dpos)
            v[p] = new_v
            C[p] = new_C
            x[p] += dt * new_v

    @ti.kernel
    def max_speed():
        vmax[None] = 0.0
        for p in v:
            ti.atomic_max(vmax[None], v[p].norm())

    x.from_numpy(pos.astype(np.float32))
    init_fields()

    # phase 1: settle behind the gate (geostatic stress), damped
    gate_x[None] = x0 + L0 + 0.5 * dx
    damp[None] = 0.995
    n_settle = int(gate_settle / dt)
    for _ in range(n_settle):
        substep()
    ti.sync()

    # phase 2: instantaneous gate removal (Series A free-fall regime)
    gate_x[None] = 1e9
    damp[None] = 1.0
    t_ff = math.sqrt(2 * H0 / g)
    t_max = max_time if max_time is not None else max(1.0, 12 * t_ff)
    n_max = int(t_max / dt)
    check = max(50, int(0.01 / dt))
    snaps = []
    t0 = time.perf_counter()
    steps = 0
    stopped = "t_max"
    while steps < n_max:
        for _ in range(check):
            substep()
        steps += check
        t_sim = steps * dt
        if snapshot_every and (steps // check) % max(1, int(snapshot_every / (check * dt))) == 0:
            snaps.append((t_sim, x.to_numpy()[:, :] - np.array([x0, y0])))
        if t_sim > 4 * t_ff:
            max_speed()
            if vmax[None] < stop_speed:
                stopped = "at_rest"
                break
    ti.sync()
    wall = time.perf_counter() - t0
    final = x.to_numpy() - np.array([x0, y0], dtype=np.float32)
    J = np.linalg.det(F.to_numpy())
    return {
        "positions": final,
        "spacing": sp,
        "n_particles": n_p,
        "grid": [nx, ny],
        "dt": dt,
        "steps": steps,
        "t_sim": steps * dt,
        "wall_s": wall,
        "rtf": (steps * dt) / wall,
        "stopped": stopped,
        "volume_ratio": float(J.sum() * p_vol / (n_p * p_vol)),
        "params": {"dx": dx, "ppc_side": ppc_side, "E": E, "nu": nu, "phi_deg": phi_deg, "rho_bulk": rho_bulk,
                   "yield_fit": yield_fit, "floor": floor, "floor_mu": floor_mu, "gate_settle": gate_settle, "alpha_dp": alpha},
        "snapshots": snaps,
    }
