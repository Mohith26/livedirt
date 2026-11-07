"""3D MLS-MPM, Drucker-Prager sand, with a kinematically driven rigid box tool and measured reaction force.

Constitutive model: Hencky-strain Drucker-Prager return map (Klar et al. 2016) written in the
granular-flow convention tau_eq <= mu p, with tau_eq = sqrt(s:s/2), p = -tr(sigma)/3. In Hencky
strain this is ||dev e|| <= -(3 la + 2 mu)/(2 mu) tr(e) * alpha with alpha = sqrt(2)/3 * mu_f.

Tool coupling: grid nodes inside the tool's signed-distance box get the tool velocity in the normal
direction with Coulomb friction mu_tool in the tangential direction. The momentum removed from the
soil at those nodes is the force the soil exerts on the tool (summed every step).

A symmetry plane at y = y0 (free-slip wall) halves the domain; forces are doubled on output.
"""
import math
import time

import numpy as np
import taichi as ti


def _arch(name):
    return {"gpu": ti.gpu, "cuda": ti.cuda, "metal": ti.metal, "vulkan": ti.vulkan, "cpu": ti.cpu}[name]


def run_plate(*, beta_deg, gamma_deg, mu_f=0.38, mu_tool=0.4, rho_bulk=1638.0, E=1.0e6, nu=0.3,
              dx=0.003, ppc_side=2, plate=(0.0381, 0.0254, 0.0064), bed=(0.22, 0.06, 0.15), speed=0.1,
              start_depth=None, travel=0.06, settle=0.1, arch="gpu", sample_every=0.002, seed=0,
              max_depth=0.10):
    """Drag a plate (attack angle beta, intrusion angle gamma, Li et al. 2013 conventions) through a bed.

    Conventions: x right, z up. Plate long axis at angle beta CCW from +x. Velocity direction
    (-cos gamma, 0, -sin gamma): gamma > 0 intrudes (moves down), gamma < 0 extracts.
    Returns time series of plate-centre depth and soil force on the plate (N, full plate).
    """
    g = 9.81
    pad = 3
    Lx, Ly, H = bed
    Lz = H + 0.03
    nx, ny, nz = int(math.ceil(Lx / dx)) + 2 * pad, int(math.ceil(Ly / dx)) + 2 * pad, int(math.ceil(Lz / dx)) + 2 * pad
    x0 = y0 = z0 = pad * dx
    surf = z0 + H
    ti.reset()
    ti.init(arch=_arch(arch), default_fp=ti.f32, random_seed=seed, offline_cache=True, device_memory_fraction=0.8)

    b, gm = math.radians(beta_deg), math.radians(gamma_deg)
    u_ax = np.array([math.cos(b), 0.0, math.sin(b)])
    w_ax = np.array([-math.sin(b), 0.0, math.cos(b)])
    vdir = np.array([-math.cos(gm), 0.0, -math.sin(gm)])
    hL, hW, hT = plate[0] / 2, plate[1] / 2, plate[2] / 2

    if start_depth is None:
        start_depth = 0.0 if gamma_deg >= 30 else (0.09 if gamma_deg <= -30 else 0.05)
    # limit travel so the centre depth stays within [0, max_depth]
    dz_per_m = math.sin(gm)
    if dz_per_m > 1e-6:
        travel = min(travel, (max_depth - start_depth) / dz_per_m)
    elif dz_per_m < -1e-6:
        travel = min(travel, (start_depth - 0.005) / -dz_per_m)
    cx0 = x0 + Lx / 2 + 0.5 * travel * math.cos(gm)
    c0 = np.array([cx0, y0, surf - start_depth])

    # particles: lattice over the bed, excluding the plate's initial volume
    sp = dx / ppc_side
    gx = (np.arange(int(Lx / sp)) + 0.5) * sp + x0
    gy = (np.arange(int(Ly / sp)) + 0.5) * sp + y0
    gz = (np.arange(int(H / sp)) + 0.5) * sp + z0
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    rel = pos - c0
    inside = (np.abs(rel @ u_ax) < hL) & (np.abs(rel[:, 1]) < hW) & (np.abs(rel @ w_ax) < hT)
    pos = pos[~inside]
    rng = np.random.default_rng(seed)
    pos += (rng.random(pos.shape) - 0.5) * 0.1 * sp
    n_p = len(pos)

    inv_dx = 1.0 / dx
    p_vol = sp ** 3
    p_mass = rho_bulk * p_vol
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    alpha = math.sqrt(2.0) / 3.0 * mu_f
    c_wave = math.sqrt((la + 2 * mu) / rho_bulk)
    dt = min(0.35 * dx / c_wave, 0.2 * dx / max(speed, 1e-6))

    x = ti.Vector.field(3, ti.f32, n_p)
    v = ti.Vector.field(3, ti.f32, n_p)
    C = ti.Matrix.field(3, 3, ti.f32, n_p)
    F = ti.Matrix.field(3, 3, ti.f32, n_p)
    grid_v = ti.Vector.field(3, ti.f32, (nx, ny, nz))
    grid_m = ti.field(ti.f32, (nx, ny, nz))
    pc = ti.Vector.field(3, ti.f32, ())  # plate centre
    pv = ti.Vector.field(3, ti.f32, ())  # plate velocity
    force = ti.Vector.field(3, ti.f32, ())
    damp = ti.field(ti.f32, ())
    ua = ti.Vector(list(u_ax.astype(np.float32)))
    wa = ti.Vector(list(w_ax.astype(np.float32)))

    @ti.func
    def plate_sdf(p):
        r = p - pc[None]
        q = ti.Vector([ti.abs(r.dot(ua)) - hL, ti.abs(r[1]) - hW, ti.abs(r.dot(wa)) - hT])
        outside = ti.Vector([ti.max(q[0], 0.0), ti.max(q[1], 0.0), ti.max(q[2], 0.0)]).norm()
        return outside + ti.min(ti.max(q[0], ti.max(q[1], q[2])), 0.0)

    @ti.func
    def plate_normal(p):
        e = 0.25 * dx
        n = ti.Vector([plate_sdf(p + ti.Vector([e, 0.0, 0.0])) - plate_sdf(p - ti.Vector([e, 0.0, 0.0])),
                       plate_sdf(p + ti.Vector([0.0, e, 0.0])) - plate_sdf(p - ti.Vector([0.0, e, 0.0])),
                       plate_sdf(p + ti.Vector([0.0, 0.0, e])) - plate_sdf(p - ti.Vector([0.0, 0.0, e]))])
        return n.normalized(1e-9)

    @ti.kernel
    def init_fields():
        for p in x:
            v[p] = ti.Vector([0.0, 0.0, 0.0])
            C[p] = ti.Matrix.zero(ti.f32, 3, 3)
            F[p] = ti.Matrix.identity(ti.f32, 3)

    @ti.kernel
    def substep(use_plate: ti.i32):
        for I in ti.grouped(grid_m):
            grid_v[I] = ti.Vector([0.0, 0.0, 0.0])
            grid_m[I] = 0.0
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            Fp = (ti.Matrix.identity(ti.f32, 3) + dt * C[p]) @ F[p]
            U, sig, V = ti.svd(Fp)
            e = ti.Vector([ti.log(ti.max(ti.abs(sig[0, 0]), 1e-4)), ti.log(ti.max(ti.abs(sig[1, 1]), 1e-4)),
                           ti.log(ti.max(ti.abs(sig[2, 2]), 1e-4))])
            tr = e.sum()
            ehat = e - tr / 3.0
            ehn = ehat.norm() + 1e-12
            if tr >= 0.0:
                e = ti.Vector([0.0, 0.0, 0.0])
            else:
                yp = ehn + (3.0 * la + 2.0 * mu) / (2.0 * mu) * tr * alpha
                if yp > 0.0:
                    e = e - (yp / ehn) * ehat
            F[p] = U @ ti.Matrix([[ti.exp(e[0]), 0.0, 0.0], [0.0, ti.exp(e[1]), 0.0], [0.0, 0.0, ti.exp(e[2])]]) @ V.transpose()
            tr2 = e.sum()
            tau_d = ti.Matrix([[2.0 * mu * e[0] + la * tr2, 0.0, 0.0], [0.0, 2.0 * mu * e[1] + la * tr2, 0.0],
                               [0.0, 0.0, 2.0 * mu * e[2] + la * tr2]])
            tau = U @ tau_d @ U.transpose()
            stress = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * tau
            affine = stress + p_mass * C[p]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
        for I in ti.grouped(grid_m):
            if grid_m[I] > 0:
                m = grid_m[I]
                gv = grid_v[I] / m
                gv[2] -= dt * g
                gv *= damp[None]
                if use_plate == 1:
                    xp = I.cast(float) * dx
                    if plate_sdf(xp) < 0.0:
                        n = plate_normal(xp)
                        vp = pv[None]
                        vr = gv - vp
                        vn = vr.dot(n)
                        if vn < 0.0:
                            vt = vr - vn * n
                            vtn = vt.norm()
                            vt_new = ti.Vector([0.0, 0.0, 0.0])
                            if vtn > 1e-12:
                                vt_new = vt * (ti.max(0.0, vtn + mu_tool * vn) / vtn)
                            gnew = vp + vt_new
                            force[None] += m * (gv - gnew) / dt
                            gv = gnew
                # walls: floor sticky (rough), sides free-slip, y=y0 symmetry
                if I[2] <= pad:
                    gv = ti.Vector([0.0, 0.0, 0.0])
                if I[0] <= pad and gv[0] < 0:
                    gv[0] = 0.0
                if I[0] >= nx - pad and gv[0] > 0:
                    gv[0] = 0.0
                if I[1] <= pad and gv[1] < 0:
                    gv[1] = 0.0
                if I[1] >= ny - pad and gv[1] > 0:
                    gv[1] = 0.0
                if I[2] >= nz - pad and gv[2] > 0:
                    gv[2] = 0.0
                grid_v[I] = gv
        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector([0.0, 0.0, 0.0])
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = offset.cast(float) - fx
                gv = grid_v[base + offset]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * gv
                new_C += 4.0 * inv_dx * weight * gv.outer_product(dpos)
            v[p] = new_v
            C[p] = new_C
            x[p] += dt * new_v

    x.from_numpy(pos.astype(np.float32))
    init_fields()
    pc[None] = ti.Vector(list(c0.astype(np.float32)))
    pv[None] = ti.Vector([0.0, 0.0, 0.0])

    # settle under gravity with the plate held still
    damp[None] = 0.995
    for _ in range(int(settle / dt)):
        substep(1)
    damp[None] = 1.0
    ti.sync()

    vel = (speed * vdir).astype(np.float32)
    pv[None] = ti.Vector(list(vel))
    n_steps = int(travel / speed / dt)
    every = max(1, int(sample_every / dt))
    rows = []
    t0 = time.perf_counter()
    c = c0.copy()
    acc = np.zeros(3)
    force[None] = ti.Vector([0.0, 0.0, 0.0])
    for s in range(n_steps):
        substep(1)
        c = c + vel * dt
        pc[None] = ti.Vector(list(c.astype(np.float32)))
        if (s + 1) % every == 0:
            f = force.to_numpy() / every  # mean force over the sample window
            force[None] = ti.Vector([0.0, 0.0, 0.0])
            rows.append([(s + 1) * dt, (s + 1) * dt * speed, surf - (c[2] - 0.5 * vel[2] * every * dt), 2 * f[0], 2 * f[2]])
    ti.sync()
    wall = time.perf_counter() - t0
    return {
        "series": np.array(rows),  # t, path length, centre depth |z|, Fx, Fz (soil on plate, full plate, N)
        "n_particles": n_p, "grid": [nx, ny, nz], "dt": dt, "steps": n_steps, "wall_s": wall,
        "rtf": n_steps * dt / wall, "start_depth": start_depth, "travel": travel,
        "params": {"mu_f": mu_f, "mu_tool": mu_tool, "rho_bulk": rho_bulk, "E": E, "nu": nu, "dx": dx,
                   "ppc_side": ppc_side, "plate": list(plate), "bed": list(bed), "speed": speed},
    }
