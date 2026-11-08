"""Reusable 3D MLS-MPM soil engine with a closed-loop rigid tool, for DigBench bucket and hybrid tasks.

Physics (same as mpm3d.run_plate, validated on the Li et al. 2013 plate stress maps):
  * Hencky-strain Drucker-Prager sand, granular convention tau_eq <= mu_f p (alpha = sqrt(2)/3 mu_f).
  * Tool = union of oriented boxes in a tool frame; pose = position + pitch about +y. Grid nodes inside the
    tool get the tool's rigid velocity (v + w x r) in the normal direction with Coulomb friction mu_tool.
    The momentum removed at those nodes is the soil force (and torque about the tool origin) on the tool.
  * Closed loop: the tool follows a commanded velocity, but the drive has a force limit (like a hydraulic
    relief valve). If the soil resistance along the commanded direction exceeds F_max, the commanded speed is
    scaled down for the next step, so the soil decides how far the bucket gets.

Heightfield coupling (for the hybrid model):
  * particles_from_heightfield(H, x0, y0, cell, zmin): fills soil columns of a heightfield inside the domain.
  * heightfield_from_particles(cell): top surface per column (max particle z + half spacing), volume-preserving
    alternative via column volume / packing.

The domain is the box [0, Lx] x [0, Ly] x [0, Lz] with `pad` cells of wall on each side. Floor is rough
(sticky), side walls free-slip; y = 0 can be a symmetry plane (forces doubled by the caller).
"""
import math
import time

import numpy as np
import taichi as ti


def _arch(name):
    return {"gpu": ti.gpu, "cuda": ti.cuda, "metal": ti.metal, "vulkan": ti.vulkan, "cpu": ti.cpu}[name]


class Sim3D:
    def __init__(self, size, dx, *, mu_f=0.38, mu_tool=0.4, rho_bulk=1600.0, E=1.0e6, nu=0.3, ppc_side=2,
                 max_particles=4_000_000, n_boxes=8, arch="gpu", symmetry_y=True, seed=0, tool_speed_max=1.0,
                 floor_sticky=True, wall_mu=0.0, mem_frac=0.85, volume_correction=False, yield_model="dp", phi_deg=None,
                 cohesion=0.0, tension=None):
        """yield_model: 'dp' (legacy Drucker-Prager cone, alpha = sqrt(2)/3 mu_f; equals Mohr-Coulomb with sin(phi) = mu_f in
        plane strain and is stronger than tan(phi) = mu_f there) or 'mc' (Mohr-Coulomb in principal Kirchhoff stresses with
        friction angle phi_deg (default atan(mu_f)), cohesion (Pa) and a hydrostatic tension limit (Pa, default the MC
        apex c cot(phi)); plastic flow is Lode-angle-preserving radial return at constant pressure (zero dilatancy)."""
        ti.reset()
        ti.init(arch=_arch(arch), default_fp=ti.f32, random_seed=seed, offline_cache=True, device_memory_fraction=mem_frac)
        self.dx, self.pad = dx, 3
        self.size = np.array(size, dtype=np.float64)
        self.n = [int(math.ceil(s / dx)) + 2 * self.pad for s in size]
        self.origin = np.array([self.pad * dx] * 3)
        self.sp = dx / ppc_side
        self.p_vol = self.sp ** 3
        self.rho = rho_bulk
        self.p_mass = rho_bulk * self.p_vol
        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu_f, self.mu_tool = mu_f, mu_tool
        alpha = math.sqrt(2.0) / 3.0 * mu_f
        self.yield_model = yield_model
        phi_r = math.atan(mu_f) if phi_deg is None else math.radians(phi_deg)
        self.phi_deg = math.degrees(phi_r)
        self.cohesion = float(cohesion)
        sphi, cphi = math.sin(phi_r), math.cos(phi_r)
        Kb = la + 2.0 * mu / 3.0
        apex_p = (cohesion * cphi / sphi) if sphi > 1e-6 else 0.0
        if tension is not None:
            apex_p = min(apex_p, float(tension))
        tr_apex = apex_p / Kb  # largest (tensile) log-volume the soil can carry
        use_mc = 1 if yield_model == "mc" else 0
        coh2 = 2.0 * cohesion * cphi
        c_wave = math.sqrt((la + 2 * mu) / rho_bulk)
        self.dt = min(0.35 * dx / c_wave, 0.2 * dx / tool_speed_max)
        self.symmetry_y = symmetry_y
        self.max_p = max_particles
        nx, ny, nz = self.n
        pad = self.pad
        inv_dx = 1.0 / dx
        dt = self.dt
        p_vol, p_mass = self.p_vol, self.p_mass
        g = 9.81
        sym = symmetry_y
        sticky = floor_sticky
        wmu = float(wall_mu)
        sym_y = 1 if symmetry_y else 0

        self.x = ti.Vector.field(3, ti.f32, max_particles)
        self.v = ti.Vector.field(3, ti.f32, max_particles)
        self.C = ti.Matrix.field(3, 3, ti.f32, max_particles)
        self.F = ti.Matrix.field(3, 3, ti.f32, max_particles)
        self.alive = ti.field(ti.i32, max_particles)
        # Tampubolon et al. 2017 volume correction: log-volume discarded by the plastic projection (tension cutoff /
        # yield) is remembered and added back to the trial trace, so spuriously bulked sand re-compacts freely
        self.vc = ti.field(ti.f32, max_particles)
        self.volume_correction = volume_correction
        self.n_p = ti.field(ti.i32, ())
        self.grid_v = ti.Vector.field(3, ti.f32, (nx, ny, nz))
        self.grid_m = ti.field(ti.f32, (nx, ny, nz))
        # per-column sticky floor (node z index); default = the flat floor at the domain bottom
        self.floor_k = ti.field(ti.i32, (nx, ny))
        self.floor_k.fill(pad)
        # tool: boxes in tool frame (centre, half extents), pose, twist
        self.nb = ti.field(ti.i32, ())
        self.box_c = ti.Vector.field(3, ti.f32, n_boxes)
        self.box_h = ti.Vector.field(3, ti.f32, n_boxes)
        self.box_a = ti.field(ti.f32, n_boxes)  # box pitch inside tool frame (rad)
        self.tpos = ti.Vector.field(3, ti.f32, ())
        self.tang = ti.field(ti.f32, ())  # tool pitch about +y (rad)
        self.tvel = ti.Vector.field(3, ti.f32, ())
        self.tomg = ti.field(ti.f32, ())  # pitch rate (rad/s)
        self.force = ti.Vector.field(3, ti.f32, ())
        self.torque = ti.Vector.field(3, ti.f32, ())
        self.damp = ti.field(ti.f32, ())
        self.damp[None] = 1.0
        x, v, C, F, alive = self.x, self.v, self.C, self.F, self.alive
        vcf = self.vc
        use_vc = 1 if volume_correction else 0
        grid_v, grid_m = self.grid_v, self.grid_m
        nb, box_c, box_h, box_a = self.nb, self.box_c, self.box_h, self.box_a
        tpos, tang, tvel, tomg = self.tpos, self.tang, self.tvel, self.tomg
        force, torque, damp, n_p = self.force, self.torque, self.damp, self.n_p
        floor_k = self.floor_k
        mu_t = mu_tool

        @ti.func
        def tool_sdf(p):
            # transform to tool frame (pitch about y)
            r = p - tpos[None]
            ca, sa = ti.cos(tang[None]), ti.sin(tang[None])
            lx = ca * r[0] + sa * r[2]
            lz = -sa * r[0] + ca * r[2]
            d = 1e9
            for b in range(nb[None]):
                qx0 = lx - box_c[b][0]
                qz0 = lz - box_c[b][2]
                cb, sb = ti.cos(box_a[b]), ti.sin(box_a[b])
                bx = cb * qx0 + sb * qz0
                bz = -sb * qx0 + cb * qz0
                q = ti.Vector([ti.abs(bx) - box_h[b][0], ti.abs(r[1] - box_c[b][1]) - box_h[b][1], ti.abs(bz) - box_h[b][2]])
                outside = ti.Vector([ti.max(q[0], 0.0), ti.max(q[1], 0.0), ti.max(q[2], 0.0)]).norm()
                d = ti.min(d, outside + ti.min(ti.max(q[0], ti.max(q[1], q[2])), 0.0))
            return d

        @ti.func
        def tool_normal(p):
            e = 0.25 * dx
            n = ti.Vector([tool_sdf(p + ti.Vector([e, 0.0, 0.0])) - tool_sdf(p - ti.Vector([e, 0.0, 0.0])),
                           tool_sdf(p + ti.Vector([0.0, e, 0.0])) - tool_sdf(p - ti.Vector([0.0, e, 0.0])),
                           tool_sdf(p + ti.Vector([0.0, 0.0, e])) - tool_sdf(p - ti.Vector([0.0, 0.0, e]))])
            return n.normalized(1e-9)

        @ti.kernel
        def substep(use_tool: ti.i32):
            for I in ti.grouped(grid_m):
                grid_v[I] = ti.Vector([0.0, 0.0, 0.0])
                grid_m[I] = 0.0
            for p in range(n_p[None]):
                if alive[p] == 1:
                    base = (x[p] * inv_dx - 0.5).cast(int)
                    fx = x[p] * inv_dx - base.cast(float)
                    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                    Fp = (ti.Matrix.identity(ti.f32, 3) + dt * C[p]) @ F[p]
                    U, sig, V = ti.svd(Fp)
                    e = ti.Vector([ti.log(ti.max(ti.abs(sig[0, 0]), 1e-4)), ti.log(ti.max(ti.abs(sig[1, 1]), 1e-4)),
                                   ti.log(ti.max(ti.abs(sig[2, 2]), 1e-4))])
                    tr_trial = e.sum()
                    if ti.static(use_vc == 1):
                        e = e + vcf[p] / 3.0
                    tr = e.sum()
                    ehat = e - tr / 3.0
                    ehn = ehat.norm() + 1e-12
                    if ti.static(use_mc == 1):
                        if tr >= tr_apex:
                            e = ti.Vector([tr_apex / 3.0, tr_apex / 3.0, tr_apex / 3.0])
                        else:
                            sdev = 2.0 * mu * ehat
                            smax = ti.max(sdev[0], ti.max(sdev[1], sdev[2]))
                            smin = ti.min(sdev[0], ti.min(sdev[1], sdev[2]))
                            pk = -Kb * tr  # Kirchhoff pressure, compression positive
                            den = (smax - smin) + (smax + smin) * sphi
                            cap = 2.0 * pk * sphi + coh2
                            if den > cap and den > 1e-12:
                                e = tr / 3.0 + (ti.max(cap, 0.0) / den) * ehat
                    else:
                        if tr >= 0.0:
                            e = ti.Vector([0.0, 0.0, 0.0])
                        else:
                            yp = ehn + (3.0 * la + 2.0 * mu) / (2.0 * mu) * tr * alpha
                            if yp > 0.0:
                                e = e - (yp / ehn) * ehat
                    if ti.static(use_vc == 1):
                        vcf[p] += tr_trial - e.sum()
                    F[p] = U @ ti.Matrix([[ti.exp(e[0]), 0.0, 0.0], [0.0, ti.exp(e[1]), 0.0], [0.0, 0.0, ti.exp(e[2])]]) @ V.transpose()
                    tr2 = e.sum()
                    tau_d = ti.Matrix([[2.0 * mu * e[0] + la * tr2, 0.0, 0.0], [0.0, 2.0 * mu * e[1] + la * tr2, 0.0],
                                       [0.0, 0.0, 2.0 * mu * e[2] + la * tr2]])
                    tau = U @ tau_d @ U.transpose()
                    affine = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * tau + p_mass * C[p]
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
                    if use_tool == 1:
                        xp = I.cast(float) * dx
                        if tool_sdf(xp) < 0.0:
                            n = tool_normal(xp)
                            r = xp - tpos[None]
                            # rigid velocity for pitch rate d(theta)/dt with x' = c x - s z, z' = s x + c z:
                            # d r / dt = theta_dot * (-r_z, 0, r_x)
                            vp = tvel[None] + ti.Vector([-tomg[None] * r[2], 0.0, tomg[None] * r[0]])
                            vr = gv - vp
                            vn = vr.dot(n)
                            if vn < 0.0:
                                vt = vr - vn * n
                                vtn = vt.norm()
                                vt_new = ti.Vector([0.0, 0.0, 0.0])
                                if vtn > 1e-12:
                                    vt_new = vt * (ti.max(0.0, vtn + mu_t * vn) / vtn)
                                gnew = vp + vt_new
                                f = m * (gv - gnew) / dt
                                force[None] += f
                                torque[None] += r.cross(f)
                                gv = gnew
                    if I[2] <= floor_k[I[0], I[1]]:
                        if ti.static(sticky):
                            gv = ti.Vector([0.0, 0.0, 0.0])
                        elif gv[2] < 0:
                            gv[2] = 0.0
                    # side walls: Coulomb friction wall_mu (an active-zone boundary stands in for the soil beyond it);
                    # the y = 0 symmetry plane is always free-slip
                    for ax in ti.static(range(2)):
                        lo_hit = I[ax] <= pad and gv[ax] < 0
                        hi_hit = I[ax] >= (nx if ax == 0 else ny) - pad and gv[ax] > 0
                        if lo_hit or hi_hit:
                            vn = ti.abs(gv[ax])
                            gv[ax] = 0.0
                            if not (ax == 1 and lo_hit and sym_y == 1):
                                vt = gv.norm()
                                if vt > 1e-12:
                                    gv = gv * (ti.max(0.0, vt - wmu * vn) / vt)
                    if I[2] >= nz - pad and gv[2] > 0:
                        gv[2] = 0.0
                    grid_v[I] = gv
            for p in range(n_p[None]):
                if alive[p] == 1:
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

        @ti.kernel
        def reset_particles(start: ti.i32, end: ti.i32):
            for p in range(start, end):
                v[p] = ti.Vector([0.0, 0.0, 0.0])
                C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                F[p] = ti.Matrix.identity(ti.f32, 3)
                vcf[p] = 0.0
                alive[p] = 1

        @ti.kernel
        def mark_inside_tool():
            for p in range(n_p[None]):
                if alive[p] == 1 and tool_sdf(x[p]) < 0.0:
                    alive[p] = 0

        self._substep = substep
        self._reset = reset_particles
        self._mark_inside_tool = mark_inside_tool
        self.n_p[None] = 0
        self.nb[None] = 0
        self.t = 0.0

    # ---------------- setup ----------------
    def add_particles(self, pos):
        pos = np.asarray(pos, dtype=np.float32)
        n0 = int(self.n_p[None])
        n1 = n0 + len(pos)
        if n1 > self.max_p:
            raise RuntimeError(f"particle budget exceeded: {n1} > {self.max_p}")
        buf = self.x.to_numpy()
        buf[n0:n1] = pos + self.origin.astype(np.float32)
        self.x.from_numpy(buf)
        self.n_p[None] = n1
        self._reset(n0, n1)
        return n1 - n0

    def fill_box(self, lo, hi, rng_seed=0):
        sp = self.sp
        axes = [(np.arange(int(round((hi[i] - lo[i]) / sp))) + 0.5) * sp + lo[i] for i in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        pos += (np.random.default_rng(rng_seed).random(pos.shape) - 0.5) * 0.1 * sp
        return self.add_particles(pos)

    def set_floor(self, Hf, cell):
        """Per-column sticky floor: grid nodes at or below Hf[ix, iy] (domain metres, cell = Hf spacing) are held
        fixed, standing in for the static soil beneath (soil below it need not be simulated)."""
        nx, ny = self.n[0], self.n[1]
        xs = (np.arange(nx) - self.pad) * self.dx
        ys = (np.arange(ny) - self.pad) * self.dx
        ix = np.clip((xs / cell).astype(int), 0, Hf.shape[0] - 1)
        iy = np.clip((ys / cell).astype(int), 0, Hf.shape[1] - 1)
        h = Hf[ix[:, None], iy[None, :]]
        self.floor_k.from_numpy((self.pad + np.floor(h / self.dx)).astype(np.int32))

    def fill_heightfield(self, H, cell, x_off=0.0, y_off=0.0, rng_seed=0, bottom=None):
        """Fill soil below heightfield H[ix, iy] (domain coordinates, metres; cell = H grid spacing); with `bottom`
        (same grid) only the soil between bottom and H is created."""
        sp = self.sp
        xs = (np.arange(int(H.shape[0] * cell / sp)) + 0.5) * sp
        ys = (np.arange(int(H.shape[1] * cell / sp)) + 0.5) * sp
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        top = H[np.clip((X / cell).astype(int), 0, H.shape[0] - 1), np.clip((Y / cell).astype(int), 0, H.shape[1] - 1)]
        nz = np.maximum(0, np.floor(top / sp).astype(int))
        k0 = np.zeros_like(nz)
        if bottom is not None:
            bot = bottom[np.clip((X / cell).astype(int), 0, H.shape[0] - 1), np.clip((Y / cell).astype(int), 0, H.shape[1] - 1)]
            k0 = np.clip(np.floor(bot / sp).astype(int), 0, None)
        cnt = np.maximum(0, nz - k0)
        cols = np.repeat(np.arange(X.size), cnt.ravel())
        k = np.concatenate([np.arange(a, b) for a, b in zip(k0.ravel(), nz.ravel()) if b > a]) if cols.size else np.zeros(0, int)
        pos = np.stack([X.ravel()[cols] + x_off, Y.ravel()[cols] + y_off, (k + 0.5) * sp], 1)
        pos += (np.random.default_rng(rng_seed).random(pos.shape) - 0.5) * 0.1 * sp
        return self.add_particles(pos)

    def set_tool(self, boxes):
        """boxes: list of (centre(3), half_extents(3), pitch_rad) in the tool frame."""
        self.nb[None] = len(boxes)
        for i, (c, h, a) in enumerate(boxes):
            self.box_c[i] = ti.Vector(list(map(float, c)))
            self.box_h[i] = ti.Vector(list(map(float, h)))
            self.box_a[i] = float(a)

    def set_pose(self, pos, pitch):
        self.tpos[None] = ti.Vector(list((np.asarray(pos) + self.origin).astype(np.float32)))
        self.tang[None] = float(pitch)

    def remove_inside_tool(self):
        self._mark_inside_tool()

    # ---------------- stepping ----------------
    def settle(self, seconds, use_tool=1):
        self.damp[None] = 0.995
        self.tvel[None] = ti.Vector([0.0, 0.0, 0.0])
        self.tomg[None] = 0.0
        for _ in range(int(seconds / self.dt)):
            self._substep(use_tool)
        self.damp[None] = 1.0
        ti.sync()

    def drive(self, path_fn, duration, *, f_max=None, limiter=None, sample_every=0.01, use_tool=1, speed_floor=0.0, on_sample=None):
        """Closed-loop tool drive. path_fn(t_cmd) -> (pos(3), pitch); the commanded trajectory parameter
        t_cmd advances at rate s in [0, 1]: s drops when the soil force opposing motion exceeds f_max (relief
        valve), so the soil can stall the tool. Returns a time series."""
        dt = self.dt
        t_cmd = 0.0
        s = 1.0
        rows = []
        every = max(1, int(sample_every / dt))
        self.force[None] = ti.Vector([0.0, 0.0, 0.0])
        self.torque[None] = ti.Vector([0.0, 0.0, 0.0])
        pos0, a0 = path_fn(0.0)
        self.set_pose(pos0, a0)
        n_steps = int(duration / dt)
        f_win = np.zeros(3)
        t0 = time.perf_counter()
        for k in range(n_steps):
            pa, aa = path_fn(t_cmd)
            pb, ab = path_fn(t_cmd + s * dt)
            vel = (np.asarray(pb) - np.asarray(pa)) / dt
            omg = (ab - aa) / dt
            self.tvel[None] = ti.Vector(list(vel.astype(np.float32)))
            self.tomg[None] = float(omg)
            self._substep(use_tool)
            t_cmd += s * dt
            self.set_pose(pb, ab)
            if (k + 1) % every == 0:
                fwin = self.force.to_numpy() / every
                twin = self.torque.to_numpy() / every
                self.force[None] = ti.Vector([0.0, 0.0, 0.0])
                self.torque[None] = ti.Vector([0.0, 0.0, 0.0])
                mult = 2.0 if self.symmetry_y else 1.0
                fx, fz = mult * fwin[0], mult * fwin[2]
                ty = mult * twin[1]
                exceeded = None
                if limiter is not None:
                    exceeded = limiter(fx, fz, ty, pb, ab)
                elif f_max is not None:
                    vdir = vel / (np.linalg.norm(vel) + 1e-12)
                    exceeded = -(fx * vdir[0] + fz * vdir[2]) > f_max
                if exceeded is not None:
                    # relief valve: proportional back-off above the limit, recover below it
                    if exceeded:
                        s = max(speed_floor, s * 0.85)
                    else:
                        s = min(1.0, s + 0.05)
                rows.append([(k + 1) * dt, t_cmd, *pb, ab, fx, fz, ty, s])
                if on_sample is not None and on_sample(self, rows[-1]) is True:
                    n_steps = k + 1  # caller asked to stop (e.g. the commanded path is complete)
                    break
        ti.sync()
        self.t += n_steps * dt
        return {"series": np.array(rows), "wall_s": time.perf_counter() - t0, "steps": n_steps, "dt": dt}

    def remove_where(self, fn):
        """Kill live particles where fn(positions (N,3) in domain coords) is True (e.g. dump the bucket)."""
        n = int(self.n_p[None])
        xs = self.x.to_numpy()[:n] - self.origin.astype(np.float32)
        al = self.alive.to_numpy()
        live = al[:n] == 1
        kill = np.zeros(n, bool)
        kill[live] = fn(xs[live])
        al[:n][kill] = 0
        self.alive.from_numpy(al)
        return int(kill.sum())

    # ---------------- readout ----------------
    def particles(self):
        n = int(self.n_p[None])
        xs = self.x.to_numpy()[:n] - self.origin.astype(np.float32)
        al = self.alive.to_numpy()[:n] == 1
        return xs[al]

    def heightfield(self, cell, shape, mode="top"):
        """Surface per column: 'top' = max particle z + sp/2; 'volume' = particle volume / cell area (bank-volume
        equivalent, i.e. conserves mass exactly)."""
        p = self.particles()
        ix = np.clip((p[:, 0] / cell).astype(int), 0, shape[0] - 1)
        iy = np.clip((p[:, 1] / cell).astype(int), 0, shape[1] - 1)
        H = np.zeros(shape)
        if mode == "top":
            np.maximum.at(H, (ix, iy), p[:, 2] + 0.5 * self.sp)
        else:
            np.add.at(H, (ix, iy), self.p_vol / (cell * cell))
        return H


def connected_surface(g, nxy, cell, sp, zmax=4.0):
    """Top of the soil column connected to the bed (stops at the first two-bin gap), so airborne spill is not
    counted as ground. g: (N, 3) particle positions in domain coordinates."""
    nz = int(zmax / sp) + 2
    occ = np.zeros((nxy[0], nxy[1], nz), dtype=bool)
    ix = np.clip((g[:, 0] / cell).astype(int), 0, nxy[0] - 1)
    iy = np.clip((g[:, 1] / cell).astype(int), 0, nxy[1] - 1)
    iz = np.clip((g[:, 2] / sp).astype(int), 0, nz - 1)
    occ[ix, iy, iz] = True
    gap = ~occ[:, :, :-1] & ~occ[:, :, 1:]
    first = np.where(gap.any(axis=2), gap.argmax(axis=2), nz - 1)
    return (first * sp).astype(np.float64)
