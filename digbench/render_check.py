#!/usr/bin/env python3
"""DigBench item 7 (rendering half): does LiveDirt's terrain deformation damage splat regions it should not touch?

Subcommands (run in this order):
  export  (local, needs pycolmap)  COLMAP sparse model + held-out list  -> inputs.npz
  prep    (local, numpy + node)    ground plane, trench placement, real Heightfield dig -> prep.npz
  render  (GPU, gsplat)            original vs deformed renders from every held-out camera, region metrics -> json

Everything the numbers depend on is an ASSUMPTION unless stated; see digbench/RENDER_CHECK.md. In short:
  * Our splat has no metric scale. We ASSUME metres-per-unit s0 such that the robust radius (90th percentile) of the
    ground-plane sparse points equals 24.5 m (scene.config.json siteRadius of the app's other scene). Sensitivity: x0.5, x2.
  * Ground = RANSAC plane through sparse points; H0 (original terrain) is that flat plane.
  * Post-dig ground truth does not exist: changed-region numbers measure departure from the pre-dig photo, NOT correctness.

deform.ts port (app/src/render/deform.ts, vertex-shader body lines 76-112), applied per splat in metric ground frame
(x,z in the ground plane, y up):
  L78  st  = bilinear(tStatic, c.xz)           -> H0 = 0 (flat plane assumption)
  L79  code = static object code               -> 0 everywhere (no object segmentation; trench placed clear of objects)
  L80  dh  = bilinear(tDyn, c.xz)              -> dH (r) and loose thickness (g) from the REAL app/src/sim/heightfield.ts
  L81  rel = c.y - st.r
  L82-83 bind = (1 - smoothstep(bindLow=0.12, bindHigh=0.45, rel)) * (1 - onObject)  (grade vec: terrainSystem.ts L62)
  L87  center.y += bind * dh.r                 (translation only; scale/rotation untouched, as in deform.ts)
  L100-101 changed = max(-dh.r, dh.r, dh.g); alpha *= 1 - bind*smoothstep(0.05, 0.16, changed)
Not ported: L88-99 / L102-111 colour re-tint, wetness, overlays, global grade. Reason: the re-tint palette is scene
metadata we do not have for this splat; it only touches splats with dug>0 or loose>0 (already inside the changed region);
global grade is identity by default (terrainSystem.ts L65). Also not rendered: the CPU "generated soil layer" that the
alpha fade hands over to in the app, so the changed region here shows holes, not what the app shows.
"""
import sys, os, json, math, subprocess, shutil, argparse
import numpy as np

DIL_PRIMARY, DIL_WIDE = 24, 64      # px dilation of the projected trench volume
BIND_LOW, BIND_HIGH = 0.12, 0.45    # terrainSystem.ts L62: grade = (tol, lumP50, 0.12, 0.45)
TRENCH = dict(length=3.0, width=1.0, depth=0.6)   # metres (assumed scale)
SITE_RADIUS_M = 24.5                # pipeline/scene.config.json siteRadius, used ONLY to set the assumed scale


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------------------------- export
def cmd_export(a):
    import pycolmap
    r = pycolmap.Reconstruction(os.path.join(a.work, 'sparse/best'))
    ims = sorted(r.images.values(), key=lambda i: i.name)
    cams = {c.camera_id: c for c in r.cameras.values()}
    W2C, K, wh, names, dist = [], [], [], [], []
    for im in ims:
        c = cams[im.camera_id]
        T = np.eye(4); T[:3] = im.cam_from_world().matrix()
        W2C.append(T); K.append(c.calibration_matrix()); wh.append([c.width, c.height]); names.append(im.name)
        dist.append(c.params[-1] if c.model.name == 'SIMPLE_RADIAL' else 0.0)
    pts = list(r.points3D.values())
    held = sorted(os.path.splitext(f)[0] + '.jpg' for f in os.listdir(os.path.join(a.work, 'out/eval_7000')) if f.endswith('.png'))
    np.savez(a.out, names=names, W2C=np.array(W2C), K=np.array(K), wh=np.array(wh), k1=np.array(dist),
             P=np.array([p.xyz for p in pts]), err=np.array([p.error for p in pts]), tl=np.array([p.track.length() for p in pts]),
             held=held)
    print('exported', len(names), 'cams,', len(pts), 'points,', len(held), 'held-out')


# ----------------------------------------------------------------------------------------------- prep
HELPER_TS = r"""
import { Heightfield } from './heightfield.ts';
import { SOILS } from './soil.ts';
import fs from 'node:fs';
const cfg = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const N = cfg.N, cell = cfg.cell, half = cfg.half;
const hf = new Heightfield({ N, cell, half, H0: new Float32Array(N * N), soil: SOILS.find((s) => s.id === cfg.soil) ?? SOILS[0] });
const hw = cfg.length / 2;
hf.setDesign({ p0: [-hw, 0], p1: [hw, 0], bottomWidth: cfg.width, depth: cfg.depth, sideSlope: 0 });
hf.beginStroke();
let prev = -1, passes = 0;
// scripted excavation: radius-0.45 tool, 0.15 m per pass, sweeping three lines along the trench, grade-controlled
while (passes < 40) {
  for (const zl of [-0.3, 0, 0.3]) for (let x = -hw; x <= hw + 1e-9; x += 0.1) hf.dig(x, zl, 0.45, 0.15, true, true);
  passes++;
  if (Math.abs(hf.ledger.bankCut - prev) < 1e-6) break;
  prev = hf.ledger.bankCut;
}
hf.settle(800);
const dH = new Float32Array(N * N), L = new Float32Array(N * N);
let minH = 0;
for (let k = 0; k < N * N; k++) { dH[k] = hf.B[k] + hf.L[k] - hf.H0[k]; L[k] = hf.L[k]; minH = Math.min(minH, dH[k]); }
fs.writeFileSync(cfg.out + '.dH.f32', Buffer.from(dH.buffer));
fs.writeFileSync(cfg.out + '.L.f32', Buffer.from(L.buffer));
console.log(JSON.stringify({ ledger: hf.ledger, passes, minDH: minH, soil: hf.soil.id, grade: hf.gradeStats() }));
"""


def run_dig(app_sim, tmp, tag, N, cell, half, soil='wet-clay'):
    d = os.path.join(tmp, 'sim'); os.makedirs(d, exist_ok=True)
    for f in ('heightfield.ts', 'soil.ts'):
        s = open(os.path.join(app_sim, f)).read()
        s = s.replace("import { SoilType, SOILS } from './soil';", "import type { SoilType } from './soil.ts';\nimport { SOILS } from './soil.ts';")
        open(os.path.join(d, f), 'w').write(s)
    open(os.path.join(d, 'dig.ts'), 'w').write(HELPER_TS)
    out = os.path.join(tmp, tag)
    cfg = dict(N=N, cell=cell, half=half, soil=soil, out=out, **TRENCH)
    json.dump(cfg, open(out + '.cfg.json', 'w'))
    node = shutil.which('node') or '/Users/mohithgajjela/.aside/runtime/bin/node'
    r = subprocess.run([node, os.path.join(d, 'dig.ts'), out + '.cfg.json'], capture_output=True, text=True)
    if r.returncode: raise SystemExit('node dig failed: ' + r.stderr[-800:])
    info = json.loads(r.stdout.strip().splitlines()[-1])
    dH = np.fromfile(out + '.dH.f32', np.float32).reshape(N, N)   # index [j(z), i(x)]
    L = np.fromfile(out + '.L.f32', np.float32).reshape(N, N)
    return dH, L, info


def fit_ground(P, tl, err, up_hint, rng):
    keep = (tl >= 3) & (err < 1.5)
    Q = P[keep]
    ext = np.percentile(np.linalg.norm(Q - np.median(Q, 0), axis=1), 90)
    thr = 0.004 * ext
    best, bn = None, -1
    for _ in range(4000):
        i = rng.choice(len(Q), 3, replace=False)
        n = np.cross(Q[i[1]] - Q[i[0]], Q[i[2]] - Q[i[0]]); nn = np.linalg.norm(n)
        if nn < 1e-12: continue
        n /= nn
        if n @ up_hint < 0: n = -n
        if n @ up_hint < math.cos(math.radians(25)): continue
        cnt = int((np.abs((Q - Q[i[0]]) @ n) < thr).sum())
        if cnt > bn: bn, best = cnt, (n, Q[i[0]])
    n, p0 = best
    for _ in range(5):  # least-squares refinement on inliers
        m = np.abs((Q - p0) @ n) < thr
        c = Q[m].mean(0); u, s, vt = np.linalg.svd(Q[m] - c); n2 = vt[2]
        n = n2 if n2 @ up_hint > 0 else -n2; p0 = c
    m = np.abs((Q - p0) @ n) < thr
    return n, p0, m.sum(), len(Q), thr, ext


def project(W2C, K, X):  # X (M,3) world -> (u,v,z)
    Xc = X @ W2C[:3, :3].T + W2C[:3, 3]
    z = Xc[:, 2]
    uv = Xc @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-9), z


def cmd_prep(a):
    z = np.load(a.inp, allow_pickle=True)
    W2C, K, wh, P, tl, err, names, held = z['W2C'], z['K'], z['wh'], z['P'], z['tl'], z['err'], list(z['names']), list(z['held'])
    rng = np.random.default_rng(0)
    cam_c = -np.einsum('nji,nj->ni', W2C[:, :3, :3], W2C[:, :3, 3])        # camera centres (world)
    up_hint = -np.mean(W2C[:, 1, :3], axis=0); up_hint /= np.linalg.norm(up_hint)  # camera y points down
    n, p0, ninl, nq, thr, ext = fit_ground(P, tl, err, up_hint, rng)
    # ground frame: e1,e2 span plane, n up. origin = plane point below inlier median
    e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    Rg = np.stack([e1, n, e2])                                              # rows: x, y(up), z   (world->ground, unit COLMAP units)
    Q = P[(tl >= 3) & (err < 1.5)]
    inl = Q[np.abs((Q - p0) @ n) < thr]
    med = np.median(inl, 0); origin = med - ((med - p0) @ n) * n
    G = (inl - origin) @ Rg.T
    r90 = np.percentile(np.hypot(G[:, 0], G[:, 2]), 90)
    s0 = SITE_RADIUS_M / r90                                                # metres per COLMAP unit (ASSUMED)
    print(f'ground plane: {ninl}/{nq} inliers, thr={thr:.4g} units, R90={r90:.4g} units -> assumed s0={s0:.4g} m/unit')
    cam_h = (cam_c - origin) @ n * s0
    print('camera height above ground (assumed metres): median %.2f  [%.2f, %.2f]' % (np.median(cam_h), cam_h.min(), cam_h.max()))
    hi = [names.index(h) for h in held]
    Pg = (P[(tl >= 2)] - origin) @ Rg.T * s0                               # metric ground frame, all decent points
    Gin = G * s0
    # ---- trench placement search (metres, ground frame): visible from many held-out cams, clear of objects, ground-supported
    Lx, Wz, D = TRENCH['length'], TRENCH['width'], TRENCH['depth']
    best = None
    for cx in np.arange(-0.5 * SITE_RADIUS_M, 0.5 * SITE_RADIUS_M, 1.0):
        for cz in np.arange(-0.5 * SITE_RADIUS_M, 0.5 * SITE_RADIUS_M, 1.0):
            near = np.hypot(Gin[:, 0] - cx, Gin[:, 2] - cz) < 1.2
            if near.sum() < 8: continue
            nearall = np.hypot(Pg[:, 0] - cx, Pg[:, 2] - cz) < 4
            for az in np.radians([0, 45, 90, 135]):
                ca, sa = math.cos(az), math.sin(az)
                loc = Pg[nearall] - [cx, 0, cz]
                lx = loc[:, 0] * ca + loc[:, 2] * sa; lz = -loc[:, 0] * sa + loc[:, 2] * ca
                col = (np.abs(lx) < Lx / 2 + 0.7) & (np.abs(lz) < Wz / 2 + 0.7)
                pn = Pg[nearall]
                if ((pn[col, 1] > 0.15) & (pn[col, 1] < 2.5)).any(): continue     # something standing over it
                cs = [(x, y, zz) for x in (-Lx / 2, Lx / 2) for zz in (-Wz / 2, Wz / 2) for y in (-D, 0.0)]
                Xg = np.array([[cx + x * ca - zz * sa, y, cz + x * sa + zz * ca] for x, y, zz in cs])
                Xw = (Xg / s0) @ Rg + origin
                nvis = 0
                for i in hi:
                    uv, zc = project(W2C[i], K[i], Xw)
                    if (zc > 0.2).all() and (uv[:, 0] > 0).all() and (uv[:, 0] < wh[i][0]).all() and (uv[:, 1] > 0).all() and (uv[:, 1] < wh[i][1]).all():
                        dist = np.linalg.norm(cam_c[i] - Xw.mean(0)) * s0
                        if 2.5 < dist < 25: nvis += 1
                score = (nvis, -math.hypot(cx, cz))
                if best is None or score > best[0]: best = (score, cx, cz, az)
    (nvis, _), cx, cz, az = best
    print(f'trench centre ({cx:.2f},{cz:.2f}) m az={math.degrees(az):.0f} deg: fully in frame in {nvis}/{len(hi)} held-out views')
    res = dict(origin=origin, Rg=Rg, s0=s0, r90=r90, n=n, cx=cx, cz=cz, az=az, ninl=ninl, nq=nq, thr=thr, nvis_search=nvis,
               cam_h_med=float(np.median(cam_h)))
    # ---- sensitivity variants: metric scale x0.5, x1, x2 (trench stays 1x3x0.6 "metres", i.e. its fraction of scene changes)
    N, cell, half = 160, 0.1, 8.0
    tmp = a.tmp; os.makedirs(tmp, exist_ok=True)
    dH, Lg, info = run_dig(a.app_sim, tmp, 'trench', N, cell, half)
    print('real Heightfield dig:', json.dumps(info))
    res.update(dH=dH, Lgrid=Lg, grid_N=N, grid_cell=cell, grid_half=half, dig_info=json.dumps(info),
               scale_variants=np.array([0.5, 1.0, 2.0]) * s0)
    # residual of the sparse ground points about the flat plane near the trench (how wrong is "H0 = flat")
    loc = Pg - [cx, 0, cz]; near = np.hypot(loc[:, 0], loc[:, 2]) < 3
    gr = Pg[near & (np.abs(Pg[:, 1]) < 0.5)]
    res['ground_resid_rms_m'] = float(np.sqrt((gr[:, 1] ** 2).mean())) if len(gr) else float('nan')
    res['ground_pts_near'] = int(len(gr))
    print('flat-plane residual near trench: rms %.3f m over %d pts' % (res['ground_resid_rms_m'], res['ground_pts_near']))
    np.savez(a.out, **res)


# ----------------------------------------------------------------------------------------------- render (GPU)
def load_ply(path):
    with open(path, 'rb') as f:
        hdr = b''
        while b'end_header' not in hdr: hdr += f.readline()
        props = [l.split()[-1].decode() for l in hdr.split(b'\n') if l.startswith(b'property')]
        n = int([l for l in hdr.split(b'\n') if l.startswith(b'element vertex')][0].split()[-1])
        data = np.fromfile(f, np.float32, n * len(props)).reshape(n, len(props))
    c = {p: i for i, p in enumerate(props)}
    g = lambda *ks: np.stack([data[:, c[k]] for k in ks], 1)
    nrest = len([p for p in props if p.startswith('f_rest_')]) // 3
    rest = g(*[f'f_rest_{i}' for i in range(nrest * 3)]).reshape(n, 3, nrest).transpose(0, 2, 1)  # (n, nrest, 3)
    sh = np.concatenate([g('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None, :], rest], 1)
    return dict(xyz=g('x', 'y', 'z'), sh=sh, op=data[:, c['opacity']], scale=g('scale_0', 'scale_1', 'scale_2'), quat=g('rot_0', 'rot_1', 'rot_2', 'rot_3'))


def deform(xyz, opa, pz, s):
    """Port of deform.ts L78-L101 (see module docstring). xyz in COLMAP units; returns new xyz, new opacity, bind, moved mask."""
    Rg, origin = pz['Rg'], pz['origin']
    cx, cz, az = float(pz['cx']), float(pz['cz']), float(pz['az'])
    N, cell, half = int(pz['grid_N']), float(pz['grid_cell']), float(pz['grid_half'])
    dHg, Lg = pz['dH'], pz['Lgrid']
    G = (xyz - origin) @ Rg.T * s                       # metric ground frame (x, up, z)
    ca, sa = math.cos(az), math.sin(az)
    lx = (G[:, 0] - cx) * ca + (G[:, 2] - cz) * sa      # trench-local x (along trench), z (across)
    lz = -(G[:, 0] - cx) * sa + (G[:, 2] - cz) * ca

    def bil(tex):                                        # ldBilinear + ldFetch clamp (deform.ts L31-41, L26-29)
        fx = (lx + half) / cell - 0.5; fz = (lz + half) / cell - 0.5
        i0 = np.floor(fx).astype(int); j0 = np.floor(fz).astype(int); wx = fx - i0; wz = fz - j0
        f = lambda j, i: tex[np.clip(j, 0, N - 1), np.clip(i, 0, N - 1)]
        return (f(j0, i0) * (1 - wx) + f(j0, i0 + 1) * wx) * (1 - wz) + (f(j0 + 1, i0) * (1 - wx) + f(j0 + 1, i0 + 1) * wx) * wz
    dh, dl = bil(dHg), bil(Lg)
    rel = G[:, 1] - 0.0                                  # H0 = flat plane (assumption)
    bind = 1 - smoothstep(BIND_LOW, BIND_HIGH, rel)      # onObject = 0 (assumption)
    shift_m = bind * dh                                  # center.y += bind*dh.r   (L87)
    n_up = Rg[1]                                         # unit up vector in COLMAP frame
    new_xyz = xyz + (shift_m / s)[:, None] * n_up[None, :]
    changed = np.maximum(np.maximum(-dh, dh), dl)        # L100
    new_opa = opa * (1 - bind * smoothstep(0.05, 0.16, changed))   # L101
    moved = (np.abs(shift_m) > 1e-9) | (new_opa != opa)
    return new_xyz, new_opa, moved, dict(lx=lx, lz=lz, gy=G[:, 1], bind=bind, dh=dh)


def cmd_render(a):
    import torch, cv2, lpips
    from gsplat import rasterization
    from skimage.metrics import structural_similarity as ssim_fn
    dev = 'cuda'
    z = np.load(a.inp, allow_pickle=True); pz = dict(np.load(a.prep, allow_pickle=True))
    W2C, K, wh, names, held = z['W2C'], z['K'], z['wh'], list(z['names']), list(z['held'])
    ply = load_ply(a.ply)
    xyz = ply['xyz'].astype(np.float64); opa0 = 1 / (1 + np.exp(-ply['op'].astype(np.float64)))
    T = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt, device=dev)
    quats = T(ply['quat']); scales = T(np.exp(ply['scale'])); shs = T(ply['sh'])
    lp = lpips.LPIPS(net='alex', spatial=True, verbose=False).to(dev)

    def render(means, opa, i):
        W, H = int(wh[i][0]), int(wh[i][1])
        out, _, _ = rasterization(T(means), quats, scales, T(opa), shs, T(W2C[i])[None], T(K[i])[None], W, H, sh_degree=3, packed=False, render_mode='RGB')
        return out[0].clamp(0, 1)

    def hull(Xw, i, dil):
        W, H = int(wh[i][0]), int(wh[i][1])
        Xc = Xw @ W2C[i][:3, :3].T + W2C[i][:3, 3]
        if (Xc[:, 2] <= 0.05).any(): return None   # volume crosses the camera plane: skip view
        uv = (Xc @ K[i].T); uv = uv[:, :2] / uv[:, 2:3]
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
        if dil > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)))
        return m.astype(bool)

    def region_metrics(a_img, b_img, gt, reg, erode_ssim=3):
        """a_img/b_img/gt float HxWx3 [0,1] numpy, reg bool mask -> dict(psnr, ssim, lpips) over the region"""
        if reg.sum() < 500: return None
        mse = float(((a_img - gt) ** 2)[reg].mean()); psnr = 10 * math.log10(1 / max(mse, 1e-12))
        _, smap = ssim_fn(gt, a_img, channel_axis=2, data_range=1, full=True)
        er = cv2.erode(reg.astype(np.uint8), np.ones((2 * erode_ssim + 1,) * 2, np.uint8)).astype(bool) if erode_ssim else reg
        ss = float(smap.mean(2)[er].mean()) if er.sum() else float('nan')
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].to(dev) * 2 - 1
        with torch.no_grad(): lm = lp(t(a_img.astype(np.float32)), t(gt.astype(np.float32)))[0, 0].cpu().numpy()
        if lm.shape != reg.shape: lm = cv2.resize(lm, reg.shape[::-1])
        return dict(psnr=psnr, ssim=ss, lpips=float(lm[reg].mean()))

    out = dict(meta=dict(ply=os.path.basename(a.ply), n_splats=int(len(xyz)), held_out=len(held), s0_m_per_unit=float(pz['s0']),
                         ground_inliers=int(pz['ninl']), ground_resid_rms_m=float(pz['ground_resid_rms_m']), dig=json.loads(str(pz['dig_info'])),
                         trench=TRENCH, dilations_px=[DIL_PRIMARY, DIL_WIDE], camera_height_median_m=float(pz['cam_h_med'])), variants={})
    gt_dir = a.frames; sanity = []
    for vi, sv in enumerate(pz['scale_variants']):
        tag = ['scale_x0.5', 'scale_x1.0', 'scale_x2.0'][vi]
        # for variant scale the trench is still 1x3x0.6 *assumed metres*; the variant changes metres-per-unit
        pzv = dict(pz); pzv['cx'] = float(pz['cx']) * float(sv) / float(pz['s0']); pzv['cz'] = float(pz['cz']) * float(sv) / float(pz['s0'])
        new_xyz, new_opa, moved, dbg = deform(xyz, opa0, pzv, float(sv))
        # trench volume box (metric ground frame, y from -depth to bind-high), margin = one grid cell for the bilinear halo
        Lx, Wz, D = TRENCH['length'] / 2 + 0.1, TRENCH['width'] / 2 + 0.1, TRENCH['depth']
        ca, sa = math.cos(float(pz['az'])), math.sin(float(pz['az'])); cx, cz = float(pzv['cx']), float(pzv['cz'])
        corners = np.array([[cx + x * ca - q * sa, y, cz + x * sa + q * ca] for x in (-Lx, Lx) for q in (-Wz, Wz) for y in (-D, BIND_HIGH)])
        Xw = (corners / float(sv)) @ pz['Rg'] + pz['origin']
        # leak diagnostics on the splat set itself
        mv = moved
        inside = (np.abs(dbg['lx']) <= Lx) & (np.abs(dbg['lz']) <= Wz)
        diag = dict(moved=int(mv.sum()), moved_outside_trench_footprint=int((mv & ~inside).sum()),
                    moved_above_bind_high=int((mv & (dbg['gy'] > BIND_HIGH)).sum()), max_shift_m=float(np.abs(dbg['bind'] * dbg['dh']).max()),
                    opacity_changed=int((new_opa != opa0).sum()))
        views = []
        for i, nm in enumerate(held):
            idx = names.index(nm)
            ms = {d: hull(Xw, idx, d) for d in (DIL_PRIMARY, DIL_WIDE)}
            if ms[DIL_PRIMARY] is None: continue
            raw = hull(Xw, idx, 0)
            H, W = ms[DIL_PRIMARY].shape
            if raw.sum() < 0.0005 * H * W or ms[DIL_PRIMARY].sum() > 0.6 * H * W: continue
            # skip views where the volume is clipped by the frame (mask must be largely in-frame)
            gt = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, nm)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            o = render(xyz, opa0, idx).cpu().numpy(); d_ = render(new_xyz, new_opa, idx).cpu().numpy()
            row = dict(frame=nm, mask_frac=float(raw.mean()), mask_dil24_frac=float(ms[DIL_PRIMARY].mean()))
            if vi == 1 and a.sanity_dir:
                ref = cv2.cvtColor(cv2.imread(os.path.join(a.sanity_dir, nm.replace('.jpg', '.png'))), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                row['full_vs_brush_eval_render_psnr'] = float(10 * math.log10(1 / max(((o - ref) ** 2).mean(), 1e-12)))
                row['orig_full_psnr'] = float(10 * math.log10(1 / ((o - gt) ** 2).mean()))
            for dil, m in ms.items():
                unch = ~m
                r_o = region_metrics(o, o, gt, unch); r_d = region_metrics(d_, d_, gt, unch)
                row[f'unchanged_dil{dil}'] = dict(orig=r_o, deformed=r_d,
                                                  max_abs_pixel_diff=float(np.abs(o - d_)[unch].max()),
                                                  frac_pixels_diff_gt_1_255=float((np.abs(o - d_).max(2)[unch] > 1 / 255).mean()))
            chg = ms[DIL_PRIMARY]
            row['changed_dil24'] = dict(orig=region_metrics(o, o, gt, chg), deformed=region_metrics(d_, d_, gt, chg))
            views.append(row)
            print(tag, nm, 'mask %.3f' % row['mask_frac'], 'unchanged maxdiff', row['unchanged_dil24']['max_abs_pixel_diff'], flush=True)
        def agg(path):
            vals = []
            for v in views:
                x = v
                for p in path: x = x.get(p) if x else None
                if x: vals.append(x)
            return {k: float(np.mean([q[k] for q in vals])) for k in ('psnr', 'ssim', 'lpips')} if vals else None
        summ = dict(n_views=len(views), diag=diag)
        for dil in (DIL_PRIMARY, DIL_WIDE):
            summ[f'unchanged_dil{dil}'] = dict(orig=agg([f'unchanged_dil{dil}', 'orig']), deformed=agg([f'unchanged_dil{dil}', 'deformed']),
                                               max_abs_pixel_diff_over_views=max((v[f'unchanged_dil{dil}']['max_abs_pixel_diff'] for v in views), default=None),
                                               mean_frac_pixels_diff_gt_1_255=float(np.mean([v[f'unchanged_dil{dil}']['frac_pixels_diff_gt_1_255'] for v in views])) if views else None)
        summ['changed_dil24'] = dict(orig=agg(['changed_dil24', 'orig']), deformed=agg(['changed_dil24', 'deformed']),
                                     caveat='No post-dig ground truth exists: these measure departure from the PRE-dig photo, not correctness.')
        if vi == 1:
            sv_ = [v for v in views if 'orig_full_psnr' in v]
            if sv_: summ['sanity_full_frame'] = dict(mean_orig_full_psnr=float(np.mean([v['orig_full_psnr'] for v in sv_])),
                                                     mean_psnr_vs_brush_eval_render=float(np.mean([v['full_vs_brush_eval_render_psnr'] for v in sv_])), n=len(sv_))
        summ['metres_per_unit'] = float(sv)
        out['variants'][tag] = dict(summary=summ, per_view=views)
        del new_xyz, new_opa
    # baseline over ALL held-out views (no mask) for reference to the repo's 23.53 / 0.723 / 0.330
    base = []
    for nm in held:
        idx = names.index(nm)
        gt = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, nm)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        o = render(xyz, opa0, idx).cpu().numpy()
        base.append(region_metrics(o, o, gt, np.ones(o.shape[:2], bool), erode_ssim=0))
    out['baseline_full_frame_all_heldout'] = {k: float(np.mean([b[k] for b in base])) for k in ('psnr', 'ssim', 'lpips')}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    e = sp.add_parser('export'); e.add_argument('--work', required=True); e.add_argument('--out', required=True)
    p = sp.add_parser('prep'); p.add_argument('--inp', required=True); p.add_argument('--out', required=True)
    p.add_argument('--app-sim', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app/src/sim')); p.add_argument('--tmp', required=True)
    r = sp.add_parser('render'); r.add_argument('--inp', required=True); r.add_argument('--prep', required=True); r.add_argument('--ply', required=True)
    r.add_argument('--frames', required=True); r.add_argument('--sanity-dir'); r.add_argument('--out', required=True)
    a = ap.parse_args()
    {'export': cmd_export, 'prep': cmd_prep, 'render': cmd_render}[a.cmd](a)
