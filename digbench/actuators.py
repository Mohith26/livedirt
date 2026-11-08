"""DX225LC-7 per-actuator limits (boom, arm, bucket cylinders) and machine stability (front lift-off, forward tip,
track sliding) for a soil wrench on the bucket. Mirrored in app/src/machine/actuators.ts.

Sources (Develon DX225LC-7 brochure D4600882 04-2023 p.17/20, and the Develon North America spec sheet):
  implement relief 34.3 MPa (350 kgf/cm2), power boost 36.3 MPa;
  cylinders (bore x rod, mm): boom 2 x 125 x 85, arm 140 x 100 (US sheet: 135 x 100), bucket 120 x 80;
  ISO digging force at normal pressure: bucket 15.6 t (153.0 kN), arm 10.9 t (106.9 kN); boom 5.7 m, arm 2.9 m;
  operating weight 23.3-23.6 t (shoe width); tumbler distance 3.65 m.
Pin coordinates are not published. Each cylinder is modelled as a torque source at its joint with an effective
moment arm r_k chosen so the model reproduces the ISO digging force at the ISO radius (bucket: wrist pin to tip
1.45 m; arm: arm pin to tip 2.9 + 1.45 = 4.35 m). The extend-side torque capacity therefore equals the ISO-implied
torque regardless of the bore discrepancy; retract-side capacity scales by the annulus/bore area ratio. The boom has
no ISO figure: its moment arm R_BOOM is an ASSUMPTION (typical 20 t class), reported with a +/-30% sensitivity.
Constant moment arms are an upper bound (real arms shrink away from the optimum angle).

Sign conventions (boom plane, r away from the machine, y up, CCW positive): M_k is the soil moment about joint k.
Arm-in and bucket-curl are cylinder EXTEND and resist M > 0; holding the boom down against an upward soil push uses
the boom RETRACT side (M > 0).
"""
import math

P_RELIEF = 34.3e6
P_BOOST = 36.3e6
G = 9.81

# effective torque capacities (N m) at P_RELIEF
T_BUCKET_EXT = 153.0e3 * 1.45
T_ARM_EXT = 106.9e3 * 4.35
_ann = lambda bore, rod: 1.0 - (rod / bore) ** 2
T_BUCKET_RET = T_BUCKET_EXT * _ann(120, 80)
T_ARM_RET = T_ARM_EXT * _ann(140, 100)
A_BOOM_BORE = 2 * math.pi / 4 * 0.125 ** 2
R_BOOM = 0.65  # ASSUMPTION: boom cylinder effective moment arm about the boom foot (m)
T_BOOM_EXT = P_RELIEF * A_BOOM_BORE * R_BOOM
T_BOOM_RET = T_BOOM_EXT * _ann(125, 85)

MASS = 23_600.0  # kg (US configuration, 800 mm shoes)
HALF_TRACK = 3.65 / 2  # tumbler distance / 2: tipping edges over the front/rear (m)
MU_TRACK = 0.6  # ASSUMPTION: track-ground traction coefficient (steel grousers on soil)
BOOM, ARM = 5.7, 2.9
BOOM_FOOT = (0.18, 2.05)  # app/src/machine/kinematics.ts (typical, not manufacturer data)


def _frac(M, t_ext, t_ret, extend_positive=True):
    if (M > 0) == extend_positive:
        return abs(M) / t_ext
    return abs(M) / t_ret


def pins_from_wrist(wrist):
    """Elbow-up two-link IK (boom foot -> elbow -> wrist) in the boom plane. Returns foot, elbow (None if out of reach)."""
    fx, fy = BOOM_FOOT
    dx, dy = wrist[0] - fx, wrist[1] - fy
    d = math.hypot(dx, dy)
    if d > BOOM + ARM - 1e-6 or d < abs(BOOM - ARM) + 1e-6:
        return (fx, fy), None
    a = (BOOM ** 2 - ARM ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(max(0.0, BOOM ** 2 - a ** 2))
    mx, my = fx + a * dx / d, fy + a * dy / d
    # elbow on the upper side of the foot->wrist line
    e1 = (mx - h * dy / d, my + h * dx / d)
    e2 = (mx + h * dy / d, my - h * dx / d)
    return (fx, fy), (e1 if e1[1] >= e2[1] else e2)


def demand(F_r, F_y, T_pivot, pivot, wrist=None, elbow=None, foot=BOOM_FOOT, bucket_xy=None, pressure=P_RELIEF):
    """Soil wrench on the bucket: force (F_r, F_y) N applied with torque T_pivot (N m) about `pivot` (r, y), all in the
    machine's boom plane (r from the swing axis, y above the track ground). Returns fractions of rated capacity per
    cylinder (>1 = exceeds relief), stability margins, and the governing limit."""
    if wrist is None:
        wrist = pivot
    if elbow is None:
        _, elbow = pins_from_wrist(wrist)
        if elbow is None:
            elbow = (foot[0] + BOOM, foot[1])
    s = pressure / P_RELIEF

    def moment_about(q):
        return T_pivot + (pivot[0] - q[0]) * F_y - (pivot[1] - q[1]) * F_r

    Mw, Me, Mb = moment_about(wrist), moment_about(elbow), moment_about(foot)
    out = {
        "M_bucket": Mw, "M_arm": Me, "M_boom": Mb,
        "bucket": _frac(Mw, s * T_BUCKET_EXT, s * T_BUCKET_RET),
        "arm": _frac(Me, s * T_ARM_EXT, s * T_ARM_RET),
        # boom: M > 0 (soil pushes the bucket up / outward) is held by the rod (retract) side
        "boom": _frac(Mb, s * T_BOOM_EXT, s * T_BOOM_RET, extend_positive=False),
    }
    # machine stability (machine CG on the swing axis, ASSUMPTION): moments of the soil force about the track edges
    W = MASS * G
    px, py = (bucket_xy if bucket_xy is not None else pivot)
    # front lifts (rotation about the rear edge) when this exceeds W * HALF_TRACK
    m_rear = (px + HALF_TRACK) * F_y - py * F_r + T_pivot
    # rear lifts (rotation about the front edge, forward tip) when -(...) exceeds W * HALF_TRACK
    m_front = (px - HALF_TRACK) * F_y - py * F_r + T_pivot
    out["front_liftoff"] = m_rear / (W * HALF_TRACK)
    out["forward_tip"] = -m_front / (W * HALF_TRACK)
    out["sliding"] = abs(F_r) / (MU_TRACK * W)
    keys = ("bucket", "arm", "boom", "front_liftoff", "forward_tip", "sliding")
    out["governing"] = max(keys, key=lambda k: out[k])
    out["max_frac"] = out[out["governing"]]
    return out
