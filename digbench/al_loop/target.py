"""The "within 10% of the simulator" target, measured against every teacher run of each gate.

Per gate, per teacher seed, three closed-loop metrics (final terrain of the learned run vs the teacher run):
  cut     |soil removed - teacher's|            <= max(10% of the teacher's,   1.1 x teacher-vs-teacher gap)
  surface RMS of the final surface over design  <= 1.1 x teacher-vs-teacher RMS (the teacher's own run-to-run noise)
  floor   |mean surface height difference|      <= max(10% of the 1.0 m design depth, 1.1 x teacher-vs-teacher gap)
The teacher-vs-teacher numbers come from the two teacher seeds of each gate. Pass = every metric on every pair.

  python -m digbench.al_loop.target <tag> [<tag> ...]
"""
import json, sys
from pathlib import Path

from digbench.compare_grids import compare, load

G = Path("~/v4/grids").expanduser(); E = Path("~/v4/eval").expanduser()
GATES = {"gate1": ["v4gate-mpm-s1.json", "v4gate-mpm-s2.json"], "gate2": ["v4gate2-mpm-s1.json", "v4gate2-mpm-s2.json"]}
DEPTH = 1.0


def floors():
    out = {}
    for g, refs in GATES.items():
        a, b = load(G / refs[0]), load(G / refs[1]); c = compare(a, b)
        out[g] = {"surface": 1.1 * c["rmse_design_m"], "cut_gap": 1.1 * abs(c["cut_volume_diff_m3"]), "floor_gap": 1.1 * abs(c["mean_diff_design_m"]),
                  "teacher_self": {k: round(v, 4) for k, v in c.items()}}
    return out


def evaluate(tag, F=None):
    F = F or floors(); res = {"tag": tag, "pass": True, "checks": []}
    for g, refs in GATES.items():
        f = E / tag / f"{g}.json"
        if not f.exists():
            res["pass"] = False; res["checks"].append({"gate": g, "missing": True}); continue
        o = load(f)
        for r in refs:
            c = compare(load(G / r), o); lim = F[g]
            cut_lim = max(0.10 * c["ref_cut_m3"], lim["cut_gap"]); floor_lim = max(0.10 * DEPTH, lim["floor_gap"])
            ck = {"gate": g, "ref": r.split("-")[-1][:2],
                  "cut_rel": round(abs(c["cut_volume_diff_m3"]) / c["ref_cut_m3"], 3), "cut_ok": abs(c["cut_volume_diff_m3"]) <= cut_lim,
                  "surface_cm": round(100 * c["rmse_design_m"], 1), "surface_lim_cm": round(100 * lim["surface"], 1), "surface_ok": c["rmse_design_m"] <= lim["surface"],
                  "floor_cm": round(100 * c["mean_diff_design_m"], 1), "floor_lim_cm": round(100 * floor_lim, 1), "floor_ok": abs(c["mean_diff_design_m"]) <= floor_lim}
            ck["ok"] = ck["cut_ok"] and ck["surface_ok"] and ck["floor_ok"]; res["pass"] &= ck["ok"]; res["checks"].append(ck)
    res["n_ok"] = sum(1 for c in res["checks"] if c.get("ok")); res["n_checks"] = len(res["checks"])
    res["n_metric_ok"] = sum(int(c.get(k, False)) for c in res["checks"] for k in ("cut_ok", "surface_ok", "floor_ok"))
    return res


if __name__ == "__main__":
    F = floors(); print(json.dumps({g: v["teacher_self"] for g, v in F.items()}))
    for t in sys.argv[1:]:
        r = evaluate(t, F)
        print(t, "PASS" if r["pass"] else "fail", f"{r['n_metric_ok']}/12 metric checks")
        for c in r["checks"]:
            print("  ", json.dumps(c))
