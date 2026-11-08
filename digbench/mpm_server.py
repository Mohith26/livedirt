"""MPM-in-the-loop bite server: the app's autonomy (app/bench/autonomy.ts --soil-backend mpm) sends each bite as a
DigTrace (terrain around the bite + the bucket's pose list); this runs the bite in 3D MPM (gen_replay.Episode, open bed,
per-actuator relief, stall stop) and returns the change of the soil surface over the bed and the carried volume.

The surface change is reported as a DELTA against MPM's own settled surface before the bite, so MPM's elastic
settlement and particle sampling of the initial terrain never leak into the app's heightfield; only what the bite did
does. Cells whose |delta| is below `--min-delta` are returned as 0.

POST /bite  {"trace": DigTrace, "soil": {"model": "mc", "phi": 36, "c": 20, "mu": 0.73}, "margin": 1.0,
             "col_depth": 0.8, "seed": 0}
  -> {"delta": [nx*ny], "nx", "ny", "cell", "margin", "carried", "completion", "peak_demand", "drag_mean_N", "wall", ...}
GET /health -> "ok"

  python -m digbench.mpm_server --port 8765 --arch cuda --dx 0.04 --volume-correction
"""
import argparse
import json
import math
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from . import gen_replay as G
from .attitude import lip_offset
from .soil_presets import mc_params

ARGS = None
LOG = []


def run_request(req):
    tr = req["trace"]
    soil = req.get("soil", {})
    G.VOLCORR[0] = bool(req.get("vc", ARGS.volume_correction))
    if "preset" in soil:
        soil = {**mc_params(soil["preset"]), **{k: v for k, v in soil.items() if k != "preset"}}
    G.YIELD.update(model=soil.get("model", "mc"), phi=soil.get("phi"), c=float(soil.get("c", 0.0)), rho=float(soil.get("rho", 1600.0)))
    mu = float(soil.get("mu", math.tan(math.radians(soil.get("phi", 30.0)))))
    M = float(req.get("margin", ARGS.margin))
    terr = np.array(tr["terrain"], dtype=np.float64).reshape(tr["nx"], tr["ny"])
    cut = int(round((tr["margin"] - M) / G.CELL))
    if cut > 0:
        terr = terr[cut:-cut, cut:-cut]
    elif cut < 0:
        raise ValueError(f"requested margin {M} exceeds the trace margin {tr['margin']}")
    poses = tr["poses"]
    lip_min = min(p[2] + lip_offset(p[3])[1] for p in poses)
    ep = G.Episode(terr, M, mu, float(req.get("dx", ARGS.dx)), ARGS.arch, 1, r0=tr["r0"], ground_win=tr["groundY"] + tr["shift"],
                   seed=int(req.get("seed", 0)), min_floor_win=lip_min, poses=poses, col_depth=req.get("col_depth", ARGS.col_depth))
    r = ep.run_bite_lean(poses, limits=bool(req.get("limits", True)))
    d = r.pop("H_after") - r.pop("H_before")
    d[np.abs(d) < ARGS.min_delta] = 0.0
    r.update(delta=np.round(d, 4).ravel().tolist(), nx=int(d.shape[0]), ny=int(d.shape[1]), cell=G.CELL, margin=M,
             removed_m3=float(-d[d < 0].sum() * G.CELL ** 2), added_m3=float(d[d > 0].sum() * G.CELL ** 2))
    return r


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, {"ok": True, "served": len(LOG)})
        if self.path.startswith("/shutdown"):
            self._send(200, {"bye": True})
            raise SystemExit(0)
        self._send(404, {})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            t0 = time.perf_counter()
            r = run_request(req)
            LOG.append({k: r[k] for k in ("carried", "completion", "peak_demand", "wall", "particles")})
            print(f"bite {len(LOG)}: carried={r['carried']:.3f} completion={r['completion']:.2f} demand={r['peak_demand']:.2f} "
                  f"removed={r['removed_m3']:.3f} added={r['added_m3']:.3f} particles={r['particles']} wall={time.perf_counter() - t0:.0f}s", flush=True)
            self._send(200, r)
        except Exception as e:  # report to the client instead of killing the server
            traceback.print_exc()
            self._send(500, {"error": repr(e)})


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--arch", default="cuda")
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--col-depth", type=float, default=None)
    ap.add_argument("--min-delta", type=float, default=0.01)
    ap.add_argument("--volume-correction", action="store_true")
    ARGS = ap.parse_args()
    srv = HTTPServer(("127.0.0.1", ARGS.port), H)
    print(f"mpm_server on 127.0.0.1:{ARGS.port}", flush=True)
    try:
        srv.serve_forever()
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
