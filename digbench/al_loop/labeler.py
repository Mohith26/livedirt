"""Forever: claim the highest-disagreement bite in ~/v4/loop/queue and have the MPM server on <port> label it.
  python -m digbench.al_loop.labeler <mpm-port>"""
import json, os, sys, time, urllib.request
from pathlib import Path

L = Path("~/v4/loop").expanduser(); Q, C, D, F = L / "queue", L / "claimed", L / "done", L / "failed"
for p in (Q, C, D, F):
    p.mkdir(parents=True, exist_ok=True)


def main():
    port = int(sys.argv[1])
    while not (L / "STOP").exists():
        items = sorted((p for p in Q.glob("*.json")), reverse=True)
        got = None
        for p in items:
            try:
                dst = C / f"{port}_{p.name}"; os.rename(p, dst); got = dst; break
            except FileNotFoundError:
                continue  # another labeller took it
        if got is None:
            time.sleep(30); continue
        it = json.loads(got.read_text()); t0 = time.time()
        body = json.dumps({k: it[k] for k in ("trace", "soil", "margin", "seed", "run")}).encode()
        rq = urllib.request.Request(f"http://127.0.0.1:{port}/bite", data=body, headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(rq, timeout=7200).read())
            os.replace(got, D / got.name)
            with open(L / "labels.log", "a") as f:
                f.write(json.dumps({"t": time.time(), "port": port, "run": it["run"], "i": it["seed"], "score": it["score"],
                                    "wall": round(time.time() - t0), "completion": r.get("completion"), "carried": r.get("carried")}) + "\n")
        except Exception as e:
            os.replace(got, F / got.name); print("label failed", got.name, e, flush=True); time.sleep(10)


if __name__ == "__main__":
    main()
