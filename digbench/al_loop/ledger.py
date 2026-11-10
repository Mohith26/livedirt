"""Append one model version to ~/v4/loop/ledger.jsonl: held-out per-bite RMSE (3 seeds), both gates vs the teacher, and
track BEST by gate score = mean(gate1 vs teacher seeds 1,2) + gate2.  python -m digbench.al_loop.ledger <tag> <n>"""
import json, sys
from pathlib import Path

from digbench.compare_grids import compare, load

L = Path("~/v4/loop").expanduser(); G = Path("~/v4/grids").expanduser(); E = Path("~/v4/eval").expanduser()


def main():
    tag, n = sys.argv[1], int(sys.argv[2]); row = {"tag": tag, "n_samples": n, "per_bite": []}
    for sd in range(3):
        for line in (L / f"train-{tag}-s{sd}.log").read_text().splitlines():
            if line.startswith("{\"n_test\""):
                row["per_bite"].append(round(json.loads(line)["rmse_model"] * 100, 2)); row["n_train"] = json.loads(line)["n_train"]
    try:
        g1 = [compare(load(G / f"v4gate-mpm-{s}.json"), load(E / tag / "gate1.json"))["rmse_design_m"] for s in ("s1", "s2")]
        g2 = compare(load(G / "v4gate2-mpm-s1.json"), load(E / tag / "gate2.json"))["rmse_design_m"]
        row.update(gate1_cm=[round(100 * x, 1) for x in g1], gate2_cm=round(100 * g2, 1), score_cm=round(100 * (sum(g1) / 2 + g2), 1))
    except Exception as e:
        row["gate_error"] = str(e)[:200]
    with open(L / "ledger.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    best = json.loads((L / "BEST.json").read_text()) if (L / "BEST.json").exists() else None
    if "score_cm" in row and (best is None or row["score_cm"] < best["score_cm"]):
        (L / "BEST.json").write_text(json.dumps(row))
    print(json.dumps(row))


if __name__ == "__main__":
    main()
