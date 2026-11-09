"""Run one weight-space programming session on the scoop model with the `weightspace` harness (Takehi et al. 2026
protocol: the LLM reads/edits tensors in a Python sandbox, may only run the network through budgeted probes on the
PRACTICE tier, never trains or fits weights). Saves the transcript, the sparse patch and the patched state dict; then
call the judge on the validation tier.

  python -m digbench.wsp.run_session --task digbench/wsp/tasks/E3_fix_stalls.md --mode fix \
      --base ~/v4/wsp/w0-s0.pt --llm anthropic:claude-opus-5-5 --budget 10 --max-cost 25 --out ~/v4/wsp/runs/E3-a
  (--llm scripted:<transcript.jsonl> replays a logged run for free)
"""
import argparse, json, time
from pathlib import Path

import torch
import weightspace as wsp

from digbench.wsp import probes as P
from digbench.wsp.common import load_model

CONTEXT = (Path(__file__).parent / "tasks" / "CONTEXT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--mode", choices=["read", "fix", "write", "improve"], required=True)
    ap.add_argument("--base", required=True); ap.add_argument("--llm", required=True)
    ap.add_argument("--budget", type=int, default=10); ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--thinking", default="high"); ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--zero-first", action="store_true", help="write mode: start the target tensors from zero")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    net = load_model(a.base); P.MODEL["net"] = net
    task = Path(a.task).read_text(); ctx = CONTEXT.read_text()
    pricing = None
    if a.llm.startswith("scripted:"):
        llm = wsp.ScriptedLLM.from_transcript(a.llm.split(":", 1)[1])
    elif a.llm.startswith("bedrock:"):
        from digbench.wsp import bedrock
        llm, pricing = bedrock.get(a.llm)
    else:
        llm = a.llm
    fn = {"read": wsp.read, "fix": wsp.fix, "write": wsp.write, "improve": wsp.improve}[a.mode]
    kw = dict(llm=llm, context=ctx, probes=[P.error_report, P.worst_bites, P.error_map, P.feature_maps], forward_budget=a.budget,
              thinking=a.thinking, cost_budget=a.max_cost if pricing else None, pricing=pricing)
    t0 = time.time()
    res = fn(net, task, **kw)
    res.save(str(out))
    torch.save(net.state_dict(), out / "patched.pt")
    (out / "meta.json").write_text(json.dumps({"task": a.task, "mode": a.mode, "llm": a.llm, "budget": a.budget, "wall_s": time.time() - t0,
                                               "stop_reason": str(getattr(res, "stop_reason", "")), "cost": getattr(res, "cost", None)}, indent=1))
    print(res)


if __name__ == "__main__":
    main()
