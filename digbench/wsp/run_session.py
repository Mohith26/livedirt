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
    ap.add_argument("--thinking", default="high"); ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--zero-first", action="store_true", help="write mode: start the target tensors from zero")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    net = load_model(a.base); P.MODEL["net"] = net
    task = Path(a.task).read_text(); ctx = CONTEXT.read_text() + "\n" + describe(net)
    pricing = None
    if a.llm.startswith("scripted:"):
        llm = wsp.ScriptedLLM.from_transcript(a.llm.split(":", 1)[1])
    elif a.llm.startswith("bedrock:"):
        from digbench.wsp import bedrock
        llm, pricing = bedrock.get(a.llm)
    else:
        llm = a.llm
        if a.llm.startswith("anthropic:"):
            from weightspace.llm.anthropic import AnthropicChat
            llm = AnthropicChat(a.llm.split(":", 1)[1], max_tokens=64000)  # 16k default let high thinking eat whole turns  # direct Anthropic API: same conservative rates as Bedrock, so --max-cost binds
            from digbench.wsp import bedrock
            m = a.llm.split(":", 1)[1]
            pricing = bedrock.PRICING["fable" if "fable" in m else "opus" if "opus" in m else "sonnet"]
    fn = {"read": wsp.read, "fix": wsp.fix, "write": wsp.write, "improve": wsp.improve}[a.mode]
    kw = dict(llm=llm, context=ctx, probes=P.ALL, forward_budget=a.budget,
              thinking=a.thinking, cost_budget=a.max_cost if pricing else None, pricing=pricing, max_steps=a.max_steps)
    t0 = time.time()
    res = fn(net, task, **kw)
    res.save(str(out))
    torch.save(net.state_dict(), out / "patched.pt")
    (out / "meta.json").write_text(json.dumps({"task": a.task, "mode": a.mode, "llm": a.llm, "budget": a.budget, "wall_s": time.time() - t0,
                                               "stop_reason": str(getattr(res, "stop_reason", "")), "cost": getattr(res, "cost", None)}, indent=1))
    ans = (getattr(res, "answer", "") or "").strip()
    if not ans:  # a session that "answered" with nothing is a harness failure (e.g. thinking used the whole turn)
        (out / "BROKEN").write_text("empty answer; stop_reason=%s\n" % getattr(res, "stop_reason", ""))
        print("WARNING: empty answer -> marked BROKEN")
    print(res)


def describe(net):
    """Exact facts about THIS copy, appended to the context so the editor never works from stale numbers."""
    from digbench.wsp.wide import EXTRA
    n = sum(p.numel() for p in net.parameters()); cin = int(net.e1[0].in_channels)
    lines = [f"## This copy: {n:,} weights, {cin} input channels" + (" (channel 7 = beyond the predicted stop)" if cin == 8 else "")]
    for b in ("e1", "e2", "e3", "e4", "d3", "d2", "d1"):
        co = net.get_submodule(b)[0].out_channels; g = net.get_submodule(b)[1].num_groups; ci = net.get_submodule(b)[0].in_channels
        lines.append(f"- {b}: in {ci}, out {co}, GroupNorm {g} groups of {co // g}")
    wide = net.e1[0].out_channels != 32
    if wide:
        lines.append("- SPARE: each block has one extra GroupNorm group appended after its original channels (e1/d1 32-35, "
                     "e2/d2 64-71, e3/d3 128-143, e4 256-287). All weights into and out of spare channels are exactly zero, so "
                     "they output 0 and the predictions equal the original network's. Concatenated decoder inputs keep each "
                     "source's spare channels right after its originals (d1.0 input: d2 0-63, spare 64-71, e1 72-103, spare "
                     "104-107; d2.0: d3 0-127, spare 128-143, e2 144-207, spare 208-215; d3.0: e4 0-255, spare 256-287, "
                     "e3 288-415, spare 416-431); out.weight inputs 32-35 are spare; head.0 input = [e4 mean 0-287 | e4 max 288-575].")
    if cin == 8:
        lines.append("- Input channel 7 enters only e1.0; e1.0.weight[:, 7] is all zero, so the network currently ignores it.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
