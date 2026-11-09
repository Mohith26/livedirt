"""Editing models through Amazon Bedrock (billed to the AWS account, so AWS credits apply), using the GPU box's
instance role. No API key is stored anywhere.

  --llm bedrock:opus     Claude Opus 5.5, Anthropic Messages API via bedrock-runtime (us geo profile), SigV4 from the role
  --llm bedrock:astra    GPT-6 Astra, OpenAI Responses API via bedrock-mantle us-east-1, short-term bearer token
                         minted from the role (aws-bedrock-token-generator), refreshed before it expires

Pricing (USD per 1M tokens) is used only to report spend and enforce --max-cost. Opus figures are set slightly ABOVE the
published Bedrock list price so the cap trips early rather than late; Astra uses the model card's In-Region rates.
"""
from __future__ import annotations

import os
import time

REGION = os.environ.get("WSP_BEDROCK_REGION", "us-east-1")

MODELS = {
    "opus": "us.anthropic.claude-opus-5-5",
    "astra": "openai.gpt-6-astra",
    "sonnet": "us.anthropic.claude-sonnet-5-5",  # cheap smoke tests only
    # open-weight models served by Bedrock itself (no Marketplace entitlement needed), via the Converse API
    "glm5": "zai.glm-5",
    "kimi": "moonshotai.kimi-k2.5",
    "deepseek": "deepseek.v3.2",
    "qwen-coder": "qwen.qwen3-coder-next",
}
CONVERSE = {"glm5", "kimi", "deepseek", "qwen-coder"}

PRICING = {
    "opus": dict(input=5.0, output=25.0, cached_input=0.5, cache_write=6.25),
    "astra": dict(input=11.0, output=55.0, cached_input=1.10, cache_write=13.75),
    "sonnet": dict(input=3.0, output=15.0, cached_input=0.3, cache_write=3.75),
    # open models: deliberately high placeholder rates (not checked against the price list) so --max-cost trips early
    "glm5": dict(input=2.0, output=8.0),
    "kimi": dict(input=2.0, output=8.0),
    "deepseek": dict(input=2.0, output=8.0),
    "qwen-coder": dict(input=2.0, output=8.0),
}


def _to_converse(messages):
    system, out = [], []
    for m in messages:
        r = m["role"]
        if r == "system":
            system.append({"text": m["content"]})
            continue
        if r == "user":
            role, blocks = "user", [{"text": m["content"] or " "}]
        elif r == "assistant":
            role = "assistant"
            blocks = list(m.get("converse_content") or [])
            if not blocks:
                blocks = ([{"text": m["content"]}] if m.get("content") else []) + [
                    {"toolUse": {"toolUseId": t["id"], "name": t["name"], "input": t["arguments"]}} for t in m.get("tool_calls") or []]
            if not blocks:
                blocks = [{"text": " "}]
        else:  # tool result
            role, blocks = "user", [{"toolResult": {"toolUseId": m["tool_call_id"], "content": [{"text": m["content"] or " "}]}}]
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    return system, out


from weightspace.llm.base import LLM as _LLM  # noqa: E402


class ConverseChat(_LLM):
    """weightspace LLM over the Bedrock Converse API (tool calling), for Bedrock-served open-weight models."""

    thinking = None
    speed = None
    session_key = None

    def __init__(self, key: str, max_tokens: int = 16000):
        import boto3
        from botocore.config import Config

        self.model, self.name, self.max_tokens = MODELS[key], f"bedrock:{key}", max_tokens
        self.client = boto3.client("bedrock-runtime", region_name=REGION,
                                   config=Config(read_timeout=900, retries={"max_attempts": 8, "mode": "adaptive"}))

    def complete(self, messages, tools=None):
        from weightspace.llm.base import LLMResponse, ToolCall
        from weightspace.usage import Usage

        system, msgs = _to_converse(messages)
        kw = dict(modelId=self.model, messages=msgs, inferenceConfig={"maxTokens": self.max_tokens})
        if system:
            kw["system"] = system
        if tools:
            kw["toolConfig"] = {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"][:4000] or t["name"],
                                                       "inputSchema": {"json": t["parameters"]}}} for t in tools]}
        r = self.client.converse(**kw)
        blocks = r["output"]["message"]["content"]
        text = [b["text"] for b in blocks if "text" in b]
        calls = [ToolCall(b["toolUse"]["toolUseId"], b["toolUse"]["name"], dict(b["toolUse"].get("input") or {}))
                 for b in blocks if "toolUse" in b]
        u = r.get("usage", {})
        tok = Usage(input_tokens=int(u.get("inputTokens", 0)), output_tokens=int(u.get("outputTokens", 0)),
                    cached_input_tokens=int(u.get("cacheReadInputTokens", 0) or 0), requests=1)
        keep = [b for b in blocks if "text" in b or "toolUse" in b]  # reasoning blocks are not replayed
        return LLMResponse(text="\n".join(text) or None, tool_calls=calls, usage=u, raw=r,
                           extra={"converse_content": keep}, tokens=tok, stop_reason=r.get("stopReason"))


def _anthropic(key: str, **kw):
    from anthropic import AnthropicBedrock
    from weightspace.llm.anthropic import AnthropicChat

    client = AnthropicBedrock(aws_region=REGION, max_retries=6, timeout=900)
    llm = AnthropicChat(MODELS[key], client=client, max_tokens=kw.pop("max_tokens", 32000), **kw)
    llm.name = f"bedrock:{key}"
    return llm


class _MantleToken:
    """Short-term Bedrock API key derived from the instance role; valid up to 12 h, we refresh every 6 h."""

    def __init__(self):
        self.tok, self.t = None, 0.0

    def get(self) -> str:
        if self.tok is None or time.time() - self.t > 6 * 3600:
            from aws_bedrock_token_generator import provide_token

            self.tok, self.t = provide_token(region=REGION), time.time()
        return self.tok


def _astra(**kw):
    from openai import OpenAI
    from weightspace.llm.openai import OpenResponses

    token = _MantleToken()
    base = f"https://bedrock-mantle.{REGION}.api.aws/openai/v1"
    llm = OpenResponses(MODELS["astra"], client=OpenAI(api_key=token.get(), base_url=base, max_retries=6, timeout=900), **kw)
    inner = llm.complete

    def complete(messages, tools=None):
        llm.client.api_key = token.get()  # refresh on long sessions
        return inner(messages, tools)

    llm.complete = complete
    llm.name = "bedrock:astra"
    return llm


def get(spec: str):
    """spec 'bedrock:<opus|astra|sonnet>' -> (llm, pricing dict)."""
    key = spec.split(":", 1)[1]
    if key not in MODELS:
        raise ValueError(f"unknown bedrock model {key!r}; choose from {sorted(MODELS)}")
    llm = _astra() if key == "astra" else ConverseChat(key) if key in CONVERSE else _anthropic(key)
    return llm, PRICING[key]


def smoke(spec: str = "bedrock:sonnet") -> dict:
    """One tiny call; returns text, token usage and the computed cost."""
    from weightspace.usage import Pricing

    llm, price = get(spec)
    tool = {"name": "add", "description": "Add two integers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}
    msgs = [{"role": "user", "content": "Use the add tool to add 2 and 3, then reply with only the result."}]
    r = llm.complete(msgs, [tool])
    if r.tool_calls:  # round-trip one tool result to prove multi-turn tool use works
        c = r.tool_calls[0]
        msgs += [r.to_message(), {"role": "tool", "tool_call_id": c.id, "name": c.name, "content": str(sum(c.arguments.values()))}]
        r2 = llm.complete(msgs, [tool])
        r = type(r)(text=f"tool_call={c.arguments} -> final={r2.text!r}", tokens=(r.tokens + r2.tokens) if r.tokens and r2.tokens else r.tokens, usage=r2.usage)
    cost = Pricing(**price).cost(r.tokens) if r.tokens else None
    return {"model": spec, "text": r.text, "usage": r.usage, "cost_usd": cost}


if __name__ == "__main__":
    import json
    import sys

    for s in sys.argv[1:] or ["bedrock:sonnet"]:
        try:
            print(json.dumps(smoke(s), default=str))
        except Exception as e:  # report and keep going so one model's access problem does not hide another's
            print(json.dumps({"model": s, "error": f"{type(e).__name__}: {e}"[:600]}))
