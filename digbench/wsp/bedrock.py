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
}

PRICING = {
    "opus": dict(input=5.0, output=25.0, cached_input=0.5, cache_write=6.25),
    "astra": dict(input=11.0, output=55.0, cached_input=1.10, cache_write=13.75),
    "sonnet": dict(input=3.0, output=15.0, cached_input=0.3, cache_write=3.75),
}


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
    llm = _astra() if key == "astra" else _anthropic(key)
    return llm, PRICING[key]


def smoke(spec: str = "bedrock:sonnet") -> dict:
    """One tiny call; returns text, token usage and the computed cost."""
    from weightspace.usage import Pricing

    llm, price = get(spec)
    r = llm.complete([{"role": "user", "content": "Reply with exactly: ok"}])
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
