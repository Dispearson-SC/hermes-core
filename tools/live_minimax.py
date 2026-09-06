"""Drive the extracted core against real MiniMax models.

Everything else in this repository proves the core works against a scripted provider.
This proves it against a real one: a live model, over the network, deciding for itself
whether to call a tool.

Costs money and needs `MINIMAX_API_KEY` in `Core/.env`, so it is a script rather than
part of the test suite -- a suite that quietly bills the person running it is a suite
people stop running.

    python tools/live_minimax.py                    # every known model
    python tools/live_minimax.py MiniMax-M2.7       # just one
    python tools/live_minimax.py --api-mode anthropic_messages
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

# MiniMax speaks two protocols on two hosts. The OpenAI-compatible one is the default
# here because it exercises the transport most hosts will use; the Anthropic-compatible
# one is what upstream Hermes selects, and is worth checking too.
ENDPOINTS = {
    "chat_completions": "https://api.minimax.io/v1",
    "anthropic_messages": "https://api.minimax.io/anthropic",
}

MODELS = [
    "MiniMax-M2",
    "MiniMax-M2.1",
    "MiniMax-M2.5",
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M3",
]

WEATHER_SCHEMA = {
    "name": "get_weather",
    "description": "Look up the current temperature in a city. Use this whenever the "
                   "user asks about weather; do not guess.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


def load_key() -> str:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    key = os.environ.get("MINIMAX_API_KEY", "")
    if not key:
        print("MINIMAX_API_KEY is not set (expected in Core/.env)", file=sys.stderr)
        raise SystemExit(2)
    return key


def configure(api_mode: str) -> None:
    from hermes_core.seams.config import DictConfigSource, set_config_source
    from hermes_core.seams.credentials import StaticCredentials, set_credential_source
    from hermes_core.seams.paths import DirectoryWorkspace, set_workspace

    set_workspace(DirectoryWorkspace(tempfile.mkdtemp(prefix="hermes-core-live-")))
    set_config_source(DictConfigSource({"model": {"provider": "minimax"}}))
    set_credential_source(
        StaticCredentials(os.environ["MINIMAX_API_KEY"], base_url=ENDPOINTS[api_mode])
    )


def register_weather_tool(calls: list) -> None:
    from hermes_core.tools.registry import registry, tool_result

    def handler(args, **_kwargs):
        calls.append(args)
        city = str(args.get("city") or "").strip()
        # A fixed answer, so a wrong reply means the model ignored the tool rather
        # than the tool being wrong.
        return tool_result(city=city, temp_c=21, conditions="clear")

    registry.register(
        name="get_weather", toolset="live", schema=WEATHER_SCHEMA,
        handler=handler, override=True,
    )


def run_one(model: str, api_mode: str, prompt: str) -> dict:
    from hermes_core.run_agent import AIAgent

    calls: list = []
    register_weather_tool(calls)

    agent = AIAgent(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url=ENDPOINTS[api_mode],
        provider="minimax",
        api_mode=api_mode,
        model=model,
        enabled_toolsets=["live"],
        quiet_mode=True,
        max_iterations=4,
    )

    started = time.monotonic()
    try:
        result = agent.run_conversation(prompt)
    except Exception as exc:  # a provider that refuses is a result, not a crash
        return {
            "model": model, "ok": False, "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.monotonic() - started, "tool_calls": calls,
            "traceback": traceback.format_exc(),
        }

    return {
        "model": model,
        "ok": bool(result.get("completed")),
        "answer": (result.get("final_response") or "").strip(),
        "api_calls": result.get("api_calls"),
        "tool_calls": calls,
        "seconds": time.monotonic() - started,
        "error": result.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=None)
    parser.add_argument("--api-mode", default="chat_completions", choices=sorted(ENDPOINTS))
    parser.add_argument(
        "--prompt",
        default="¿Qué temperatura hace en Rosario? Usá la herramienta y respondé en una frase corta.",
    )
    args = parser.parse_args()

    load_key()
    configure(args.api_mode)

    models = args.models or MODELS
    print(f"api_mode: {args.api_mode}   endpoint: {ENDPOINTS[args.api_mode]}")
    print(f"prompt:   {args.prompt}\n")

    results = []
    for model in models:
        print(f"--- {model} ---", flush=True)
        outcome = run_one(model, args.api_mode, args.prompt)
        results.append(outcome)

        status = "OK " if outcome["ok"] else "FAIL"
        print(f"  {status} {outcome['seconds']:.1f}s  api_calls={outcome.get('api_calls')}")
        if outcome["tool_calls"]:
            print(f"  tool ran with: {outcome['tool_calls']}")
        else:
            print("  tool NOT called")
        if outcome.get("answer"):
            print(f"  answer: {outcome['answer'][:300]}")
        if outcome.get("error"):
            print(f"  error:  {str(outcome['error'])[:300]}")
        print(flush=True)

    print("=" * 72)
    print(f"{'model':26} {'turn':6} {'tool':6} {'calls':6} {'seconds':>8}")
    for outcome in results:
        print(f"{outcome['model']:26} "
              f"{'ok' if outcome['ok'] else 'fail':6} "
              f"{'yes' if outcome['tool_calls'] else 'no':6} "
              f"{str(outcome.get('api_calls') or '-'):6} "
              f"{outcome['seconds']:>8.1f}")

    working = sum(1 for r in results if r["ok"] and r["tool_calls"])
    print(f"\n{working}/{len(results)} models completed a turn and used the tool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
