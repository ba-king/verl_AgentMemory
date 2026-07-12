from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official MemoryArena shopping runner while preserving exact API token usage."
    )
    parser.add_argument("--memoryarena-dir", required=True)
    parser.add_argument("--usage-log", required=True)
    args, runner_args = parser.parse_known_args()

    root = Path(args.memoryarena_dir).expanduser().resolve()
    if not (root / "run_shopping.py").is_file():
        raise FileNotFoundError(f"MemoryArena run_shopping.py not found: {root}")
    usage_log = Path(args.usage_log).expanduser().resolve()
    usage_log.parent.mkdir(parents=True, exist_ok=True)
    os.environ["CSL_ANSWER_USAGE_LOG"] = str(usage_log)

    sys.path.insert(0, str(root))
    os.chdir(root)

    import run_shopping  # type: ignore[import-not-found]
    from agent.webshop import (  # type: ignore[import-not-found]
        LLMFatalError,
        LLMRetryableError,
        WebShopAgent,
        _is_billing_error,
    )

    class UsageTrackingWebShopAgent(WebShopAgent):
        def _create_completion(self, messages: list[dict[str, str]]) -> str:
            payload = list(messages)
            transition_repair = bool(getattr(self, "_dart_csl_transition_repair", False))
            if not any((message.get("role") or "").lower() == "system" for message in payload):
                payload = [{"role": "system", "content": self.system_prompt}] + payload

            try:
                max_retries = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "3")))
            except ValueError:
                max_retries = 3
            params: dict[str, Any] = {
                "model": self.model_name,
                "messages": payload,
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                token_key = "max_completion_tokens" if self._use_completion_tokens_param() else "max_tokens"
                params[token_key] = self.max_tokens

            for attempt in range(1, max_retries + 1):
                started = time.perf_counter()
                try:
                    response = self.client.chat.completions.create(**params)
                    latency_ms = (time.perf_counter() - started) * 1000
                    message = response.choices[0].message
                    self._last_raw_output = message.content or ""
                    self._last_reasoning_content = getattr(message, "reasoning_content", None)
                    usage = response.usage.model_dump() if response.usage is not None else {}
                    _append_usage(
                        usage_log,
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "model": self.model_name,
                            "attempt": attempt,
                            "latency_ms": latency_ms,
                            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                            "completion_tokens": int(usage.get("completion_tokens") or 0),
                            "total_tokens": int(usage.get("total_tokens") or 0),
                            "contract_repair": False,
                            "transition_repair": transition_repair,
                        },
                    )
                    contracted = _extract_action_contract(self._last_raw_output)
                    if contracted is None:
                        contracted = _extract_action_contract(self._last_reasoning_content)
                    contract_violation = (
                        _action_contract_violation(contracted, payload)
                        if contracted is not None
                        else "the response did not contain exactly one action"
                    )
                    if contracted is not None and contract_violation is None:
                        return contracted

                    repair_params = dict(params)
                    visible_clicks = _visible_click_values(_current_environment_text(payload))
                    repair_params["messages"] = [
                        *payload,
                        {
                            "role": "user",
                            "content": (
                                f"Your previous response was not executable: {contract_violation}. "
                                "Return only one legal action in the form search[...] or click[...]. "
                                "A click value must occur on the final current environment page, not in memory. "
                                f"Currently visible click values include: {visible_clicks[:30]}. "
                                "Do not explain or reason."
                            ),
                        },
                    ]
                    token_key = "max_completion_tokens" if self._use_completion_tokens_param() else "max_tokens"
                    repair_params[token_key] = 128
                    repair_started = time.perf_counter()
                    repair_response = self.client.chat.completions.create(**repair_params)
                    repair_latency_ms = (time.perf_counter() - repair_started) * 1000
                    repair_message = repair_response.choices[0].message
                    repair_usage = repair_response.usage.model_dump() if repair_response.usage is not None else {}
                    _append_usage(
                        usage_log,
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "model": self.model_name,
                            "attempt": attempt,
                            "latency_ms": repair_latency_ms,
                            "prompt_tokens": int(repair_usage.get("prompt_tokens") or 0),
                            "completion_tokens": int(repair_usage.get("completion_tokens") or 0),
                            "total_tokens": int(repair_usage.get("total_tokens") or 0),
                            "contract_repair": True,
                            "transition_repair": transition_repair,
                            "contract_repair_reason": contract_violation,
                        },
                    )
                    repaired = _extract_action_contract(repair_message.content)
                    if repaired is None:
                        repaired = _extract_action_contract(getattr(repair_message, "reasoning_content", None))
                    if repaired is None:
                        raise ValueError("LLM response contract repair did not produce one legal action")
                    repaired_violation = _action_contract_violation(repaired, payload)
                    if repaired_violation is not None:
                        raise ValueError(
                            f"LLM response contract repair remained illegal: {repaired_violation}"
                        )
                    self._last_raw_output = repaired
                    return repaired
                except Exception as exc:
                    if _is_billing_error(exc):
                        raise LLMFatalError(
                            f"Fatal LLM error (billing/quota): {exc}",
                            original_exception=exc,
                        ) from exc
                    if attempt >= max_retries:
                        raise LLMRetryableError(
                            f"LLM error after {max_retries} attempts: {exc}",
                            original_exception=exc,
                        ) from exc
                    time.sleep(1 + attempt)
            raise LLMRetryableError("LLM generation failed without a response.")

    run_shopping.WebShopAgent = UsageTrackingWebShopAgent
    sys.argv = [str(root / "run_shopping.py"), *runner_args]
    run_shopping.main()


def _append_usage(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_action_contract(value: Any) -> str | None:
    text = str(value or "").strip()
    matches = re.findall(r"\b(search|click)\[([^\]\n]+)\]", text, flags=re.I)
    if len(matches) != 1:
        return None
    action, argument = matches[0]
    argument = " ".join(argument.strip().split())
    if not argument:
        return None
    return f"{action.lower()}[{argument}]"


def _current_environment_text(messages: list[dict[str, Any]]) -> str:
    user_messages = [
        str(message.get("content") or "")
        for message in reversed(messages)
        if str(message.get("role") or "").lower() == "user"
    ]
    latest_user = next(
        (
            content
            for content in user_messages
            if "</memory_context>" in content
            or "[SEP]" in content
            or "Current environment remains unchanged:" in content
        ),
        user_messages[0] if user_messages else "",
    )
    if "</memory_context>" in latest_user:
        latest_user = latest_user.rsplit("</memory_context>", 1)[-1]
    marker = "Current environment remains unchanged:"
    if marker in latest_user:
        latest_user = latest_user.rsplit(marker, 1)[-1]
    return latest_user.strip()


def _visible_click_values(current_environment: str) -> list[str]:
    values = []
    for raw in current_environment.split("[SEP]")[1:]:
        value = " ".join(raw.strip().split())
        if value and value not in values:
            values.append(value)
    return values


def _action_contract_violation(
    action: str,
    messages: list[dict[str, Any]],
) -> str | None:
    if action.startswith("search["):
        return None
    match = re.fullmatch(r"click\[([^\]]+)\]", action, flags=re.I)
    if match is None:
        return "the action is not a supported WebShop command"
    argument = " ".join(match.group(1).strip().split())
    visible_values = _visible_click_values(_current_environment_text(messages))
    if argument.lower() not in {value.lower() for value in visible_values}:
        return f"click value {argument!r} is not present on the final current environment page"
    return None


if __name__ == "__main__":
    main()
