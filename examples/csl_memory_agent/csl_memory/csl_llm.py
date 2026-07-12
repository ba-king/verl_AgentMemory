from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any


class CSLJsonError(RuntimeError):
    pass


class CSLLLMClient:
    """Tiny OpenAI-compatible client with a deterministic mock fallback."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.provider = str(self.config.get("provider", "mock")).lower()
        self.model = self.config.get("model", "mock")
        self.temperature = float(self.config.get("temperature", 0))
        self.max_retries = int(self.config.get("max_retries", 3))
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.latency_ms = 0.0
        self.last_usage: dict[str, int] = {}

    def call_text_prompt(self, prompt: str) -> str:
        if self.provider == "mock":
            return self._mock_text(prompt)
        response = self._call_openai_compatible(prompt)
        return response

    def call_json_prompt(self, prompt: str) -> dict[str, Any]:
        if self.provider == "mock":
            return self._mock_json(prompt)
        last_error: Exception | None = None
        for _attempt in range(self.max_retries):
            try:
                return repair_json(self.call_text_prompt(prompt))
            except Exception as exc:  # pragma: no cover - external API path
                last_error = exc
                time.sleep(0.5)
        raise CSLJsonError(f"Could not parse JSON response: {last_error}") from last_error

    def _call_openai_compatible(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return self._call_openai_compatible_stdlib(prompt)

        client = OpenAI(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("base_url"),
            timeout=self.config.get("timeout", 120),
        )
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "Return only what the user prompt requests."},
                {"role": "user", "content": prompt},
            ],
        )
        self._record_usage(response.usage, (time.perf_counter() - start) * 1000)
        return response.choices[0].message.content or ""

    def _call_openai_compatible_stdlib(self, prompt: str) -> str:
        base_url = str(self.config.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        api_key = self.config.get("api_key")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "Return only what the user prompt requests."},
                {"role": "user", "content": prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=float(self.config.get("timeout", 120))) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - external API path
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CSL LLM API HTTP {exc.code}: {detail}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"CSL LLM API response has no choices: {body}")
        self._record_usage(body.get("usage"), (time.perf_counter() - start) * 1000)
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    def usage_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
        }

    def _record_usage(self, usage: Any, latency_ms: float) -> None:
        self.calls += 1
        self.latency_ms += latency_ms
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        else:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _mock_text(self, prompt: str) -> str:
        if "Return plain text" in prompt and "[CSL SOFT VERIFIER]" in prompt:
            return (
                "[CSL SOFT VERIFIER]\n"
                "Before final action, verify:\n"
                "1. The action is consistent with final states in memory.\n"
                "2. Active constraints have not been violated.\n"
                "3. Uncertain or missing fields are acknowledged before committing."
            )
        return "{}"

    def _mock_json(self, prompt: str) -> dict[str, Any]:
        if "checkpoint summarizer" in prompt.lower():
            return {
                "summary": "Mock summary generated from visible trace.",
                "outcome": "uncertain",
                "final_decisions": [],
                "important_observations": [],
                "rejected_or_failed_attempts": [],
                "discovered_constraints": [],
                "unresolved_uncertainties": [],
                "evidence_event_ids": [],
            }
        if "schema induction" in prompt.lower() or "schema card" in prompt.lower():
            return {
                "task_family_description": "Execution state induced from visible trace.",
                "entity_types": [],
                "state_variables": [],
                "constraint_types": [],
                "dependency_types": [],
                "status_rules": {},
                "provenance_requirements": ["source_event_ids"],
                "update_rules": [],
                "schema_update_notes": [],
                "confidence": 0.55,
            }
        return {}


def repair_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise CSLJsonError("Empty JSON response")
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise CSLJsonError("JSON response is not an object")
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
        raise CSLJsonError("JSON response is not an object")
