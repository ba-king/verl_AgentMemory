from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from .csl_memory import DARTCSLMemorySystem


class DARTCSLMemoryArenaMemorySystem:
    """MemoryArena-compatible wrapper exposing add_chunk and wrap_user_prompt."""

    def __init__(
        self,
        user_id: str | None = None,
        state_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ):
        config = dict(config or {})
        state_root = state_dir or config.get("state_dir")
        if state_root is not None:
            namespace = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(user_id or uuid.uuid4().hex))
            config["state_dir"] = str(Path(state_root) / namespace)
        self.user_id = user_id
        self.system = DARTCSLMemorySystem(config=config, user_id=user_id)
        self._contract_initialized = self.system.storage.load_contract_card() is not None
        self._contract_scope: str | None = None
        self.metrics_path = self.system.storage.state_dir / "metrics.json"
        self._persist_metrics()

    def add_chunk(self, chunk: str) -> dict[str, Any]:
        result = self.system.add_chunk(chunk)
        self._persist_metrics()
        return result

    def add(self, chunk: str) -> dict[str, Any]:
        return self.add_chunk(chunk)

    def wrap_user_prompt(self, question: str) -> str:
        scope = _contract_scope(question)
        refresh_contract = (
            not self._contract_initialized
            or (scope is not None and scope != self._contract_scope)
        )
        if scope is not None and scope != self._contract_scope:
            self.system.set_scope(scope)
        prompt = self.system.wrap_user_prompt(
            question,
            task_instruction=question if refresh_contract else None,
        )
        self._contract_initialized = True
        if scope is not None:
            self._contract_scope = scope
        self._persist_metrics()
        return prompt

    def guard_action(self, question: str, action: str) -> dict[str, Any]:
        result = self.system.guard_action(question, action)
        self._persist_metrics()
        return result

    def validate_transition(self, question: str, action: str) -> dict[str, Any]:
        result = self.system.validate_transition(question, action)
        self._persist_metrics()
        return result

    def audit_selection(self, question: str, selected_option_id: str) -> dict[str, Any]:
        result = self.system.audit_selection(question, selected_option_id)
        self._persist_metrics()
        return result

    def get_metrics(self) -> dict[str, int | float]:
        return self.system.get_metrics()

    @property
    def last_state_packet(self) -> str | None:
        return self.system.last_state_packet

    def _persist_metrics(self) -> None:
        payload = {
            "user_id": self.user_id,
            "task_id": self.system.task_id,
            "state_version": self.system.storage.state_version(),
            "metrics": self.system.get_metrics(),
        }
        temporary = self.metrics_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.metrics_path)


def _contract_scope(question: str) -> str | None:
    current_step = re.search(
        r"(?:What You Need.*?Current|Current\s+(?:Step|Stage|Task|Question)).*?"
        r"\b(Product|Step|Stage|Task|Question|Round)\s+(\d+)\s*:",
        question,
        flags=re.I | re.S,
    )
    if current_step:
        return f"{current_step.group(1).lower()}-{current_step.group(2)}"
    headings = re.findall(
        r"(?m)^\s*(Product|Step|Stage|Task|Question|Round)\s+(\d+)\s*:",
        question,
        flags=re.I,
    )
    if len(headings) == 1:
        label, index = headings[0]
        return f"{label.lower()}-{index}"
    return None
