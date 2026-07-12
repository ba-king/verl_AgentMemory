from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .csl_audit import canonical_json_hash
from .csl_schema import (
    Checkpoint,
    DependencyEdge,
    Event,
    MaterializedState,
    StateDelta,
    TaskContractCard,
    TaskSchemaCard,
    from_dict,
    to_dict,
)


class JSONLCSLStorage:
    """Small append-only JSONL store for CSL example state."""

    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.state_dir / "events.jsonl"
        self.checkpoints_path = self.state_dir / "checkpoints.jsonl"
        self.deltas_path = self.state_dir / "deltas.jsonl"
        self.edges_path = self.state_dir / "edges.jsonl"
        self.contract_path = self.state_dir / "task_contract.json"
        self.schema_path = self.state_dir / "schema_card.json"
        self.materialized_path = self.state_dir / "materialized_state.json"

    def reset(self) -> None:
        for path in [
            self.events_path,
            self.checkpoints_path,
            self.deltas_path,
            self.edges_path,
            self.contract_path,
            self.schema_path,
            self.materialized_path,
        ]:
            if path.exists():
                path.unlink()

    def append_event(self, event: Event) -> None:
        self._append_jsonl(self.events_path, [to_dict(event)])

    def load_events(self) -> list[Event]:
        return [from_dict(Event, row) for row in self._read_jsonl(self.events_path)]

    def count_events(self) -> int:
        return len(self._read_jsonl(self.events_path))

    def append_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._append_jsonl(self.checkpoints_path, [to_dict(checkpoint)])

    def load_checkpoints(self) -> list[Checkpoint]:
        return [from_dict(Checkpoint, row) for row in self._read_jsonl(self.checkpoints_path)]

    def append_deltas(self, deltas: Iterable[StateDelta]) -> None:
        self._append_jsonl(self.deltas_path, [to_dict(delta) for delta in deltas])

    def load_deltas(self) -> list[StateDelta]:
        return [from_dict(StateDelta, row) for row in self._read_jsonl(self.deltas_path)]

    def count_deltas(self) -> int:
        return len(self._read_jsonl(self.deltas_path))

    def rewrite_deltas(self, deltas: Iterable[StateDelta]) -> None:
        self._write_jsonl(self.deltas_path, [to_dict(delta) for delta in deltas])

    def append_edges(self, edges: Iterable[DependencyEdge]) -> None:
        self._append_jsonl(self.edges_path, [to_dict(edge) for edge in edges])

    def load_edges(self) -> list[DependencyEdge]:
        return [from_dict(DependencyEdge, row) for row in self._read_jsonl(self.edges_path)]

    def upsert_contract_card(self, contract: TaskContractCard) -> None:
        self.contract_path.write_text(json.dumps(to_dict(contract), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_contract_card(self) -> TaskContractCard | None:
        if not self.contract_path.exists():
            return None
        return from_dict(TaskContractCard, json.loads(self.contract_path.read_text(encoding="utf-8")))

    def current_contract_version(self) -> int:
        contract = self.load_contract_card()
        return contract.version if contract else 0

    def upsert_schema_card(self, schema_card: TaskSchemaCard) -> None:
        self.schema_path.write_text(json.dumps(to_dict(schema_card), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_schema_card(self) -> TaskSchemaCard | None:
        if not self.schema_path.exists():
            return None
        return from_dict(TaskSchemaCard, json.loads(self.schema_path.read_text(encoding="utf-8")))

    def current_schema_version(self) -> int:
        schema = self.load_schema_card()
        return schema.version if schema else 0

    def save_materialized_state(self, states: Iterable[MaterializedState]) -> None:
        rows = [to_dict(state) for state in states]
        self.materialized_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_materialized_state(self) -> list[MaterializedState]:
        if not self.materialized_path.exists():
            return []
        rows = json.loads(self.materialized_path.read_text(encoding="utf-8"))
        return [from_dict(MaterializedState, row) for row in rows]

    def state_version(self) -> str:
        """Hash semantic decision state while ignoring internal audit churn."""
        contract = self.load_contract_card()
        states = sorted(
            (_semantic_state_row(state) for state in self.load_materialized_state()),
            key=lambda row: (
                str((row.get("attributes") or {}).get("scope_id")),
                str(row.get("subject")),
                str(row.get("variable")),
            ),
        )
        events = [
            _semantic_event_row(event)
            for event in self.load_events()
            if "dart-csl guard blocked" not in event.raw_chunk.lower()
        ]
        return canonical_json_hash(
            {
                "contract": {
                    "id": contract.contract_id if contract else None,
                    "version": contract.version if contract else 0,
                    "source_hash": contract.source_hash if contract else None,
                },
                "visible_events": events,
                "materialized_state": states,
            }
        )

    def load_schema_card_summary(self) -> str:
        schema = self.load_schema_card()
        if schema is None:
            return "None"
        variables = ", ".join(item.get("name", "") for item in schema.state_variables[:8])
        constraints = ", ".join(item.get("name", "") for item in schema.constraint_types[:6])
        return (
            f"schema_id={schema.schema_id}; version={schema.version}; "
            f"description={schema.task_family_description}; variables=[{variables}]; "
            f"constraints=[{constraints}]"
        )

    def load_contract_summary(self, limit: int = 8) -> str:
        contract = self.load_contract_card()
        if contract is None:
            return "None"
        rules = []
        for item in (contract.grounding_rules + contract.invariant_constraints + contract.completion_conditions)[:limit]:
            rule = item.get("rule") or item.get("constraint") or item.get("condition") or item.get("pattern") or str(item)
            rules.append(str(rule))
        return (
            f"contract_id={contract.contract_id}; version={contract.version}; "
            f"summary={contract.task_role_summary}; rules={rules}"
        )

    def load_materialized_state_summary(self, limit: int = 24) -> str:
        states = self.load_materialized_state()
        if not states:
            return "None"
        lines = []
        for state in states[:limit]:
            lines.append(f"{state.subject}.{state.variable}={state.value!r} ({state.status})")
        if len(states) > limit:
            lines.append(f"... {len(states) - limit} more")
        return "\n".join(lines)

    def _append_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows


def _semantic_state_row(state: MaterializedState) -> dict[str, Any]:
    attributes = {
        key: value
        for key, value in state.attributes.items()
        if key not in {"short_evidence", "conflict_delta_ids"}
    }
    return {
        "subject": state.subject,
        "variable": state.variable,
        "value": state.value,
        "attributes": attributes,
        "status": state.status,
        "contract_id": state.contract_id,
        "schema_id": state.schema_id,
        "valid": state.valid,
        "publishable": state.publishable,
    }


def _semantic_event_row(event: Event) -> dict[str, Any]:
    return {
        "action": event.action,
        "observation": event.observation,
        "reward": event.reward,
        "feedback": event.feedback,
        "is_final_action": event.is_final_action,
        "is_success": event.is_success,
        "is_failure": event.is_failure,
        "session_id": event.session_id,
    }
