from __future__ import annotations

import time
from typing import Any

from .schemas import ActionTrace, MemoryAction, MemoryEntry, MemoryEdge, utc_now
from .store import SQLiteMemoryStore


class MemoryExecutor:
    """Deterministic mutation layer for memory actions.

    Policies can propose actions, but only this executor mutates the store.
    """

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def execute(
        self,
        action: MemoryAction | str,
        payload: dict[str, Any] | None = None,
        *,
        phase: str = "remember",
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        payload = payload or {}
        action_value = action.value if isinstance(action, MemoryAction) else str(action)
        valid = True
        error_code: str | None = None
        result: dict[str, Any]

        try:
            match action_value:
                case MemoryAction.WRITE_NEW.value:
                    result = self._write_new(payload)
                case MemoryAction.UPDATE_EXISTING.value:
                    result = self._update_existing(payload)
                case MemoryAction.DEPRECATE.value:
                    result = self._deprecate(payload)
                case MemoryAction.MERGE.value:
                    result = self._merge(payload)
                case MemoryAction.NO_OP.value:
                    result = {"ok": True, "action": action_value, "reason": payload.get("reason", "")}
                case _:
                    valid = False
                    error_code = "UNKNOWN_ACTION"
                    result = {"ok": False, "error": f"Unknown action: {action_value}"}
        except Exception as exc:  # keep traces even for invalid policy proposals
            valid = False
            error_code = type(exc).__name__
            result = {"ok": False, "error": str(exc)}

        latency_ms = (time.perf_counter() - started) * 1000
        self.store.append_action_trace(
            ActionTrace(
                phase=phase,  # type: ignore[arg-type]
                action=action_value,
                observation_json=observation or {},
                action_json={"action": action_value, "payload": payload},
                tool_result_json=result,
                valid_format=valid,
                error_code=error_code,
                latency_ms=latency_ms,
            )
        )
        return result

    def _coerce_entry(self, payload: dict[str, Any]) -> MemoryEntry:
        if "entry" in payload:
            entry_payload = payload["entry"]
            if isinstance(entry_payload, MemoryEntry):
                return entry_payload
            return MemoryEntry.model_validate(entry_payload)
        return MemoryEntry.model_validate(payload)

    def _write_new(self, payload: dict[str, Any]) -> dict[str, Any]:
        entry = self._coerce_entry(payload)
        now = utc_now()
        entry.created_at = entry.created_at or now
        entry.updated_at = now
        self.store.upsert_memory(entry)
        return {"ok": True, "memory_id": entry.memory_id, "version": entry.version}

    def _update_existing(self, payload: dict[str, Any]) -> dict[str, Any]:
        memory_id = payload["memory_id"]
        current = self.store.get_memory(memory_id)
        if current is None:
            raise ValueError(f"Memory not found: {memory_id}")

        self.store.add_memory_version(current, MemoryAction.UPDATE_EXISTING.value)
        updates = payload.get("updates", {})
        before_snapshot = current.model_dump(mode="json")
        for field, value in updates.items():
            if field in {"memory_id", "created_at", "version"}:
                continue
            if hasattr(current, field):
                setattr(current, field, value)
        current.version += 1
        current.updated_at = utc_now()
        current.history.append(
            {
                "version": before_snapshot["version"],
                "content": before_snapshot["content"],
                "summary": before_snapshot["summary"],
                "superseded_at": current.updated_at.isoformat(),
            }
        )
        self.store.upsert_memory(current)
        return {"ok": True, "memory_id": current.memory_id, "version": current.version}

    def _deprecate(self, payload: dict[str, Any]) -> dict[str, Any]:
        memory_id = payload["memory_id"]
        current = self.store.get_memory(memory_id)
        if current is None:
            raise ValueError(f"Memory not found: {memory_id}")

        self.store.add_memory_version(current, MemoryAction.DEPRECATE.value)
        current.status = "deprecated"
        current.version += 1
        current.updated_at = utc_now()
        current.metadata["deprecation_reason"] = payload.get("reason", "")
        self.store.upsert_memory(current)
        return {"ok": True, "memory_id": current.memory_id, "status": current.status, "version": current.version}

    def _merge(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_ids = payload["source_memory_ids"]
        sources = self.store.get_memories(source_ids)
        if len(sources) != len(source_ids):
            missing = sorted(set(source_ids) - {entry.memory_id for entry in sources})
            raise ValueError(f"Missing source memories for merge: {missing}")

        merged_payload = payload.get("entry") or {
            "content": payload.get("content") or " ".join(entry.content for entry in sources),
            "summary": payload.get("summary") or " / ".join(entry.summary for entry in sources),
            "user_id": sources[0].user_id,
            "project_id": sources[0].project_id,
            "scope": sources[0].scope,
            "facets": sorted({facet for entry in sources for facet in entry.facets}),
            "source_refs": sorted({ref for entry in sources for ref in entry.source_refs} | set(source_ids)),
            "confidence": min(entry.confidence for entry in sources),
            "importance": max(entry.importance for entry in sources),
            "keywords": sorted({keyword for entry in sources for keyword in entry.keywords}),
            "entities": sorted({entity for entry in sources for entity in entry.entities}),
        }
        merged = MemoryEntry.model_validate(merged_payload)
        self.store.upsert_memory(merged)

        old_status = payload.get("old_status", "deprecated")
        if old_status not in {"deprecated", "superseded"}:
            old_status = "deprecated"
        for source in sources:
            self.store.add_memory_version(source, MemoryAction.MERGE.value)
            source.status = old_status
            source.version += 1
            source.updated_at = utc_now()
            source.metadata["merged_into"] = merged.memory_id
            self.store.upsert_memory(source)
            self.store.upsert_edge(
                MemoryEdge(
                    src_memory_id=merged.memory_id,
                    dst_memory_id=source.memory_id,
                    relation_type="DERIVED_FROM",
                    evidence="MERGE action",
                )
            )

        return {
            "ok": True,
            "memory_id": merged.memory_id,
            "merged_from": source_ids,
            "old_status": old_status,
        }
