from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .csl_audit import (
    GuardAuditCache,
    certificate_from_structured_event,
    extract_decision_certificate,
    is_selection_action,
)
from .csl_backfill import backfill_missing_fields, detect_missing_fields
from .csl_contract import compile_task_contract, relevant_contract_rules, summarize_task_contract
from .csl_extraction import (
    build_dependency_edges,
    canonicalize_and_validate_deltas,
    extract_state_deltas,
    parse_event_chunk,
    summarize_checkpoint,
)
from .csl_guard import deterministic_guard_precheck, is_irreversible_or_final, llm_hard_action_guard
from .csl_lanes import derive_state_lanes, lanes_to_dict, materialize_state, retrieve_state_for_need
from .csl_llm import CSLLLMClient
from .csl_metrics import DARTCSLMetrics
from .csl_packet import (
    build_dart_state_packet,
    build_decision_support_capsule,
    build_path_summary_packet,
    build_soft_verifier,
    build_state_packet,
)
from .csl_schema import (
    Checkpoint,
    DecisionSignature,
    DependencyEdge,
    Event,
    StateDelta,
    event_from_chunk,
    new_id,
    to_dict,
    utc_now,
)
from .csl_schema_induction import induce_schema_card, summarize_schema_card
from .csl_storage import JSONLCSLStorage
from .csl_task_adapter import (
    ActionDirective,
    NoProgressTracker,
    TaskAdapter,
    TaskAdapterRegistry,
    TaskIR,
    is_declared_commit_action,
    validate_action_directive,
)
from .csl_visible_entities import WebShopTaskAdapter


DEFAULT_CONFIG: dict[str, Any] = {
    "method_name": "dart_csl_vtcc",
    "expanded_name": "Decision-Aware, Action-Guarded, Repairable, Token-Bounded Causal State Ledger with Visible Task Contract Compilation",
    "short_name": "DART-CSL",
    "storage_backend": "jsonl",
    "state_dir": None,
    "schema_induction": {
        "enabled": True,
        "allow_fixed_task_categories": False,
        "allow_category_specific_prompts": False,
        "update_on_write": True,
        "update_on_read": True,
        "min_confidence_to_update": 0.55,
    },
    "llm": {"provider": "mock", "model": "mock", "temperature": 0, "max_retries": 3, "json_repair": True},
    "write_time": {
        "extract_every_n_events": 4,
        "extract_on_boundary_signal": True,
        "extract_on_success_or_failure_signal": True,
        "force_extract_on_wrap": True,
        "lazy_write": True,
        "enable_checkpoint_summary": True,
        "enable_contract_compilation": True,
        "enable_schema_card_induction": True,
        "enable_generic_delta_extractor": True,
        "enable_delta_validator": True,
        "enable_canonicalizer": True,
        "enable_dependency_edges": True,
        "enable_repair_barrier": True,
    },
    "read_time": {
        "enable_state_need_parser": True,
        "enable_backfill": True,
        "max_backfill_events": 24,
        "dependency_expand_depth": 2,
        "max_state_packet_tokens": 800,
        "max_packet_tokens": 800,
        "max_contract_rules": 4,
        "max_final_states": 12,
        "max_constraints": 12,
        "max_dependencies": 6,
        "max_negative_states": 8,
        "max_evidence_pointers": 4,
        "max_active_path_events": 4,
        "include_recent_events": True,
        "recent_event_count": 1,
        "enable_soft_verifier": True,
        "enable_hard_action_guard": True,
        "hard_guard_only_on_irreversible_actions": True,
        "enable_guard_audit_cache": True,
        "guard_audit_policy_version": "selection-proof-v1",
    },
    "progress": {
        "enabled": True,
        "repeat_limit": 2,
        "history_limit": 32,
        "cycle_window": 6,
    },
    "transition_validation": {
        "enabled": True,
        "max_consecutive_same_recovery": 2,
    },
    "state_lanes": {
        "enable_contract_lane": True,
        "enable_decision_lane": True,
        "enable_constraint_lane": True,
        "enable_dependency_lane": True,
        "enable_negative_lane": True,
        "enable_evidence_lane": True,
        "enable_recent_lane": True,
        "enable_uncertainty_lane": True,
        "enable_repair_lane": True,
    },
    "ablation": {
        "disable_task_contract": False,
        "disable_schema_induction": False,
        "disable_ledger": False,
        "disable_constraints": False,
        "disable_dependency_graph": False,
        "disable_negative_memory": False,
        "disable_repair_barrier": False,
        "disable_backfill": False,
        "disable_decision_signature": False,
        "disable_soft_verifier": False,
        "disable_hard_action_guard": False,
        "disable_transition_validation": False,
        "use_path_summary_only": False,
        "use_semantic_retrieval_only": False,
    },
}


class DARTCSLMemorySystem:
    """Agent-safe DART-CSL memory backend."""

    def __init__(self, config: dict[str, Any] | None = None, user_id: str | None = None):
        self.config = _deep_merge(DEFAULT_CONFIG, config or {})
        self.task_id = user_id or str(self.config.get("task_id") or new_id("task"))
        state_dir = self.config.get("state_dir")
        if not state_dir:
            state_dir = Path(tempfile.mkdtemp(prefix="csl_memory_")) / self.task_id
        self.storage = JSONLCSLStorage(state_dir)
        self.audit_cache = GuardAuditCache(
            self.storage.state_dir,
            policy_version=str(self.config["read_time"].get("guard_audit_policy_version", "selection-proof-v1")),
        )
        self.llm = CSLLLMClient(self.config.get("llm", {}))
        self.adapter_registry = TaskAdapterRegistry([WebShopTaskAdapter()])
        self._scope_adapters: dict[str, Any] = {}
        progress_config = self.config.get("progress", {})
        self.progress = NoProgressTracker(
            repeat_limit=int(progress_config.get("repeat_limit", 2)),
            history_limit=int(progress_config.get("history_limit", 32)),
            cycle_window=int(progress_config.get("cycle_window", 6)),
        )
        self.recent_events: list[Event] = []
        self.last_state_packet: str | None = None
        self.last_retrieved_lanes: dict[str, Any] | None = None
        self.last_decision_signature: DecisionSignature | None = None
        self.last_capsule: Any = None
        self.last_action_directive: ActionDirective | None = None
        self.current_scope_id: str | None = None
        self.metrics = DARTCSLMetrics()

    def register_task_adapter(self, adapter: TaskAdapter, *, priority: bool = True) -> None:
        """Register a public-protocol adapter without changing the ledger core."""
        self.adapter_registry.register(adapter, priority=priority)

    def set_scope(self, scope_id: str | None) -> None:
        normalized = str(scope_id).strip() if scope_id is not None else None
        normalized = normalized or None
        if normalized == self.current_scope_id:
            return
        if self.recent_events:
            self._flush_recent_events()
        if self.current_scope_id is not None:
            self._emit_decision_receipt(self.current_scope_id)
        self.current_scope_id = normalized
        self.metrics.scope_switches += 1
        self.last_state_packet = None
        self.last_retrieved_lanes = None
        self.last_decision_signature = None
        self.last_capsule = None
        self.last_action_directive = None

    def add_chunk(
        self,
        chunk: str,
        *,
        task_instruction: str | None = None,
        tool_description: str | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if task_instruction or tool_description:
            self.ensure_task_contract(task_instruction=task_instruction, tool_description=tool_description)
        parsed = parse_event_chunk(chunk)
        event = event_from_chunk(
            task_id=self.task_id,
            step_id=self.storage.count_events() + 1,
            chunk=chunk,
            action=parsed.get("action"),
            observation=parsed.get("observation"),
            reward=parsed.get("reward"),
            feedback=parsed.get("feedback"),
            is_final_action=bool(parsed.get("is_final_action")),
            is_success=bool(parsed.get("is_success")),
            is_failure=bool(parsed.get("is_failure")),
            boundary_signal=bool(parsed.get("boundary_signal")),
            boundary_reason=parsed.get("boundary_reason"),
            metadata={key: value for key, value in parsed.items() if key not in {"action", "observation", "reward"}},
            session_id=self.current_scope_id,
        )
        self.storage.append_event(event)
        self.recent_events.append(event)
        if self.config.get("progress", {}).get("enabled", True):
            signal = self.progress.observe(
                scope_id=self.current_scope_id,
                action=parsed.get("action"),
                observation=parsed.get("observation"),
                evidence_items=[
                    *[str(item) for item in parsed.get("explicit_entities", [])],
                    *[str(item) for item in parsed.get("explicit_values", [])],
                ],
            )
            if signal is not None and signal.stale:
                self.metrics.no_progress_events += 1
                self.metrics.stale_actions_detected += 1
        self.last_state_packet = None
        self.last_retrieved_lanes = None
        self.last_decision_signature = None
        self.last_capsule = None
        self.last_action_directive = None

        if not self._ablation("disable_ledger") and self._should_extract(parsed):
            self._flush_recent_events()

        structured = parsed.get("structured_event")
        if isinstance(structured, dict) and str(structured.get("event_type") or "").lower() == "selection_committed":
            certificate = certificate_from_structured_event(parsed)
            if certificate is not None:
                self.audit_cache.bind_selection_commit(
                    certificate=certificate,
                    state_version=self.storage.state_version(),
                )
                self.metrics.guard_audit_records = self.audit_cache.count()

        self.metrics.add_chunk_latency_ms = (time.perf_counter() - start) * 1000
        self.metrics.num_total_deltas = self.storage.count_deltas()
        self.metrics.num_active_deltas = len(self.storage.load_materialized_state())
        return {
            "method_name": self.config.get("method_name"),
            "num_events": self.storage.count_events(),
            "num_deltas": self.storage.count_deltas(),
            "contract_version": self.storage.current_contract_version(),
            "schema_version": self.storage.current_schema_version(),
            "state_version": self.storage.state_version(),
        }

    def add(self, chunk: str) -> dict[str, Any]:
        return self.add_chunk(chunk)

    def wrap_user_prompt(
        self,
        question: str,
        *,
        task_instruction: str | None = None,
        tool_description: str | None = None,
    ) -> str:
        start = time.perf_counter()
        if task_instruction or tool_description:
            self.ensure_task_contract(task_instruction=task_instruction, tool_description=tool_description)
        if self._ablation("use_path_summary_only"):
            packet = self._build_path_summary(question)
            self.last_state_packet = packet
            return f"<memory_context>\n{packet}\n</memory_context>\n\n{question}"

        if self.config["write_time"].get("force_extract_on_wrap", True) and self.recent_events:
            self._flush_recent_events()

        contract = self.storage.load_contract_card()
        if contract is None and not self._ablation("disable_task_contract"):
            contract = self.ensure_task_contract(task_instruction=task_instruction or question, tool_description=tool_description)

        schema_card = self.storage.load_schema_card()
        if self.config["schema_induction"].get("enabled", True) and not self._ablation("disable_schema_induction"):
            schema_card = induce_schema_card(
                task_id=self.task_id,
                task_instruction=question,
                environment_description=summarize_task_contract(contract),
                existing_schema=schema_card,
                existing_state_summary=self.storage.load_materialized_state_summary(),
            )
            self.storage.upsert_schema_card(schema_card)

        adapter = self.adapter_registry.resolve(question)
        if self.current_scope_id is not None:
            existing_adapter = self._scope_adapters.get(self.current_scope_id)
            if existing_adapter is None or adapter.name != "generic":
                self._scope_adapters[self.current_scope_id] = adapter
        task_ir = adapter.compile_task(question)
        progress_context = self.progress.context(self.current_scope_id)
        scoped_state = self._scoped_materialized_state()
        decision_signature = self._parse_decision_signature(question, contract, schema_card)
        state_need = to_dict(decision_signature)
        lane_config = self._lane_config()
        lanes = derive_state_lanes(
            materialized_state=scoped_state,
            deltas=self._scoped_deltas(),
            edges=self.storage.load_edges(),
            recent_events=self._scoped_events(self.recent_events or self.storage.load_events()),
            contract=contract,
            decision_signature=state_need,
            config=lane_config,
        )
        retrieved = retrieve_state_for_need(state_need=state_need, lanes=lanes)
        missing_fields = detect_missing_fields(
            state_need=state_need,
            retrieved_lanes=lanes_to_dict(retrieved),
            schema_card=schema_card,
        )

        if missing_fields and self.config["read_time"].get("enable_backfill", True) and not self._ablation("disable_backfill"):
            events = self._scoped_events(self.storage.load_events())[
                -int(self.config["read_time"].get("max_backfill_events", 30)) :
            ]
            backfilled = backfill_missing_fields(
                task_id=self.task_id,
                schema_card=schema_card,
                missing_fields=missing_fields,
                events=events,
                question=question,
            )
            if backfilled:
                for delta in backfilled:
                    if delta.session_id is None:
                        delta.session_id = self.current_scope_id
                self.storage.append_deltas(backfilled)
                self.metrics.backfill_count += len(backfilled)
                self._materialize_state()
                scoped_state = self._scoped_materialized_state()
                lanes = derive_state_lanes(
                    materialized_state=scoped_state,
                    deltas=self._scoped_deltas(),
                    edges=self.storage.load_edges(),
                    recent_events=self._scoped_events(self.storage.load_events()),
                    contract=contract,
                    decision_signature=state_need,
                    config=lane_config,
                )
                retrieved = retrieve_state_for_need(state_need=state_need, lanes=lanes)

        adapter_missing = _adapter_dependency_gaps(task_ir, scoped_state)
        existing_missing = {
            (str(item.get("subject") or ""), str(item.get("variable") or ""))
            for item in missing_fields
        }
        missing_fields.extend(
            item
            for item in adapter_missing
            if (str(item.get("subject") or ""), str(item.get("variable") or ""))
            not in existing_missing
        )

        candidate_ledger = adapter.build_candidate_ledger(
            task=task_ir,
            question=question,
            materialized_state=scoped_state,
            progress=progress_context,
        )
        if candidate_ledger and self.config.get("progress", {}).get("enabled", True):
            progress_signal = self.progress.observe_decision_state(
                scope_id=self.current_scope_id,
                decision_state={
                    "candidate_ledger": _decision_state_ledger(candidate_ledger),
                    "missing_variables": sorted(
                        str(item.get("variable") or "") for item in missing_fields
                    ),
                },
            )
            if progress_signal is not None:
                self.metrics.no_progress_events += 1
                self.metrics.stale_actions_detected += 1
                progress_context = self.progress.context(self.current_scope_id)
                candidate_ledger = adapter.build_candidate_ledger(
                    task=task_ir,
                    question=question,
                    materialized_state=scoped_state,
                    progress=progress_context,
                )
        action_directive = _adapter_action_directive(
            adapter=adapter,
            task=task_ir,
            question=question,
            materialized_state=scoped_state,
            progress=progress_context,
        )
        action_directive = self._bind_action_directive(action_directive)
        candidate_guidance = (
            action_directive.display_text
            if action_directive is not None
            else adapter.recommend_next_action(
                task=task_ir,
                question=question,
                materialized_state=scoped_state,
                progress=progress_context,
            )
        )
        active_path = _active_execution_path(
            self._scoped_events(self.storage.load_events()),
            progress=progress_context,
            limit=int(self.config["read_time"].get("max_active_path_events", 4)),
        )

        max_chars = int(
            self.config["read_time"].get(
                "max_state_packet_tokens",
                self.config["read_time"].get("max_packet_tokens", 1200),
            )
        ) * 4
        if self._ablation("disable_decision_signature"):
            packet = build_state_packet(
                question=question,
                schema_card=schema_card,
                state_need=state_need,
                retrieved=retrieved,
                missing_fields=missing_fields,
                max_chars=max_chars,
            )
            capsule = None
        else:
            capsule = build_decision_support_capsule(
                task_id=self.task_id,
                decision_signature=decision_signature,
                contract=contract,
                retrieved=retrieved,
                missing_fields=missing_fields,
                max_items=self.config["read_time"],
            )
            packet = build_dart_state_packet(
                question=question,
                capsule=capsule,
                decision_signature=decision_signature,
                active_path=active_path,
                candidate_ledger=candidate_ledger,
                candidate_guidance=candidate_guidance,
                action_directive=(
                    action_directive.to_dict() if action_directive is not None else None
                ),
                max_chars=max_chars,
            )
        if self.config["read_time"].get("enable_soft_verifier", True) and not self._ablation("disable_soft_verifier"):
            verifier = build_soft_verifier(question=question, state_packet=packet, schema_card=schema_card)
            packet = _fit_packet_with_suffix(packet, verifier, max_chars=max_chars)

        self.last_state_packet = packet
        self.last_retrieved_lanes = lanes_to_dict(retrieved)
        self.last_decision_signature = decision_signature
        self.last_capsule = capsule
        self.last_action_directive = action_directive
        self.metrics.wrap_user_prompt_latency_ms = (time.perf_counter() - start) * 1000
        self.metrics.update_tokens_from_packet(packet)
        self.metrics.contract_tokens_injected = retrieved.get("contract").token_estimate if retrieved.get("contract") else 0
        self.metrics.num_retrieved_constraints = len(retrieved.get("constraint").entries) if retrieved.get("constraint") else 0
        self.metrics.num_retrieved_dependencies = len(retrieved.get("dependency").entries) if retrieved.get("dependency") else 0
        receipt_entries = [
            entry
            for lane_name in ("decision", "dependency")
            for entry in (retrieved.get(lane_name).entries if retrieved.get(lane_name) else [])
            if str(entry.get("variable") or "") == "decision_receipt"
        ]
        self.metrics.cross_scope_receipts_retrieved = len(
            {str(entry.get("subject") or entry.get("state_id") or entry) for entry in receipt_entries}
        )
        self.metrics.num_negative_states_retrieved = len(retrieved.get("negative").entries) if retrieved.get("negative") else 0
        self.metrics.missing_field_count = len(missing_fields)
        return f"<memory_context>\n{packet}\n</memory_context>\n\n{question}"

    def ensure_task_contract(
        self,
        *,
        task_instruction: str | None = None,
        tool_description: str | None = None,
    ):
        if self._ablation("disable_task_contract"):
            return None
        existing = self.storage.load_contract_card()
        contract = compile_task_contract(
            task_id=self.task_id,
            visible_task_prompt=task_instruction,
            tool_description=tool_description,
            existing_contract=existing,
        )
        self.storage.upsert_contract_card(contract)
        return contract

    def validate_transition(self, question: str, action: str) -> dict[str, Any]:
        """Enforce adapter-declared evidence obligations before environment execution."""
        start = time.perf_counter()
        self.metrics.transition_validation_calls += 1
        state_version = self.storage.state_version()
        if (
            not self.config.get("transition_validation", {}).get("enabled", True)
            or self._ablation("disable_transition_validation")
        ):
            result = {
                "allow": True,
                "status": "disabled",
                "directive": None,
                "repair_instruction": None,
                "state_version": state_version,
            }
        else:
            if self.recent_events:
                self._flush_recent_events()
                state_version = self.storage.state_version()
            adapter = self.adapter_registry.resolve(question)
            task_ir = adapter.compile_task(question)
            progress_context = self.progress.context(self.current_scope_id)
            directive = _adapter_action_directive(
                adapter=adapter,
                task=task_ir,
                question=question,
                materialized_state=self._scoped_materialized_state(),
                progress=progress_context,
            )
            directive = self._bind_action_directive(directive)
            self.last_action_directive = directive
            lease_id = self._transition_lease_id(state_version)
            allow, violation = validate_action_directive(
                action,
                directive,
                current_lease_id=lease_id,
                progress=progress_context,
            )
            if not allow:
                self.metrics.transition_validation_blocks += 1
                if violation and "prohibited" in violation:
                    self.metrics.transition_forbidden_action_blocks += 1
                elif violation and "consecutive" in violation:
                    self.metrics.transition_repeat_budget_blocks += 1
                else:
                    self.metrics.transition_action_set_blocks += 1
            if directive is None:
                status = "no_directive"
            elif not directive.is_required:
                status = "advisory"
            elif directive.mode == "required_state_change" and allow:
                status = "state_change_action_allowed"
            elif directive.mode == "required_state_change":
                status = "state_change_action_violation"
            elif allow:
                status = "required_satisfied"
            else:
                status = "required_violation"
            repair = None
            if not allow and directive is not None:
                recoveries = [directive.action] if directive.action else []
                recoveries.extend(item.action for item in directive.recovery_actions)
                choices = "; ".join(dict.fromkeys(recoveries))
                forbidden = "; ".join(directive.forbidden_actions)
                repair = (
                    f"Complete evidence obligation {directive.obligation_id or 'current'} by causing "
                    f"the required state change: {', '.join(directive.required_state_delta) or directive.reason}. "
                    + (f"Do not use: {forbidden}. " if forbidden else "")
                    + (f"Choose one currently legal recovery: {choices}. " if choices else "")
                    + "The previous directive expires after the next environment transition."
                )
            result = {
                "allow": allow,
                "status": status,
                "proposed_action": action,
                "violation": violation,
                "directive": directive.to_dict() if directive is not None else None,
                "repair_instruction": repair,
                "state_version": state_version,
                "transition_lease_id": lease_id,
                "memory_llm_used": False,
            }
        self.metrics.transition_validation_latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _transition_lease_id(self, state_version: str | None = None) -> str:
        events = self._scoped_events(self.storage.load_events())
        latest_event = events[-1].event_id if events else "initial"
        scope = self.current_scope_id or "global"
        return f"{scope}:{latest_event}:{state_version or self.storage.state_version()}"

    def _bind_action_directive(
        self,
        directive: ActionDirective | None,
    ) -> ActionDirective | None:
        if directive is None or not directive.is_required:
            return directive
        repeat_limit = directive.max_consecutive_action_repeats
        if repeat_limit is None:
            repeat_limit = int(
                self.config.get("transition_validation", {}).get(
                    "max_consecutive_same_recovery",
                    2,
                )
            )
        return replace(
            directive,
            lease_id=self._transition_lease_id(),
            max_consecutive_action_repeats=max(1, repeat_limit),
        )

    def audit_selection(self, question: str, selected_option_id: str) -> dict[str, Any]:
        """Audit a visible selection without blocking reversible navigation."""
        self.metrics.selection_audit_calls += 1
        if self.recent_events:
            self._flush_recent_events()
        adapter = self.adapter_registry.resolve(question)
        task_ir = adapter.compile_task(question)
        certificate = adapter.build_decision_certificate(
            task=task_ir,
            question=question,
            materialized_state=self._scoped_materialized_state(),
            selected_option_id=selected_option_id,
        )
        uncertainties = certificate.get("uncertainties") if certificate is not None else []
        if certificate is None or uncertainties:
            self.metrics.selection_audit_deferred += 1
            if certificate is None:
                self.metrics.certificate_build_failures += 1
            if isinstance(uncertainties, list):
                self.metrics.certificate_uncertainty_count += len(uncertainties)
            return {
                "allow": True,
                "status": "deferred",
                "reason": "candidate comparison evidence is not complete yet",
                "missing_evidence": uncertainties if isinstance(uncertainties, list) else [],
                "memory_llm_used": False,
            }

        action = f"select[{selected_option_id}]\nDecisionCertificate: {json.dumps(certificate, ensure_ascii=False)}"
        result = self.guard_action(question, action)
        if result.get("allow") is True:
            state_version = self.storage.state_version()
            record = self.audit_cache.bind_selection_commit(
                certificate=certificate,
                state_version=state_version,
            )
            if record is not None:
                result["audit_cache"] = {
                    "hit": False,
                    "status": "bound",
                    "audit_id": record["audit_id"],
                    "certificate_hash": record["certificate_hash"],
                    "state_version": state_version,
                    "policy_version": record["policy_version"],
                }
        result["status"] = "audited"
        result["decision_certificate_source"] = "memory_state"
        return result

    def guard_action(self, question: str, action: str) -> dict[str, Any]:
        start = time.perf_counter()
        self.metrics.action_guard_calls += 1
        if self.recent_events:
            self._flush_recent_events()
        contract = self.storage.load_contract_card()
        adapter = self.adapter_registry.resolve(question)
        task = adapter.compile_task(question)
        effective_action = action
        certificate = extract_decision_certificate(effective_action)
        certificate_source: str | None = None
        if certificate is None and is_declared_commit_action(task, action):
            synthesized = adapter.build_decision_certificate(
                task=task,
                question=question,
                materialized_state=self._scoped_materialized_state(),
            )
            if synthesized is not None:
                certificate = synthesized
                certificate_source = "memory_state"
                effective_action = (
                    f"{action}\nDecisionCertificate: "
                    f"{json.dumps(certificate, ensure_ascii=False)}"
                )
        state_version = self.storage.state_version()
        cache_enabled = bool(self.config["read_time"].get("enable_guard_audit_cache", True))
        commit_action = is_declared_commit_action(
            task,
            effective_action,
        ) or is_irreversible_or_final(effective_action, contract)
        cache_miss_reason: str | None = None
        if (
            self.config["read_time"].get("hard_guard_only_on_irreversible_actions", True)
            and not is_irreversible_or_final(effective_action, contract)
        ):
            result = {"allow": True, "violations": [], "repair_instruction": None, "needs_backfill": False, "missing_evidence": []}
        else:
            packet = self.last_state_packet or self.wrap_user_prompt(question)
            deterministic = deterministic_guard_precheck(
                action=effective_action,
                contract=contract,
                state_packet=packet,
            )
            if deterministic["allow"] is False or self._ablation("disable_hard_action_guard") or not self.config["read_time"].get("enable_hard_action_guard", True):
                result = deterministic
            else:
                cached_record = None
                cached_source: str | None = None
                if cache_enabled and certificate is not None and commit_action:
                    cached_record, cache_miss_reason = self.audit_cache.lookup_commit(
                        certificate=certificate,
                        state_version=state_version,
                    )
                    if cached_record is not None:
                        cached_source = "selection"
                if cache_enabled and cached_record is None and commit_action:
                    cached_record = self.audit_cache.lookup_denial(
                        action=effective_action,
                        state_version=state_version,
                    )
                    if cached_record is not None:
                        cached_source = "denial"
                if cached_record is not None:
                    cached_result = cached_record.get("result") or {}
                    if cached_source == "selection":
                        result = dict(deterministic)
                        if isinstance(cached_result.get("decision_audit"), dict):
                            result["decision_audit"] = cached_result["decision_audit"]
                    else:
                        result = dict(cached_result)
                    result["memory_llm_used"] = False
                    result["audit_cache"] = {
                        "hit": True,
                        "audit_id": cached_record.get("audit_id"),
                        "certificate_hash": cached_record.get("certificate_hash"),
                        "transition_hash": cached_record.get("transition_hash"),
                        "state_version": state_version,
                        "policy_version": cached_record.get("policy_version"),
                        "reused_from": cached_source,
                    }
                    self.metrics.guard_audit_cache_hits += 1
                    self.metrics.guard_llm_calls_avoided += 1
                else:
                    if cache_enabled and commit_action:
                        self.metrics.guard_audit_cache_misses += 1
                    result = llm_hard_action_guard(
                        question=question,
                        contract=contract,
                        state_packet=packet,
                        action=effective_action,
                        llm=self.llm,
                    )
                    denial_record = None
                    if cache_enabled and commit_action and result.get("allow") is False:
                        denial_record = self.audit_cache.record_denial(
                            action=effective_action,
                            state_version=state_version,
                            result=dict(result),
                        )
                    if denial_record is not None:
                        result["audit_cache"] = {
                            "hit": False,
                            "status": "blocked_recorded",
                            "audit_id": denial_record["audit_id"],
                            "transition_hash": denial_record["transition_hash"],
                            "state_version": state_version,
                            "policy_version": denial_record["policy_version"],
                            "reason": cache_miss_reason,
                        }
                        self.metrics.guard_audit_records = self.audit_cache.count()
                    elif cache_miss_reason is not None:
                        result["audit_cache"] = {
                            "hit": False,
                            "reason": cache_miss_reason,
                            "state_version": state_version,
                        }

        if (
            cache_enabled
            and certificate is not None
            and is_selection_action(effective_action)
            and result.get("allow") is True
        ):
            record = self.audit_cache.record_selection(
                certificate=certificate,
                state_version=state_version,
                result=dict(result),
            )
            if record is not None:
                result["audit_cache"] = {
                    "hit": False,
                    "status": record["status"],
                    "audit_id": record["audit_id"],
                    "certificate_hash": record["certificate_hash"],
                    "state_version": state_version,
                    "policy_version": record["policy_version"],
                }
                self.metrics.guard_audit_records = self.audit_cache.count()
        if certificate_source is not None:
            result["decision_certificate_source"] = certificate_source
        if result.get("allow") is False:
            self.metrics.action_guard_blocks += 1
        llm_usage = self.llm.usage_dict()
        self.metrics.memory_llm_calls = int(llm_usage["calls"])
        self.metrics.memory_llm_prompt_tokens = int(llm_usage["prompt_tokens"])
        self.metrics.memory_llm_completion_tokens = int(llm_usage["completion_tokens"])
        self.metrics.memory_llm_latency_ms = float(llm_usage["latency_ms"])
        self.metrics.read_llm_tokens = int(llm_usage["total_tokens"])
        self.metrics.total_tokens = (
            self.metrics.prompt_tokens
            + self.metrics.generation_tokens
            + self.metrics.write_llm_tokens
            + self.metrics.read_llm_tokens
        )
        self.metrics.guard_action_latency_ms = (time.perf_counter() - start) * 1000
        return result

    def get_metrics(self) -> dict[str, int | float]:
        return self.metrics.as_dict()

    def reset(self) -> None:
        self.storage.reset()
        self.audit_cache.reset()
        self.recent_events.clear()
        self.last_state_packet = None
        self.last_retrieved_lanes = None
        self.last_decision_signature = None
        self.last_capsule = None
        self.last_action_directive = None
        self.current_scope_id = None
        self._scope_adapters.clear()
        self.progress.reset()
        self.metrics = DARTCSLMetrics()

    def _scoped_materialized_state(self):
        states = self.storage.load_materialized_state()
        if self.current_scope_id is None:
            return states
        return [
            state
            for state in states
            if state.attributes.get("scope_id") == self.current_scope_id
            or (
                state.variable == "decision_receipt"
                and state.attributes.get("global") is True
            )
        ]

    def _scoped_deltas(self):
        deltas = self.storage.load_deltas()
        if self.current_scope_id is None:
            return deltas
        return [
            delta
            for delta in deltas
            if delta.session_id == self.current_scope_id
            or (
                delta.variable == "decision_receipt"
                and delta.attributes.get("global") is True
            )
        ]

    def _scoped_events(self, events: list[Event]) -> list[Event]:
        if self.current_scope_id is None:
            return events
        return [event for event in events if event.session_id == self.current_scope_id]

    def _flush_recent_events(self) -> None:
        if not self.recent_events:
            return
        events = list(self.recent_events)
        contract = self.storage.load_contract_card()
        checkpoint_json = summarize_checkpoint(events)
        checkpoint = Checkpoint(
            checkpoint_id=new_id("checkpoint"),
            task_id=self.task_id,
            subtask_id=None,
            start_event_id=events[0].event_id,
            end_event_id=events[-1].event_id,
            summary=checkpoint_json["summary"],
            outcome=checkpoint_json["outcome"],
            covered_delta_ids=[],
            metadata=checkpoint_json,
        )

        schema_card = self.storage.load_schema_card()
        if (
            self.config["write_time"].get("enable_schema_card_induction", True)
            and self.config["schema_induction"].get("enabled", True)
            and not self._ablation("disable_schema_induction")
        ):
            schema_card = induce_schema_card(
                task_id=self.task_id,
                environment_description=summarize_task_contract(contract),
                recent_trace=self._format_events(events),
                existing_schema=schema_card,
                existing_state_summary=self.storage.load_materialized_state_summary(),
            )
            self.storage.upsert_schema_card(schema_card)

        if self.config["write_time"].get("enable_generic_delta_extractor", True):
            deltas = extract_state_deltas(
                task_id=self.task_id,
                checkpoint=checkpoint,
                schema_card=schema_card,
                events=events,
                contract=contract,
            )
        else:
            deltas = []

        if self.config["write_time"].get("enable_canonicalizer", True) or self.config["write_time"].get("enable_delta_validator", True):
            deltas = canonicalize_and_validate_deltas(deltas)
        deltas = self._drop_unchanged_deltas(deltas)

        checkpoint.covered_delta_ids = [delta.delta_id for delta in deltas]
        self.storage.append_checkpoint(checkpoint)
        self.storage.append_deltas(deltas)

        if self.config["write_time"].get("enable_dependency_edges", True) and not self._ablation("disable_dependency_graph"):
            existing_map = self._existing_delta_key_map(exclude_delta_ids={delta.delta_id for delta in deltas})
            edges = build_dependency_edges(task_id=self.task_id, deltas=deltas, existing_delta_ids_by_key=existing_map)
            self.storage.append_edges(edges)

        self._materialize_state()
        self.recent_events.clear()

    def _emit_decision_receipt(self, scope_id: str) -> None:
        existing = [
            delta
            for delta in self.storage.load_deltas()
            if delta.variable == "decision_receipt"
            and delta.attributes.get("source_scope_id") == scope_id
        ]
        if existing:
            return
        scoped_states = [
            state
            for state in self.storage.load_materialized_state()
            if state.attributes.get("scope_id") == scope_id
        ]
        prior_receipts = [
            state
            for state in self.storage.load_materialized_state()
            if state.variable == "decision_receipt"
            and state.attributes.get("global") is True
            and state.attributes.get("source_scope_id") != scope_id
        ]
        scoped_events = [
            event for event in self.storage.load_events() if event.session_id == scope_id
        ]
        if not scoped_states or not scoped_events:
            return
        visible_context = next(
            (
                event.observation or event.raw_chunk
                for event in reversed(scoped_events)
                if event.observation or event.raw_chunk
            ),
            "",
        )
        adapter = self._scope_adapters.get(scope_id)
        if adapter is None:
            adapter = self.adapter_registry.resolve(visible_context)
            if adapter.name == "generic":
                for event in reversed(scoped_events):
                    candidate = self.adapter_registry.resolve(event.observation or event.raw_chunk)
                    if candidate.name != "generic":
                        adapter = candidate
                        break
        receipt = adapter.build_decision_receipt(
            scope_id=scope_id,
            materialized_state=[*scoped_states, *prior_receipts],
            events=scoped_events,
        )
        if receipt is None:
            return
        receipt.state_version = self.storage.state_version()
        receipt_state_ids = set(receipt.evidence_state_ids)
        grounded_states = [
            state for state in scoped_states if state.state_id in receipt_state_ids
        ]
        if not grounded_states:
            grounded_states = scoped_states
        source_delta_ids = sorted(
            {
                str(delta_id)
                for state in grounded_states
                for delta_id in state.source_delta_ids
                if delta_id
            }
        )
        receipt_delta = StateDelta(
            delta_id=new_id("delta"),
            task_id=self.task_id,
            op="FINALIZE",
            subject=f"decision_receipt {scope_id}",
            variable="decision_receipt",
            value=receipt.to_dict(),
            status="final",
            confidence=receipt.confidence,
            source_event_ids=receipt.source_event_ids,
            source_span_ids=receipt.source_event_ids,
            session_id=None,
            attributes={
                "global": True,
                "source_scope_id": scope_id,
                "receipt_status": receipt.status,
            },
            valid=True,
            publishable=True,
            created_at=utc_now(),
            metadata={"adapter": adapter.name},
        )
        self.storage.append_deltas([receipt_delta])
        self.storage.append_edges(
            [
                DependencyEdge(
                    edge_id=new_id("edge"),
                    task_id=self.task_id,
                    source_delta_id=receipt_delta.delta_id,
                    target_delta_id=source_delta_id,
                    relation="derived_from",
                    rule="committed decision receipt is grounded in visible prior-scope evidence",
                    confidence=receipt.confidence,
                    metadata={
                        "source_scope_id": scope_id,
                        "cross_scope": True,
                    },
                )
                for source_delta_id in source_delta_ids
            ]
        )
        self._materialize_state()
        self.metrics.decision_receipts_emitted += 1
        if receipt.status != "committed":
            self.metrics.provisional_receipts_emitted += 1

    def _materialize_state(self) -> None:
        states = materialize_state(self.storage.load_deltas(), self.storage.load_edges())
        self.storage.save_materialized_state(states)

    def _should_extract(self, parsed_event: dict[str, Any]) -> bool:
        every = int(self.config["write_time"].get("extract_every_n_events", 4))
        if every > 0 and len(self.recent_events) >= every:
            return True
        if self.config["write_time"].get("extract_on_boundary_signal", True) and parsed_event.get("boundary_signal"):
            return True
        if self.config["write_time"].get("extract_on_success_or_failure_signal", True) and (
            parsed_event.get("is_success") or parsed_event.get("is_failure")
        ):
            return True
        return False

    def _parse_state_need(self, question: str, schema_card: Any) -> dict[str, Any]:
        lowered = question.lower()
        needed_variables = []
        for variable in ["decision", "constraint", "dependency", "score", "attribute", "id", "status", "evidence"]:
            if variable in lowered or variable in {"decision", "constraint", "evidence"}:
                needed_variables.append(variable)
        if "avoid" in lowered or "failed" in lowered or "invalid" in lowered or "rejected" in lowered:
            negative = ["visible rejected or failed states"]
        else:
            negative = []
        prior_dependency = bool(
            any(
                marker in lowered
                for marker in [
                    "previous",
                    "prior",
                    "earlier",
                    "history",
                    "depends on",
                    "based on",
                    "compatible with",
                ]
            )
        )
        if prior_dependency and "dependency" not in needed_variables:
            needed_variables.append("dependency")
        subjects = []
        if schema_card is not None:
            for example in schema_card.examples_from_trace[:8]:
                value = example.get("text") if isinstance(example, dict) else str(example)
                if value and str(value).lower() in lowered:
                    subjects.append(str(value))
        return {
            "question": question,
            "needed_subjects": subjects,
            "needed_variables": needed_variables,
            "needed_constraints": ["active constraints"] if "constraint" in needed_variables else [],
            "needed_dependencies": ["prior committed decision receipts and dependency ancestors"]
            if "dependency" in needed_variables
            else [],
            "needed_negative_memory": negative,
            "needed_evidence": ["provenance for important states"],
            "missing_field_risks": [],
            "expected_final_action_type": "unknown",
            "priority": {
                "must_have": ["final states", "active constraints", "relevant rejected or failed states"],
                "nice_to_have": ["source evidence", "recent raw events"],
            },
            "rationale": "Generic need inferred from visible subtask text and current schema.",
        }

    def _parse_decision_signature(self, question: str, contract: Any, schema_card: Any) -> DecisionSignature:
        if self._ablation("disable_decision_signature"):
            state_need = self._parse_state_need(question, schema_card)
            return DecisionSignature(
                decision_id=new_id("decision"),
                task_id=self.task_id,
                current_instruction=question,
                expected_action_type=state_need.get("expected_final_action_type", "unknown"),
                irreversible_risk="none",
                needed_contract_rules=[],
                needed_subjects=state_need["needed_subjects"],
                needed_variables=state_need["needed_variables"],
                needed_final_states=["final states"],
                needed_constraints=state_need["needed_constraints"],
                needed_dependencies=state_need["needed_dependencies"],
                needed_negative_memory=state_need["needed_negative_memory"],
                needed_evidence=state_need["needed_evidence"],
                missing_field_risks=[],
                must_have=state_need["priority"]["must_have"],
                nice_to_have=state_need["priority"]["nice_to_have"],
                rationale=state_need["rationale"],
            )
        lowered = question.lower()
        expected_action = "final_or_irreversible" if is_irreversible_or_final(question, contract) else "unknown"
        risk = "high" if is_irreversible_or_final(question, contract) else ("medium" if any(word in lowered for word in ["choose", "select", "submit"]) else "none")
        rules = relevant_contract_rules(contract, question, limit=int(self.config["read_time"].get("max_contract_rules", 8)))
        state_need = self._parse_state_need(question, schema_card)
        return DecisionSignature(
            decision_id=new_id("decision"),
            task_id=self.task_id,
            current_instruction=question,
            expected_action_type=expected_action,
            irreversible_risk=risk,
            needed_contract_rules=[str(rule.get("rule") or rule.get("constraint") or rule.get("condition") or rule.get("pattern") or rule) for rule in rules],
            needed_subjects=state_need["needed_subjects"],
            needed_variables=state_need["needed_variables"],
            needed_final_states=["final states"],
            needed_constraints=state_need["needed_constraints"] or ["active contract and ledger constraints"],
            needed_dependencies=state_need["needed_dependencies"],
            needed_negative_memory=state_need["needed_negative_memory"],
            needed_evidence=state_need["needed_evidence"],
            missing_field_risks=[],
            must_have=["relevant public task rules", "final states", "active constraints", "evidence pointers"],
            nice_to_have=["recent raw events", "negative memory", "dependency ancestors"],
            rationale="Generic decision signature inferred from visible subtask text, contract, and schema.",
        )

    def _format_events(self, events: list[Event]) -> str:
        return "\n".join(f"{event.event_id} step={event.step_id}: {event.raw_chunk}" for event in events)

    def _existing_delta_key_map(
        self,
        exclude_delta_ids: set[str],
    ) -> dict[tuple[str | None, str, str], str]:
        mapping: dict[tuple[str | None, str, str], str] = {}
        for delta in self.storage.load_deltas():
            if delta.delta_id in exclude_delta_ids or not delta.variable:
                continue
            mapping[(delta.session_id, delta.subject.lower(), delta.variable.lower())] = delta.delta_id
        return mapping

    def _drop_unchanged_deltas(self, deltas: list[StateDelta]) -> list[StateDelta]:
        existing = {
            (
                state.attributes.get("scope_id"),
                state.subject.lower(),
                str(state.variable or "").lower(),
            ): state
            for state in self.storage.load_materialized_state()
        }
        keep: list[StateDelta] = []
        state_changing_ops = {"FINALIZE", "REJECT", "SUPERSEDE", "INVALIDATE", "BACKFILL"}
        for delta in deltas:
            if delta.op in state_changing_ops:
                keep.append(delta)
                continue
            key = (delta.session_id, delta.subject.lower(), str(delta.variable or "").lower())
            current = existing.get(key)
            if (
                current is not None
                and current.status == delta.status
                and current.valid == delta.valid
                and current.publishable == delta.publishable
                and current.value == delta.value
            ):
                continue
            keep.append(delta)
        return keep

    def _lane_config(self) -> dict[str, Any]:
        config = {**self.config.get("state_lanes", {}), **self.config.get("read_time", {})}
        if self._ablation("disable_task_contract"):
            config["enable_contract_lane"] = False
        if self._ablation("disable_constraints"):
            config["enable_constraint_lane"] = False
        if self._ablation("disable_dependency_graph"):
            config["enable_dependency_lane"] = False
        if self._ablation("disable_negative_memory"):
            config["enable_negative_lane"] = False
        if self._ablation("disable_repair_barrier"):
            config["enable_repair_lane"] = False
        return config

    def _build_path_summary(self, question: str) -> str:
        checkpoints = self.storage.load_checkpoints()
        events = self.storage.load_events()
        failed = [delta.value for delta in self.storage.load_deltas() if delta.status in {"rejected", "failed"}]
        return build_path_summary_packet(
            question=question,
            path_summaries=[checkpoint.summary for checkpoint in checkpoints],
            recent_events=[event.raw_chunk for event in events[-5:]],
            failed_hints=[str(item) for item in failed],
        )

    def _ablation(self, key: str) -> bool:
        return bool(self.config.get("ablation", {}).get(key, False))


def _adapter_dependency_gaps(task: TaskIR, states: list[Any]) -> list[dict[str, Any]]:
    if not task.dependencies:
        return []
    receipts = []
    for state in states:
        if getattr(state, "variable", None) != "decision_receipt":
            continue
        value = getattr(state, "value", None)
        if isinstance(value, dict):
            receipts.append(value)
    step_index = task.metadata.get("step_index")
    expected_scope = (
        f"product-{int(step_index) - 1}"
        if isinstance(step_index, int) and step_index > 1
        else None
    )
    if expected_scope is not None:
        receipts = [row for row in receipts if str(row.get("scope_id") or "") == expected_scope]
    usable = [
        row
        for row in receipts
        if isinstance(row.get("committed_decision"), dict)
        and (
            row["committed_decision"].get("semantic_option")
            or row["committed_decision"].get("entity")
        )
    ]
    if usable:
        return []
    return [
        {
            "subject": "previous_scope",
            "variable": "prior_decision_summary",
            "why_needed": (
                "The public task declares a prior-decision dependency, but no committed or "
                "provisional scope summary is available."
            ),
            "suggested_trace_scope": expected_scope or "previous_scope",
            "priority": "high",
        }
    ]


def _adapter_action_directive(
    *,
    adapter: TaskAdapter,
    task: TaskIR,
    question: str,
    materialized_state: list[Any],
    progress: Any,
) -> ActionDirective | None:
    builder = getattr(adapter, "build_action_directive", None)
    if not callable(builder):
        return None
    directive = builder(
        task=task,
        question=question,
        materialized_state=materialized_state,
        progress=progress,
    )
    return directive if isinstance(directive, ActionDirective) else None


def _active_execution_path(
    events: list[Event],
    *,
    progress: Any,
    limit: int,
) -> list[str]:
    rows = []
    for event in events[-max(0, limit) :]:
        action = " ".join(str(event.action or "observe").split())
        if len(action) > 160:
            action = action[:157].rstrip() + "..."
        if event.is_success:
            outcome = "success"
        elif event.is_failure:
            outcome = "failure"
        elif event.is_final_action:
            outcome = "final-pending"
        else:
            outcome = "continued"
        progress_status = "stale" if progress.is_stale(event.action) else "active"
        rows.append(f"step {event.step_id}: {action} -> {outcome}, path={progress_status}")
    return rows


def _decision_state_ledger(lines: list[str]) -> list[str]:
    normalized = []
    for line in lines:
        if line.startswith("progress:"):
            continue
        compact = re.sub(r"\b[A-Z0-9]{10}\b", "<entity>", line)
        compact = compact.replace("historical evidence (not currently clickable)", "<entity>")
        compact = compact.replace(", selected", "")
        normalized.append(compact)
    return normalized


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _fit_packet_with_suffix(packet: str, suffix: str, *, max_chars: int) -> str:
    separator = "\n\n"
    if len(packet) + len(separator) + len(suffix) <= max_chars:
        return packet + separator + suffix
    available = max_chars - len(separator) - len(suffix) - 42
    if available <= 0:
        return suffix[:max_chars]
    clipped = packet[:available].rstrip() + "\n[truncated for total packet budget]"
    return clipped + separator + suffix
