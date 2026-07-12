from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DARTCSLMetrics:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    write_llm_tokens: int = 0
    read_llm_tokens: int = 0
    total_tokens: int = 0
    state_packet_tokens: int = 0
    contract_tokens_injected: int = 0
    num_active_deltas: int = 0
    num_total_deltas: int = 0
    num_retrieved_constraints: int = 0
    num_retrieved_dependencies: int = 0
    num_negative_states_retrieved: int = 0
    missing_field_count: int = 0
    backfill_count: int = 0
    stale_or_blocked_field_count: int = 0
    action_guard_calls: int = 0
    action_guard_blocks: int = 0
    guard_audit_cache_hits: int = 0
    guard_audit_cache_misses: int = 0
    guard_llm_calls_avoided: int = 0
    guard_audit_records: int = 0
    selection_audit_calls: int = 0
    selection_audit_deferred: int = 0
    certificate_build_failures: int = 0
    certificate_uncertainty_count: int = 0
    scope_switches: int = 0
    no_progress_events: int = 0
    stale_actions_detected: int = 0
    transition_validation_calls: int = 0
    transition_validation_blocks: int = 0
    transition_action_set_blocks: int = 0
    transition_forbidden_action_blocks: int = 0
    transition_repeat_budget_blocks: int = 0
    transition_validation_latency_ms: float = 0.0
    decision_receipts_emitted: int = 0
    provisional_receipts_emitted: int = 0
    cross_scope_receipts_retrieved: int = 0
    memory_llm_calls: int = 0
    memory_llm_prompt_tokens: int = 0
    memory_llm_completion_tokens: int = 0
    memory_llm_latency_ms: float = 0.0
    add_chunk_latency_ms: float = 0.0
    wrap_user_prompt_latency_ms: float = 0.0
    guard_action_latency_ms: float = 0.0

    def update_tokens_from_packet(self, packet: str) -> None:
        self.state_packet_tokens = max(1, len(packet) // 4) if packet else 0
        self.total_tokens = self.prompt_tokens + self.generation_tokens + self.write_llm_tokens + self.read_llm_tokens

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)
