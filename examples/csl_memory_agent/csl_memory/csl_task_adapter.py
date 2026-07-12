from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ObjectiveSpec:
    metric: str = "satisfaction"
    direction: str = "satisfy"


@dataclass(frozen=True)
class ConstraintAtom:
    field: str
    operator: str
    value: Any
    source_text: str
    hard: bool = True


@dataclass(frozen=True)
class DependencySpec:
    source: str
    relation: str
    target: str
    source_text: str


@dataclass(frozen=True)
class EvidenceRequirement:
    field: str
    source_level: str
    confidence: float = 1.0
    freshness: str = "current"


@dataclass(frozen=True)
class RecoveryAction:
    action: str
    operator: str
    reason: str
    expected_evidence: tuple[str, ...] = ()
    match_mode: str = "exact"
    semantic_target: str | None = None
    required_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionDirective:
    """Task-adapter output consumed by the generic transition validator."""

    action: str
    reason: str
    mode: str = "advisory"
    obligation_id: str | None = None
    match_mode: str = "exact"
    semantic_target: str | None = None
    required_terms: tuple[str, ...] = ()
    required_state_delta: tuple[str, ...] = ()
    recovery_actions: tuple[RecoveryAction, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    lease_id: str | None = None
    max_consecutive_action_repeats: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_required(self) -> bool:
        return self.mode in {"required", "required_state_change"}

    @property
    def display_text(self) -> str:
        if not self.action:
            return self.reason
        return f"{self.action} ({self.reason})" if self.reason else self.action

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskIR:
    adapter_name: str
    goal_id: str
    options: list[str] = field(default_factory=list)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    constraints: list[ConstraintAtom] = field(default_factory=list)
    dependencies: list[DependencySpec] = field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = field(default_factory=list)
    commit_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionReceipt:
    scope_id: str
    status: str
    committed_decision: dict[str, Any]
    satisfied_constraints: list[str]
    evidence_state_ids: list[str]
    source_event_ids: list[str]
    confidence: float
    state_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgressSignal:
    scope_id: str
    action: str
    observation_fingerprint: str
    repeat_count: int
    evidence_novel: bool
    stale: bool
    reason: str | None = None


@dataclass(frozen=True)
class ProgressContext:
    scope_id: str
    stale_actions: frozenset[str] = frozenset()
    signals: tuple[ProgressSignal, ...] = ()

    def is_stale(self, action: str | None) -> bool:
        return canonical_action(action) in self.stale_actions

    def consecutive_action_count(self, action: str | None) -> int:
        target = canonical_action(action)
        if not target:
            return 0
        count = 0
        for signal in reversed(self.signals):
            if signal.action != target:
                break
            count += 1
        return count

    @property
    def latest_stale_signal(self) -> ProgressSignal | None:
        return next((signal for signal in reversed(self.signals) if signal.stale), None)


class TaskAdapter(Protocol):
    name: str

    def supports(self, question: str) -> bool: ...

    def compile_task(self, question: str) -> TaskIR: ...

    def build_candidate_ledger(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        progress: ProgressContext,
    ) -> list[str]: ...

    def recommend_next_action(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        progress: ProgressContext,
    ) -> str | None: ...

    def build_action_directive(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        progress: ProgressContext,
    ) -> ActionDirective | None: ...

    def build_decision_certificate(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        selected_option_id: str | None = None,
    ) -> dict[str, Any] | None: ...

    def build_decision_receipt(
        self,
        *,
        scope_id: str,
        materialized_state: list[Any],
        events: list[Any],
    ) -> DecisionReceipt | None: ...


class NoProgressTracker:
    """Detect repeated zero-gain transitions without knowing the task domain."""

    def __init__(self, repeat_limit: int = 2, history_limit: int = 32, cycle_window: int = 6):
        self.repeat_limit = max(2, int(repeat_limit))
        self.history_limit = max(4, int(history_limit))
        self.cycle_window = max(2, int(cycle_window))
        self._transition_counts: dict[str, dict[tuple[str, str], int]] = {}
        self._evidence_seen: dict[str, set[str]] = {}
        self._signals: dict[str, list[ProgressSignal]] = {}
        self._stale_actions: dict[str, set[str]] = {}
        self._observation_history: dict[str, list[str]] = {}
        self._decision_state_counts: dict[str, dict[str, int]] = {}
        self._latest_actions: dict[str, str] = {}

    def observe(
        self,
        *,
        scope_id: str | None,
        action: str | None,
        observation: str | None,
        evidence_items: list[str] | tuple[str, ...] | None = None,
    ) -> ProgressSignal | None:
        normalized_action = canonical_action(action)
        if not normalized_action or not observation:
            return None
        scope = scope_id or "global"
        self._latest_actions[scope] = normalized_action
        observation_hash = semantic_fingerprint(observation)
        transition_key = (normalized_action, observation_hash)
        counts = self._transition_counts.setdefault(scope, {})
        counts[transition_key] = counts.get(transition_key, 0) + 1

        evidence_payload = [str(item) for item in evidence_items or () if str(item).strip()]
        evidence_hash = semantic_fingerprint(json.dumps(sorted(evidence_payload), ensure_ascii=True))
        seen_evidence = self._evidence_seen.setdefault(scope, set())
        evidence_novel = bool(evidence_payload and evidence_hash not in seen_evidence)
        if evidence_payload:
            seen_evidence.add(evidence_hash)

        repeat_count = counts[transition_key]
        observation_history = self._observation_history.setdefault(scope, [])
        repeated_state = observation_hash in observation_history[-self.cycle_window :]
        observation_history.append(observation_hash)
        del observation_history[:-self.history_limit]
        stale = (
            repeat_count >= self.repeat_limit
            or repeated_state
            or _has_repeating_suffix(observation_history, max_period=self.cycle_window)
        ) and not evidence_novel
        reason = None
        if stale:
            reason = (
                "the action returned to an equivalent decision state without adding evidence; "
                "choose a different legal recovery action"
            )
            self._stale_actions.setdefault(scope, set()).add(normalized_action)

        signal = ProgressSignal(
            scope_id=scope,
            action=normalized_action,
            observation_fingerprint=observation_hash,
            repeat_count=repeat_count,
            evidence_novel=evidence_novel,
            stale=stale,
            reason=reason,
        )
        signals = self._signals.setdefault(scope, [])
        signals.append(signal)
        del signals[:-self.history_limit]
        return signal

    def observe_decision_state(
        self,
        *,
        scope_id: str | None,
        decision_state: Any,
    ) -> ProgressSignal | None:
        """Mark the latest action stale when the decision-relevant state stops changing."""
        scope = scope_id or "global"
        action = self._latest_actions.get(scope)
        if not action:
            return None
        state_hash = semantic_fingerprint(
            json.dumps(decision_state, ensure_ascii=True, sort_keys=True, default=str)
        )
        counts = self._decision_state_counts.setdefault(scope, {})
        counts[state_hash] = counts.get(state_hash, 0) + 1
        repeat_count = counts[state_hash]
        if repeat_count != self.repeat_limit:
            return None
        self._stale_actions.setdefault(scope, set()).add(action)
        signal = ProgressSignal(
            scope_id=scope,
            action=action,
            observation_fingerprint=state_hash,
            repeat_count=repeat_count,
            evidence_novel=False,
            stale=True,
            reason=(
                "the candidate/constraint state did not improve after the latest action; "
                "do not continue the same evidence path"
            ),
        )
        signals = self._signals.setdefault(scope, [])
        signals.append(signal)
        del signals[:-self.history_limit]
        return signal

    def context(self, scope_id: str | None) -> ProgressContext:
        scope = scope_id or "global"
        return ProgressContext(
            scope_id=scope,
            stale_actions=frozenset(self._stale_actions.get(scope, set())),
            signals=tuple(self._signals.get(scope, ())),
        )

    def reset(self) -> None:
        self._transition_counts.clear()
        self._evidence_seen.clear()
        self._signals.clear()
        self._stale_actions.clear()
        self._observation_history.clear()
        self._decision_state_counts.clear()
        self._latest_actions.clear()


class GenericTaskAdapter:
    name = "generic"

    _COMMIT_VARIABLES = {
        "answer",
        "answer_status",
        "booking_status",
        "decision",
        "purchase_status",
        "submission_status",
    }

    def supports(self, question: str) -> bool:
        del question
        return True

    def compile_task(self, question: str) -> TaskIR:
        objective = ObjectiveSpec()
        lowered = question.lower()
        if re.search(r"\b(?:maximize|highest|largest|most)\b", lowered):
            objective = ObjectiveSpec(metric="visible objective", direction="maximize")
        elif re.search(r"\b(?:minimize|lowest|smallest|least|cheapest)\b", lowered):
            objective = ObjectiveSpec(metric="visible objective", direction="minimize")
        return TaskIR(
            adapter_name=self.name,
            goal_id=semantic_fingerprint(question)[:16],
            objective=objective,
            metadata={"source": "public_task_text"},
        )

    def build_candidate_ledger(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        progress: ProgressContext,
    ) -> list[str]:
        del task, question, materialized_state, progress
        return []

    def recommend_next_action(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        progress: ProgressContext,
    ) -> str | None:
        del task, question, materialized_state
        stale = progress.latest_stale_signal
        if stale is None:
            return None
        return f"Do not repeat {stale.action}; choose a legal action that can add new evidence."

    def build_action_directive(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        progress: ProgressContext,
    ) -> ActionDirective | None:
        guidance = self.recommend_next_action(
            task=task,
            question=question,
            materialized_state=materialized_state,
            progress=progress,
        )
        if guidance is None:
            return None
        return ActionDirective(action="", reason=guidance, mode="advisory")

    def build_decision_certificate(
        self,
        *,
        task: TaskIR,
        question: str,
        materialized_state: list[Any],
        selected_option_id: str | None = None,
    ) -> dict[str, Any] | None:
        del task, question, materialized_state, selected_option_id
        return None

    def build_decision_receipt(
        self,
        *,
        scope_id: str,
        materialized_state: list[Any],
        events: list[Any],
    ) -> DecisionReceipt | None:
        states = [_as_dict(state) for state in materialized_state]
        event_rows = [_as_dict(event) for event in events]
        has_commit = any(
            row.get("is_success") is True
            or (
                str(row.get("variable") or "").lower() in self._COMMIT_VARIABLES
                and str(row.get("status") or "").lower() == "final"
            )
            for row in [*states, *event_rows]
        )
        has_failure = any(row.get("is_failure") is True for row in event_rows)
        committed_rows = [
            row
            for row in states
            if str(row.get("variable") or "").lower() != "decision_receipt"
            and (
                str(row.get("status") or "").lower() == "final"
                or str(row.get("variable") or "").lower() in self._COMMIT_VARIABLES
            )
        ]
        candidate_rows = [
            row
            for row in states
            if str(row.get("status") or "").lower() in {"candidate", "provisional"}
            and str(row.get("variable") or "").lower()
            in {"answer", "decision", "selected_option_id", "selection"}
        ]
        receipt_rows = committed_rows if has_commit and committed_rows else candidate_rows[-4:]
        committed = [
            {
                "subject": row.get("subject"),
                "variable": row.get("variable"),
                "value": row.get("value"),
            }
            for row in receipt_rows
        ]
        if not committed:
            return None
        status = "committed" if has_commit and committed_rows else "failed" if has_failure else "provisional"
        confidence = 0.95 if status == "committed" else 0.45 if status == "failed" else 0.6
        return DecisionReceipt(
            scope_id=scope_id,
            status=status,
            committed_decision={"states": committed},
            satisfied_constraints=[],
            evidence_state_ids=sorted(
                str(row.get("state_id")) for row in receipt_rows if row.get("state_id")
            ),
            source_event_ids=sorted(
                str(row.get("event_id")) for row in event_rows if row.get("event_id")
            ),
            confidence=confidence,
        )


class TaskAdapterRegistry:
    def __init__(self, adapters: list[TaskAdapter] | None = None):
        self._adapters = list(adapters or [])
        self._fallback = GenericTaskAdapter()

    def register(self, adapter: TaskAdapter, *, priority: bool = False) -> None:
        if priority:
            self._adapters.insert(0, adapter)
        else:
            self._adapters.append(adapter)

    def resolve(self, question: str) -> TaskAdapter:
        return next((adapter for adapter in self._adapters if adapter.supports(question)), self._fallback)


def canonical_action(action: str | None) -> str:
    normalized = str(action or "").strip().lower()
    normalized = re.sub(r"[,.;:]+", " ", normalized)
    return " ".join(normalized.split())


def is_declared_commit_action(task: TaskIR, action: str | None) -> bool:
    """Match a proposed action against commit syntax declared by its adapter."""
    command = str(action or "").split("DecisionCertificate:", 1)[0].strip()
    proposed = canonical_action(command)
    proposed_parts = split_action(command)
    for declared in task.commit_actions:
        expected = str(declared or "").strip()
        if not expected:
            continue
        if proposed == canonical_action(expected):
            return True
        expected_parts = split_action(expected)
        if expected.endswith("[*]") and proposed_parts is not None:
            operator = expected.split("[", 1)[0].strip().lower()
            if proposed_parts[0] == operator:
                return True
        if expected_parts is None and proposed_parts is not None:
            if proposed_parts[0] == expected.lower().rstrip("[]"):
                return True
    return False


def validate_action_directive(
    action: str | None,
    directive: ActionDirective | None,
    *,
    current_lease_id: str | None = None,
    progress: ProgressContext | None = None,
) -> tuple[bool, str | None]:
    """Validate a proposed action without embedding any task-domain vocabulary."""
    if directive is None or not directive.is_required:
        return True, None
    if directive.lease_id and current_lease_id and directive.lease_id != current_lease_id:
        return True, None
    proposed = canonical_action(action)
    if not proposed:
        return False, "the proposed action is empty"

    for forbidden in directive.forbidden_actions:
        if _action_pattern_matches(action, forbidden):
            obligation = directive.obligation_id or "current required evidence"
            return False, f"the action is prohibited while obligation {obligation} is unresolved"

    if progress is not None and progress.is_stale(action):
        return False, "the proposed action is a known zero-progress transition"

    repeat_limit = directive.max_consecutive_action_repeats
    if repeat_limit is not None and progress is not None:
        prior_repeats = progress.consecutive_action_count(action)
        if prior_repeats >= max(1, int(repeat_limit)):
            return (
                False,
                f"the same recovery action has already been used {prior_repeats} consecutive times",
            )

    accepted = [
        RecoveryAction(
            action=directive.action,
            operator=(split_action(directive.action) or ("", ""))[0],
            reason=directive.reason,
            expected_evidence=directive.required_state_delta,
            match_mode=directive.match_mode,
            semantic_target=directive.semantic_target,
            required_terms=directive.required_terms,
        )
    ] if directive.action else []
    accepted.extend(directive.recovery_actions)
    if any(_matches_recovery_action(action, candidate) for candidate in accepted):
        return True, None

    obligation = directive.obligation_id or "current required evidence"
    accepted_text = ", ".join(item.action for item in accepted[:5]) or "a legal evidence-producing action"
    return (
        False,
        f"the action does not satisfy obligation {obligation}; admissible recoveries include {accepted_text}",
    )


def _matches_recovery_action(action: str | None, recovery: RecoveryAction) -> bool:
    if canonical_action(action) == canonical_action(recovery.action):
        return True
    if recovery.match_mode != "semantic_target":
        return _action_pattern_matches(action, recovery.action)
    proposed_parts = split_action(action)
    expected_parts = split_action(recovery.action)
    if proposed_parts is None or expected_parts is None:
        return False
    proposed_operator, proposed_argument = proposed_parts
    expected_operator, expected_argument = expected_parts
    return proposed_operator == expected_operator and _semantic_target_matches(
        proposed_argument,
        recovery.semantic_target or expected_argument,
        required_terms=recovery.required_terms,
    )


def _action_pattern_matches(action: str | None, pattern: str | None) -> bool:
    proposed = canonical_action(action)
    expected = canonical_action(pattern)
    if not proposed or not expected:
        return False
    if proposed == expected:
        return True
    expected_parts = split_action(pattern)
    proposed_parts = split_action(action)
    if expected_parts is None or proposed_parts is None:
        return False
    expected_operator, expected_argument = expected_parts
    proposed_operator, _proposed_argument = proposed_parts
    return expected_operator == proposed_operator and expected_argument.strip() == "*"


def split_action(action: str | None) -> tuple[str, str] | None:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_-]*)\[([^\]]+)\]\s*", str(action or ""))
    if match is None:
        return None
    return match.group(1).lower(), " ".join(match.group(2).strip().split())


def semantic_fingerprint(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _semantic_target_matches(
    proposed: str,
    expected: str,
    *,
    required_terms: tuple[str, ...],
) -> bool:
    proposed_tokens = set(_content_tokens(proposed))
    if not proposed_tokens:
        return False
    required = {token.lower() for token in required_terms if token.strip()}
    if required:
        needed = max(1, (len(required) * 3 + 3) // 4)
        return len(proposed_tokens & required) >= needed
    expected_tokens = set(_content_tokens(expected))
    if not expected_tokens:
        return canonical_action(proposed) == canonical_action(expected)
    needed = max(1, (len(expected_tokens) + 1) // 2)
    return len(proposed_tokens & expected_tokens) >= needed


def _content_tokens(value: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._'+-]*", value)
        if token.lower() not in stopwords
    ]


def _has_repeating_suffix(values: list[str], *, max_period: int) -> bool:
    for period in range(2, min(max_period, len(values) // 2) + 1):
        if values[-period:] == values[-2 * period : -period]:
            return True
    return False


def _as_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {}
