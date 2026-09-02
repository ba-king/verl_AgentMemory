"""Route one validated MemoryAction to the Store or Retriever."""

from __future__ import annotations

from typing import cast

from examples.memory_agent_grpo.memory_agent.action.protocol import (
    AddArguments,
    DeleteArguments,
    MemoryAction,
    MemoryActionName,
    RetrieveArguments,
    UpdateArguments,
)
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState
from examples.memory_agent_grpo.memory_agent.bank.store import MemoryBankStore, StoreMutationStatus
from examples.memory_agent_grpo.memory_agent.retrieval.bm25 import BM25Retriever

from .outcome import ExecutionCode, ExecutionResult, ExecutionStatus
from .validation import (
    ExecutorValidationDisposition,
    validate_action_for_execution,
)


class MemoryActionExecutor:
    """Execute one typed action against fixed deterministic components."""

    def __init__(
        self,
        *,
        store: MemoryBankStore | None = None,
        retriever: BM25Retriever | None = None,
    ) -> None:
        self._store = MemoryBankStore() if store is None else store
        self._retriever = BM25Retriever() if retriever is None else retriever

    @property
    def state(self) -> MemoryBankState:
        return self._store.state

    def execute(self, action: object) -> ExecutionResult:
        """Preflight and execute one action; never parse raw model output."""

        validation = validate_action_for_execution(action, state=self._store.state)
        action_name = _safe_action_name(action)
        if validation.disposition is ExecutorValidationDisposition.REJECTED:
            return ExecutionResult(
                status=ExecutionStatus.REJECTED,
                code=validation.code or ExecutionCode.INVALID_ACTION,
                action=action_name,
                error_message=validation.error_message,
            )
        if validation.disposition is ExecutorValidationDisposition.NO_CHANGE:
            return ExecutionResult(
                status=ExecutionStatus.NO_CHANGE,
                code=validation.code or ExecutionCode.INVALID_ACTION,
                action=action_name,
                record=validation.record,
            )

        typed_action = cast(MemoryAction, action)
        if typed_action.action is MemoryActionName.RETRIEVE:
            arguments = cast(RetrieveArguments, typed_action.arguments)
            retrieval = self._retriever.retrieve(arguments.query, state=self._store.state)
            return ExecutionResult(
                status=ExecutionStatus.SUCCEEDED,
                code=ExecutionCode.RETRIEVED,
                action=typed_action.action,
                retrieval=retrieval,
            )
        if typed_action.action is MemoryActionName.ADD:
            arguments = cast(AddArguments, typed_action.arguments)
            mutation = self._store.add(arguments.content)
            status = (
                ExecutionStatus.SUCCEEDED
                if mutation.status is StoreMutationStatus.APPLIED
                else ExecutionStatus.NO_CHANGE
            )
            code = (
                ExecutionCode.ADDED
                if mutation.status is StoreMutationStatus.APPLIED
                else ExecutionCode.EXACT_DUPLICATE_ADD
            )
            return ExecutionResult(
                status=status,
                code=code,
                action=typed_action.action,
                record=mutation.record,
            )
        if typed_action.action is MemoryActionName.UPDATE:
            arguments = cast(UpdateArguments, typed_action.arguments)
            mutation = self._store.update(arguments.memory_id, arguments.content)
            status = (
                ExecutionStatus.SUCCEEDED
                if mutation.status is StoreMutationStatus.APPLIED
                else ExecutionStatus.NO_CHANGE
            )
            code = (
                ExecutionCode.UPDATED
                if mutation.status is StoreMutationStatus.APPLIED
                else ExecutionCode.UNCHANGED_UPDATE
            )
            return ExecutionResult(
                status=status,
                code=code,
                action=typed_action.action,
                record=mutation.record,
            )

        arguments = cast(DeleteArguments, typed_action.arguments)
        mutation = self._store.delete(arguments.memory_id)
        return ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            code=ExecutionCode.DELETED,
            action=typed_action.action,
            record=mutation.record,
        )


def _safe_action_name(action: object) -> MemoryActionName | None:
    if isinstance(action, MemoryAction) and isinstance(action.action, MemoryActionName):
        return action.action
    return None


__all__ = ["MemoryActionExecutor"]
