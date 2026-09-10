import json
from dataclasses import replace
from pathlib import Path

import pytest

from examples.memory_agent_grpo.episode.runtime.memory_decision import MemoryDecisionStepError, execute_memory_decision
from examples.memory_agent_grpo.episode.validation import validate_memory_decision_trace
from examples.memory_agent_grpo.memory_agent.action.protocol import MemoryDecisionPhase
from examples.memory_agent_grpo.memory_agent.bank.bank_state import MemoryBankState, MemoryRecord
from examples.memory_agent_grpo.memory_agent.bank.store import MemoryBankStore
from examples.memory_agent_grpo.memory_agent.decision.action_gateway import ActionToExecutorGateway
from examples.memory_agent_grpo.memory_agent.decision.context import MemoryDecisionContext, TaskResult
from examples.memory_agent_grpo.memory_agent.executor.engine import MemoryActionExecutor
from examples.memory_agent_grpo.tests.episode.runtime._support import ScriptedMemoryClient, add_action


def _run(*, enabled=True, observer=None):
    raw = add_action("EXACT_MEMORY")[:-1]
    client = ScriptedMemoryClient(outcomes=[raw])
    gateway = ActionToExecutorGateway()
    context = MemoryDecisionContext(
        phase=MemoryDecisionPhase.POST_TASK, task_input="task", bank_record_count=0,
        task_result=TaskResult("answer"),
    )
    return client, gateway, dict(
        memory_client=client, gateway=gateway, context=context,
        repair_add_outer_brace=enabled, trace_observer=observer,
    )


def test_runtime_repairs_once_and_preserves_raw_evidence():
    observed = []
    client, gateway, kwargs = _run(observer=observed.append)
    trace = execute_memory_decision(**kwargs)
    assert len(client.calls) == len(observed) == 1
    assert gateway.state.record_count == 1
    assert gateway.state.records[0].content == "EXACT_MEMORY"
    assert trace.raw_output == add_action("EXACT_MEMORY")[:-1]
    assert trace.effective_output == trace.raw_output + "}"
    assert trace.to_dict()["output_repair"]["original_validation"]["error_code"] == "INVALID_JSON"
    assert validate_memory_decision_trace(trace).is_valid


def test_strict_mode_still_rejects_without_writing():
    observed = []
    client, gateway, kwargs = _run(enabled=False, observer=observed.append)
    with pytest.raises(MemoryDecisionStepError):
        execute_memory_decision(**kwargs)
    assert len(client.calls) == 1
    assert gateway.state.record_count == 0
    assert observed[0].output_repair is None


@pytest.mark.parametrize("tamper", ["raw", "effective", "rule", "validation", "missing", "bank"])
def test_forged_repair_or_bank_transition_is_rejected(tamper):
    _, _, kwargs = _run()
    trace = execute_memory_decision(**kwargs)
    repair = trace.output_repair
    if tamper == "raw":
        forged = replace(trace, raw_output=add_action("OTHER")[:-1])
    elif tamper == "effective":
        forged = replace(trace, output_repair=replace(repair, repaired_output=add_action("OTHER")))
    elif tamper == "rule":
        forged = replace(trace, output_repair=replace(repair, rule="UNRESTRICTED_REPAIR"))
    elif tamper == "validation":
        forged = replace(trace, output_repair=replace(repair, original_validation=trace.gateway_result.validation))
    elif tamper == "missing":
        forged = replace(trace, output_repair=None)
    else:
        forged = replace(trace, bank_after=trace.bank_before)
    assert not validate_memory_decision_trace(forged).is_valid


def test_observer_failure_does_not_repeat_the_repaired_write():
    def fail(trace):
        raise RuntimeError("observer failed")
    client, gateway, kwargs = _run(observer=fail)
    with pytest.raises(MemoryDecisionStepError):
        execute_memory_decision(**kwargs)
    assert len(client.calls) == gateway.state.record_count == 1


CASES = json.loads((Path(__file__).parents[2] / "memory_agent/action/fixtures/outer_brace_failures.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_historical_output_executes_once_against_recorded_bank(case):
    before = MemoryBankState(
        records=tuple(MemoryRecord(**record) for record in case["bank_before"]["records"]),
        next_id=case["bank_before"]["next_id"],
    )
    gateway = ActionToExecutorGateway(executor=MemoryActionExecutor(store=MemoryBankStore(before)))
    client = ScriptedMemoryClient(outcomes=[case["raw_output"]])
    trace = execute_memory_decision(
        memory_client=client, gateway=gateway,
        repair_add_outer_brace=True,
        context=MemoryDecisionContext(
            phase=MemoryDecisionPhase.POST_TASK, task_input="Offline fixed-output replay",
            bank_record_count=before.record_count, task_result=TaskResult("Recorded task completed"),
        ),
    )
    assert len(client.calls) == 1
    assert trace.raw_output == case["raw_output"]
    assert trace.output_repair is not None
    assert gateway.state.record_count == before.record_count + 1
    assert gateway.state.records[:-1] == before.records
    assert gateway.state.records[-1].content == case["expected_content"]
    assert validate_memory_decision_trace(trace).is_valid
