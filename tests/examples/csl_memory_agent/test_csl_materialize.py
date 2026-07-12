from examples.csl_memory_agent.csl_memory.csl_lanes import derive_state_lanes, materialize_state
from examples.csl_memory_agent.csl_memory.csl_schema import DependencyEdge, StateDelta, new_id, utc_now


def _delta(op, subject, variable, value, status="candidate"):
    return StateDelta(
        delta_id=new_id("delta"),
        task_id="task1",
        checkpoint_id=None,
        schema_id=None,
        op=op,
        subject=subject,
        variable=variable,
        value=value,
        attributes={"short_evidence": str(value), "confidence": 0.8},
        status=status,
        confidence=0.8,
        source_event_ids=["event1"],
        source_span_ids=["event1:raw_chunk"],
        created_at=utc_now(),
    )


def test_materialize_keeps_latest_final_and_negative_lane():
    old = _delta("SET", "option A", "score", 1.0, "candidate")
    new = _delta("SET", "option A", "score", 2.0, "final")
    reject = _delta("REJECT", "option B", "negative_memory", "failed validation", "failed")
    edges = [
        DependencyEdge(
            edge_id=new_id("edge"),
            task_id="task1",
            source_delta_id=new.delta_id,
            target_delta_id=old.delta_id,
            relation="supersedes",
            rule="new final score",
        )
    ]

    states = materialize_state([old, new, reject], edges)
    assert len(states) == 1
    assert states[0].value == 2.0

    lanes = derive_state_lanes(materialized_state=states, deltas=[old, new, reject], edges=edges, recent_events=[])
    assert lanes["negative"].entries
    assert lanes["decision"].entries
