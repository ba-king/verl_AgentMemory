from examples.csl_memory_agent.csl_memory.csl_schema import (
    DependencyEdge,
    Event,
    StateDelta,
    TaskSchemaCard,
    new_id,
    utc_now,
)
from examples.csl_memory_agent.csl_memory.csl_storage import JSONLCSLStorage


def test_jsonl_storage_round_trips_required_objects(tmp_path):
    store = JSONLCSLStorage(tmp_path)
    event = Event(
        event_id=new_id("event"),
        task_id="task1",
        subtask_id=None,
        step_id=1,
        raw_chunk="Action: inspect. Observation: value=1",
        action="inspect",
        observation="value=1",
        reward=None,
        timestamp=utc_now(),
    )
    schema = TaskSchemaCard(
        schema_id=new_id("schema"),
        task_id="task1",
        version=1,
        task_family_description="visible state schema",
        entity_types=[],
        state_variables=[],
        constraint_types=[],
        dependency_types=[],
        action_outcome_types=[],
        status_rules={},
        provenance_requirements=["source_event_ids"],
        update_rules=[],
        examples_from_trace=[],
        confidence=0.6,
        created_at=utc_now(),
    )
    delta = StateDelta(
        delta_id=new_id("delta"),
        task_id="task1",
        checkpoint_id=None,
        schema_id=schema.schema_id,
        op="SET",
        subject="subject",
        variable="value",
        value=1,
        attributes={},
        status="candidate",
        confidence=0.7,
        source_event_ids=[event.event_id],
        source_span_ids=[f"{event.event_id}:raw_chunk"],
        created_at=utc_now(),
    )
    edge = DependencyEdge(
        edge_id=new_id("edge"),
        task_id="task1",
        source_delta_id=delta.delta_id,
        target_delta_id=delta.delta_id,
        relation="evidence_for",
        rule="self evidence",
    )

    store.append_event(event)
    store.upsert_schema_card(schema)
    store.append_deltas([delta])
    store.append_edges([edge])

    assert store.load_events()[0].raw_chunk == event.raw_chunk
    assert store.load_schema_card().schema_id == schema.schema_id
    assert store.load_deltas()[0].value == 1
    assert store.load_edges()[0].relation == "evidence_for"
