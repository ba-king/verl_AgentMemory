from examples.memory_agent_grpo.memory_agent.action.protocol import (
    AddArguments,
    DeleteArguments,
    MemoryAction,
    MemoryActionName,
    MemoryDecision,
    RetrieveArguments,
    UpdateArguments,
)


def test_executable_action_set_contains_exactly_four_actions() -> None:
    assert [action.value for action in MemoryActionName] == [
        "RETRIEVE",
        "ADD",
        "UPDATE",
        "DELETE",
    ]


def test_absent_action_serializes_as_null() -> None:
    assert MemoryDecision(memory_action=None).to_dict() == {"memory_action": None}


def test_retrieve_action_serialization() -> None:
    decision = MemoryDecision(
        memory_action=MemoryAction(
            action=MemoryActionName.RETRIEVE,
            arguments=RetrieveArguments(query="Kevin hotel price"),
        )
    )

    assert decision.to_dict() == {
        "memory_action": {
            "action": "RETRIEVE",
            "arguments": {"query": "Kevin hotel price"},
        }
    }


def test_add_action_serialization() -> None:
    decision = MemoryDecision(
        memory_action=MemoryAction(
            action=MemoryActionName.ADD,
            arguments=AddArguments(content="Kevin selected Hotel A."),
        )
    )

    assert decision.to_dict() == {
        "memory_action": {
            "action": "ADD",
            "arguments": {"content": "Kevin selected Hotel A."},
        }
    }


def test_update_action_serialization() -> None:
    decision = MemoryDecision(
        memory_action=MemoryAction(
            action=MemoryActionName.UPDATE,
            arguments=UpdateArguments(memory_id="M003", content="New complete content"),
        )
    )

    assert decision.to_dict() == {
        "memory_action": {
            "action": "UPDATE",
            "arguments": {"memory_id": "M003", "content": "New complete content"},
        }
    }


def test_delete_action_serialization() -> None:
    decision = MemoryDecision(
        memory_action=MemoryAction(
            action=MemoryActionName.DELETE,
            arguments=DeleteArguments(memory_id="M003"),
        )
    )

    assert decision.to_dict() == {
        "memory_action": {
            "action": "DELETE",
            "arguments": {"memory_id": "M003"},
        }
    }
