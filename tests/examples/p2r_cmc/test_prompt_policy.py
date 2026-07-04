import pytest

from examples.p2r_cmc.memory_system.controller.action_parser import USE_ACTIONS, parse_controller_action
from examples.p2r_cmc.memory_system.controller.prompt_policy import RuleBasedPromptPolicy
from examples.p2r_cmc.memory_system.schemas import ClusterCard, MemoryAction, PolicyObservation


def test_policy_outputs_legal_use_json():
    observation = PolicyObservation(
        query="P2R-CMC MemoryArena",
        budget={"max_clusters": 1},
        cluster_cards=[
            ClusterCard(
                cluster_id="c1",
                title="P2R",
                summary="P2R-CMC MemoryArena",
                representative_memory="P2R-CMC uses MemoryArena.",
                size=1,
                retrieval_score=0.8,
                user_id="u1",
                project_id="p1",
            )
        ],
    )
    action = RuleBasedPromptPolicy().decide_use(observation)
    assert action.action == MemoryAction.EXPAND_CLUSTER
    assert action.cluster_ids == ["c1"]


def test_action_parser_rejects_illegal_action():
    with pytest.raises(ValueError):
        parse_controller_action({"action": "WRITE_NEW"}, allowed_actions=USE_ACTIONS)


def test_use_action_parser_rejects_check_conflict():
    with pytest.raises(ValueError):
        parse_controller_action({"action": "CHECK_CONFLICT"}, allowed_actions=USE_ACTIONS)
