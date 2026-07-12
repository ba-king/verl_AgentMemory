from examples.csl_memory_agent.csl_memory import DARTCSLMemorySystem


def test_generic_state_packet_with_prior_final_decision(tmp_path):
    mem = DARTCSLMemorySystem(
        {
            "state_dir": str(tmp_path),
            "llm": {"provider": "mock"},
        }
    )

    mem.add_chunk("Action: Inspect option A. Observation: Option A has attribute alpha, score 4.7, id A1.")
    mem.add_chunk("Action: Select option A as the final choice. Observation: The choice was accepted.")

    prompt = mem.wrap_user_prompt(
        "Continue the task and choose the next compatible option. Prefer the highest score among valid options."
    )

    assert "<memory_context>" in prompt
    assert "option A" in prompt
    assert "final" in prompt.lower()
    assert "attribute" in prompt
    assert "alpha" in prompt
    assert "Before final action" in prompt


def test_ablation_can_disable_negative_lane(tmp_path):
    mem = DARTCSLMemorySystem(
        {
            "state_dir": str(tmp_path),
            "ablation": {"disable_negative_memory": True},
        }
    )
    mem.add_chunk("Action: Try candidate B. Observation: candidate B failed validation and should be avoided.")
    prompt = mem.wrap_user_prompt("Choose a valid next state.")
    assert "Negative Lane" not in prompt
