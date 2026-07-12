import json

from examples.csl_memory_agent.csl_memory import DARTCSLMemorySystem
from examples.csl_memory_agent.csl_memory.csl_audit import certificate_hash
from examples.csl_memory_agent.csl_memory.csl_contract import compile_task_contract
from examples.csl_memory_agent.csl_memory.csl_extraction import parse_event_chunk
from examples.csl_memory_agent.csl_memory.csl_guard import deterministic_guard_precheck, llm_hard_action_guard
from examples.csl_memory_agent.csl_memory.csl_visible_entities import (
    build_candidate_evidence_guidance,
    build_candidate_ledger,
    build_visible_decision_certificate,
)


def test_visible_contract_rule_injected_without_domain_label(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    task_prompt = "Before final action, verify all candidates were checked and total cost stays below budget."

    prompt = mem.wrap_user_prompt("Pick the current item.", task_instruction=task_prompt)

    assert "[DART-CSL STATE PACKET]" in prompt
    assert "all candidates" in prompt.lower()
    assert "budget" in prompt.lower()
    assert "MemoryArena" not in prompt


def test_action_guard_blocks_missing_evidence(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    task_prompt = "Do not finish unless the answer is supported by a read document."
    mem.wrap_user_prompt("Answer the current question.", task_instruction=task_prompt)

    result = mem.guard_action("Answer the current question.", "Finish[Some answer without document evidence]")

    assert result["allow"] is False
    assert result["needs_backfill"] is True


def test_invalidated_descendant_not_published(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    mem.add_chunk("Action: Finalize value X=10. Observation: accepted.")
    mem.add_chunk("Action: Derive Y=20 from X. Observation: accepted.")
    mem.add_chunk("Feedback: X was wrong; correct X is 12.")

    prompt = mem.wrap_user_prompt("Use the current valid derived state.")

    assert "Y=20" not in prompt or "invalid" in prompt.lower() or "repair" in prompt.lower()


def test_non_mock_hard_guard_calls_memory_llm():
    llm = _FakeGuardLLM()
    contract = compile_task_contract(
        task_id="guard_test",
        visible_task_prompt="Compare all candidates before final purchase. Total budget below 70.",
        tool_description="click[Buy Now] is final.",
    )

    result = llm_hard_action_guard(
        question="Buy the selected product.",
        contract=contract,
        state_packet="[DART-CSL STATE PACKET]\nEvidence pointers:\n- event_1",
        action="click[Buy Now]",
        llm=llm,
    )

    assert llm.calls == 1
    assert result["memory_llm_used"] is True
    assert result["allow"] is True


def test_memory_llm_block_remains_fail_closed_with_visible_evidence():
    llm = _BlockingGuardLLM()

    result = llm_hard_action_guard(
        question="Buy the selected product.",
        contract=None,
        state_packet=(
            "[DART-CSL STATE PACKET]\n"
            "Final states to preserve:\n"
            "- selected option S1O3. Comparison evidence before selection: "
            "S1O1 price=4.49; S1O2 price=6.99; S1O3 price=9.99\n"
            "Evidence pointers:\n- event_1"
        ),
        action="click[Buy Now]",
        llm=llm,
    )

    assert result["allow"] is False
    assert result["memory_llm_used"] is True
    assert "memory_llm_block_overridden_by_visible_evidence" not in result


def test_decision_certificate_blocks_non_optimal_selection():
    certificate = _certificate(selected="S1O2")

    result = deterministic_guard_precheck(
        action=f"select[S1O2]\nDecisionCertificate: {json.dumps(certificate)}",
        contract=None,
        state_packet="[DART-CSL STATE PACKET]\nEvidence pointers:\n- event_1",
    )

    assert result["allow"] is False
    assert result["decision_certificate_validation"]["valid"] is False
    assert "not objective-optimal" in str(result["violations"])


def test_decision_certificate_allows_complete_optimal_selection():
    certificate = _certificate(selected="S1O3")

    result = deterministic_guard_precheck(
        action=f"select[S1O3]\nDecisionCertificate: {json.dumps(certificate)}",
        contract=None,
        state_packet="[DART-CSL STATE PACKET]\nEvidence pointers:\n- event_1",
    )

    assert result["allow"] is True
    assert result["decision_certificate_validation"]["valid"] is True


def test_structured_event_keeps_properties_bound_to_each_entity(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    event = {
        "event_type": "search_result",
        "action": "search[visible candidates]",
        "observation": "two visible candidates",
        "entities": [
            {"subject": "option S1O1", "properties": {"id": "DEMO-S1O1", "price": 4.49}},
            {"subject": "option S1O2", "properties": {"id": "DEMO-S1O2", "price": 6.99}},
        ],
    }
    mem.add_chunk("DART_EVENT_JSON:\n" + json.dumps(event))
    mem.wrap_user_prompt("Compare the visible candidates.", task_instruction="Compare all visible candidates.")

    values = {(delta.subject, delta.variable): delta.value for delta in mem.storage.load_deltas()}

    assert values[("option S1O1", "id")] == "DEMO-S1O1"
    assert values[("option S1O2", "id")] == "DEMO-S1O2"
    assert values[("option S1O1", "price")] == 4.49
    assert values[("option S1O2", "price")] == 6.99


def test_buy_reuses_bound_selection_audit_when_state_is_unchanged(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    mem.llm = _CountingGuardLLM()
    certificate = _certificate(selected="S1O3", grounded=True)
    question = "Compare every candidate and buy the highest-priced eligible option."
    mem.wrap_user_prompt(question, task_instruction=question)
    mem.last_state_packet = "[DART-CSL STATE PACKET]\nEvidence pointers:\n- event_1"

    selection = mem.guard_action(
        question,
        f"select[S1O3]\nDecisionCertificate: {json.dumps(certificate)}",
    )
    assert selection["allow"] is True
    assert selection["audit_cache"]["status"] == "pending_selection_commit"
    assert mem.llm.calls == 1

    mem.add_chunk(_selection_event(certificate))
    mem.wrap_user_prompt(question, task_instruction=question)
    purchase = mem.guard_action(
        question,
        f"click[Buy Now]\nDecisionCertificate: {json.dumps(certificate)}",
    )

    assert purchase["allow"] is True
    assert purchase["audit_cache"]["hit"] is True
    assert purchase["memory_llm_used"] is False
    assert mem.llm.calls == 1
    assert mem.get_metrics()["guard_llm_calls_avoided"] == 1


def test_buy_reaudits_when_state_changes_after_selection(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    mem.llm = _CountingGuardLLM()
    certificate = _certificate(selected="S1O3", grounded=True)
    question = "Compare every candidate and buy the highest-priced eligible option."
    mem.wrap_user_prompt(question, task_instruction=question)
    mem.last_state_packet = "[DART-CSL STATE PACKET]\nEvidence pointers:\n- event_1"
    mem.guard_action(
        question,
        f"select[S1O3]\nDecisionCertificate: {json.dumps(certificate)}",
    )
    mem.add_chunk(_selection_event(certificate))
    mem.add_chunk(
        "DART_EVENT_JSON:\n"
        + json.dumps(
            {
                "event_type": "observation_update",
                "action": "observe[availability]",
                "observation": "availability was refreshed",
            }
        )
    )
    mem.wrap_user_prompt(question, task_instruction=question)

    purchase = mem.guard_action(
        question,
        f"click[Buy Now]\nDecisionCertificate: {json.dumps(certificate)}",
    )

    assert purchase["allow"] is True
    assert purchase["audit_cache"]["hit"] is False
    assert purchase["audit_cache"]["reason"] == "state_version_changed"
    assert mem.llm.calls == 2


def test_repeated_blocked_buy_reuses_denial_until_visible_state_changes(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    mem.llm = _CountingBlockingGuardLLM()
    packet = "[DART-CSL STATE PACKET]\nEvidence pointers:\n- event_1"
    mem.last_state_packet = packet

    first = mem.guard_action("Buy the inspected item.", "click[Buy Now]")
    state_version = mem.storage.state_version()
    mem.add_chunk(
        "Agent Action: click[Buy Now]\n"
        "Env Observation: DART-CSL guard blocked the proposed action. Gather evidence.\n"
        "Done: False"
    )
    mem.last_state_packet = packet
    second = mem.guard_action("Buy the inspected item.", "click[Buy Now]")

    assert first["allow"] is False
    assert first["audit_cache"]["status"] == "blocked_recorded"
    assert mem.storage.state_version() == state_version
    assert second["allow"] is False
    assert second["audit_cache"]["hit"] is True
    assert second["audit_cache"]["reused_from"] == "denial"
    assert mem.llm.calls == 1
    assert mem.get_metrics()["guard_llm_calls_avoided"] == 1


def test_certificate_hash_ignores_ephemeral_evidence_pointer_ids():
    left = _certificate(selected="S1O3", grounded=True)
    right = json.loads(json.dumps(left))
    left["candidates"][0]["evidence"] = ["state-old"]
    right["candidates"][0]["evidence"] = ["state-new"]

    assert certificate_hash(left) == certificate_hash(right)


def test_repeated_schema_induction_does_not_bump_version_without_new_shape(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = "Preserve the selected decision and its evidence."

    mem.wrap_user_prompt(question, task_instruction=question)
    first_version = mem.storage.current_schema_version()
    mem.wrap_user_prompt(question, task_instruction=question)

    assert mem.storage.current_schema_version() == first_version


def test_memoryarena_labeled_turn_is_parsed_without_task_specific_logic():
    parsed = parse_event_chunk(
        "Step 3:\n"
        "User Instruction: choose from the visible page\n"
        "Agent Action: click[B00TEST]\n"
        "Env Observation: product page with price $9.99\n"
        "Done: False"
    )

    assert parsed["action"] == "click[B00TEST]"
    assert parsed["observation"] == "product page with price $9.99"


def test_memoryarena_done_false_does_not_finalize_repeated_instruction_text():
    parsed = parse_event_chunk(
        "Step 1:\n"
        "User Instruction: Select compatible products after every option is completed.\n"
        "Agent Action: search[cake mix]\n"
        "Env Observation: WebShop [SEP] Search\n"
        "Done: False"
    )

    assert parsed["is_success"] is False
    assert parsed["is_final_action"] is False
    assert parsed["boundary_signal"] is False
    assert parsed["explicit_done"] is False


def test_memoryarena_soft_exit_is_failure_even_when_done_is_true():
    parsed = parse_event_chunk(
        "Step 17:\n"
        "Agent Action: click[Next >]\n"
        "Env Observation: WEBSHOPPLUS SOFT EXIT\n"
        "Reason: repeat_action>=7\n"
        "Done: True"
    )

    assert parsed["is_success"] is False
    assert parsed["is_failure"] is True
    assert parsed["boundary_signal"] is True


def test_webshop_detail_page_binds_price_and_rating_to_asin(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = _webshop_question()
    observation = _webshop_detail(
        question,
        "DUNCAN0001",
        "Duncan Hines Signature Cake Mix, Strawberry Supreme, 16.5 oz",
        3.59,
        4.6,
    )
    mem.add_chunk(_memoryarena_turn("click[DUNCAN0001]", observation))
    mem.wrap_user_prompt(observation, task_instruction=question)

    values = {(state.subject, state.variable): state.value for state in mem.storage.load_materialized_state()}
    states = {(state.subject, state.variable): state for state in mem.storage.load_materialized_state()}

    assert values[("product DUNCAN0001", "price")] == 3.59
    assert values[("product DUNCAN0001", "rating")] == 4.6
    assert values[("product DUNCAN0001", "detail_observed")] is True
    assert values[("current_selection", "selected_option_id")] == "DUNCAN0001"
    assert states[("current_selection", "selected_option_id")].status == "candidate"
    assert not any(subject == "option you" for subject, _variable in values)


def test_candidate_ledger_tracks_exact_visible_evidence(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = _webshop_question()
    observations = [
        _webshop_detail(
            question,
            "DUNCAN0001",
            "Duncan Hines Signature Cake Mix, Strawberry Supreme, 16.5 oz",
            3.59,
            4.6,
        ),
        _webshop_detail(
            question,
            "SWEETLOW01",
            "Sweet 'N Low Lemon Cake Mix",
            5.90,
            3.6,
        ),
    ]
    for asin, observation in zip(["DUNCAN0001", "SWEETLOW01"], observations):
        mem.add_chunk(_memoryarena_turn(f"click[{asin}]", observation))

    wrapped = mem.wrap_user_prompt(observations[-1], task_instruction=question)
    ledger = build_candidate_ledger(
        question=question,
        materialized_state=mem.storage.load_materialized_state(),
    )

    assert any("Duncan Hines" in row and "detail=yes" in row for row in ledger)
    assert any("Sweet 'N Low" in row and "rating=3.6" in row for row in ledger)
    assert all("DUNCAN0001" not in row and "SWEETLOW01" not in row for row in ledger)
    assert any("unresolved" in row for row in ledger)
    assert "Candidate evidence ledger" in wrapped


def test_candidate_guidance_resolves_remaining_options_before_ranking(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = _webshop_question()
    observation = (
        f"WebShop [SEP] Instruction: [SEP] {question} [SEP] Back to Search "
        "[SEP] Page 1 [SEP] DUNCAN0001 "
        "[SEP] Duncan Hines Signature Cake Mix, Strawberry Supreme, 16.5 oz "
        "[SEP] $3.59"
    )
    mem.add_chunk(_memoryarena_turn("search[duncan strawberry cake mix]", observation))
    wrapped = mem.wrap_user_prompt(observation, task_instruction=question)

    guidance = build_candidate_evidence_guidance(
        question=observation,
        materialized_state=mem.storage.load_materialized_state(),
    )

    assert guidance is not None
    assert guidance.startswith("search[Sweet 'N Low lemon cake mix")
    assert "Required state change" in wrapped
    assert "search[Sweet 'N Low lemon cake mix" in wrapped


def test_candidate_guidance_rejects_implicit_multipack(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = _webshop_question()
    observations = [
        _webshop_detail(
            question,
            "DUNCAN0001",
            "Duncan Hines Signature Cake Mix, Strawberry Supreme, 16.5 oz",
            3.59,
            4.6,
        ),
        _webshop_detail(
            question,
            "SWEETLOW01",
            "Sweet 'N Low Lemon Cake Mix",
            5.90,
            3.6,
        ),
        _webshop_detail(
            question,
            "SIMPLEPACK",
            "Simple Mills Almond Flour Baking Mix, Gluten Free Vanilla Cake Mix, 11.5oz, 3 Count",
            23.97,
            4.6,
        ),
    ]
    for asin, detail in zip(["DUNCAN0001", "SWEETLOW01", "SIMPLEPACK"], observations):
        mem.add_chunk(_memoryarena_turn(f"click[{asin}]", detail))
    observation = observations[-1]
    mem.wrap_user_prompt(observation, task_instruction=question)

    guidance = build_candidate_evidence_guidance(
        question=observation,
        materialized_state=mem.storage.load_materialized_state(),
    )

    assert guidance is not None
    assert guidance.startswith("click[Back to Search]")
    assert "package/variant" in guidance

    search_guidance = build_candidate_evidence_guidance(
        question=question + " [SEP] Search",
        materialized_state=mem.storage.load_materialized_state(),
    )
    assert search_guidance is not None
    assert search_guidance.startswith("search[Simple Mills Almond Flour Baking Mix")
    assert "11.5 oz" in search_guidance
    assert "3 Count" not in search_guidance
    assert "single pack" not in search_guidance

    target_results = (
        f"WebShop [SEP] Instruction: [SEP] {question} [SEP] Back to Search [SEP] Page 1 "
        "[SEP] SIMPLE0001 "
        "[SEP] Simple Mills Almond Flour Baking Mix, Gluten Free Vanilla Cake Mix, "
        "Muffin pan ready, Made with whole foods, (Packaging May Vary) [SEP] $6.99"
    )
    mem.add_chunk(_memoryarena_turn("search[simple mills vanilla cake mix 11.5 oz]", target_results))
    mem.wrap_user_prompt(target_results, task_instruction=question)
    target_guidance = build_candidate_evidence_guidance(
        question=target_results,
        materialized_state=mem.storage.load_materialized_state(),
    )
    assert target_guidance is not None
    assert target_guidance.startswith("click[SIMPLE0001]")


def test_identity_ranking_keeps_top_hypotheses_and_demotes_extra_bundle(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = _webshop_question()
    duncan = (
        question
        + " [SEP] Back to Search [SEP] DUNCAN0001 "
        "[SEP] Duncan Hines Signature Cake Mix, Strawberry Supreme, 16.5 oz [SEP] $3.59"
    )
    sweet = (
        question
        + " [SEP] Back to Search [SEP] SWEETLOW01 "
        "[SEP] Sweet 'N Low Lemon Cake Mix [SEP] $5.90"
    )
    simple = (
        question
        + " [SEP] Back to Search [SEP] Page 1 [SEP] Next > "
        "[SEP] B08FRM79XT [SEP] Simple Mills Almond Flour Baking Mix, Gluten Free "
        "Vanilla Cake Mix, Muffin pan ready, 3 Count & Baking Mix Variety Pack, "
        "Banana Muffin & Bread, Chocolate Muffin & Cake, 3 Count [SEP] $46.57 "
        "[SEP] B015NBNZ98 [SEP] Simple Mills Almond Flour Baking Mix, Gluten Free "
        "Vanilla Cake Mix, Muffin pan ready, Good for Baking, Nutrient Dense, "
        "11.5oz, 3 Count [SEP] $23.97"
    )
    mem.add_chunk(_memoryarena_turn("search[duncan]", duncan))
    mem.add_chunk(_memoryarena_turn("search[sweet]", sweet))
    mem.add_chunk(_memoryarena_turn("search[simple]", simple))
    wrapped = mem.wrap_user_prompt(simple, task_instruction=question)

    assert "B015NBNZ98" in wrapped
    assert "B08FRM79XT (bundle-conflict" in wrapped
    assert "11.5 oz" in wrapped
    assert mem.validate_transition(simple, "click[B015NBNZ98]")["allow"] is True
    assert mem.validate_transition(simple, "click[Buy Now]")["allow"] is False


def test_buy_with_visible_variant_uncertainty_fails_closed_before_llm(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    mem.llm = _CountingGuardLLM()
    question = _webshop_question()
    observations = [
        _webshop_detail(
            question,
            "DUNCAN0001",
            "Duncan Hines Signature Cake Mix, Strawberry Supreme, 16.5 oz",
            3.59,
            4.6,
        ),
        _webshop_detail(
            question,
            "SWEETLOW01",
            "Sweet 'N Low Lemon Cake Mix",
            5.90,
            3.6,
        ),
        _webshop_detail(
            question,
            "SIMPLEPACK",
            "Simple Mills Almond Flour Baking Mix, Gluten Free Vanilla Cake Mix, 11.5oz, 3 Count",
            23.97,
            4.6,
        ),
    ]
    for asin, observation in zip(["DUNCAN0001", "SWEETLOW01", "SIMPLEPACK"], observations):
        mem.add_chunk(_memoryarena_turn(f"click[{asin}]", observation))
    mem.wrap_user_prompt(observations[-1], task_instruction=question)

    purchase = mem.guard_action(observations[-1], "click[Buy Now]")

    assert purchase["allow"] is False
    assert purchase["decision_certificate_source"] == "memory_state"
    assert purchase["decision_certificate_validation"]["valid"] is False
    assert "unresolved uncertainties" in " ".join(
        purchase["decision_certificate_validation"]["errors"]
    )
    assert mem.llm.calls == 0


def test_webshop_selection_audit_is_bound_and_reused_for_buy(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    mem.llm = _CountingGuardLLM()
    question = _webshop_question()
    mem.wrap_user_prompt(question, task_instruction=question)

    observations = [
        _webshop_detail(
            question,
            "DUNCAN0001",
            "Duncan Hines Signature Cake Mix, Strawberry Supreme, 16.5 oz",
            3.59,
            4.6,
        ),
        _webshop_detail(
            question,
            "SWEETLOW01",
            "Sweet 'N Low Lemon Cake Mix",
            5.90,
            3.6,
        ),
        _webshop_detail(
            question,
            "SIMPLE0001",
            "Simple Mills Almond Flour Baking Mix, Gluten Free Vanilla Cake Mix, Muffin pan ready",
            6.99,
            4.6,
        ),
    ]
    for asin, observation in zip(["DUNCAN0001", "SWEETLOW01", "SIMPLE0001"], observations):
        mem.add_chunk(_memoryarena_turn(f"click[{asin}]", observation))

    selection = mem.audit_selection(observations[-1], "SIMPLE0001")

    assert selection["allow"] is True
    assert selection["status"] == "audited"
    assert selection["audit_cache"]["status"] == "bound"
    assert mem.llm.calls == 1

    purchase = mem.guard_action(observations[-1], "click[Buy Now]")

    assert purchase["allow"] is True
    assert purchase["decision_certificate_source"] == "memory_state"
    assert purchase["audit_cache"]["hit"] is True
    assert purchase["memory_llm_used"] is False
    assert mem.llm.calls == 1


def test_visible_na_rating_is_evaluated_as_unavailable_not_missing(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    mem.llm = _CountingGuardLLM()
    question = """You are shopping. All items combined must not exceed $70.
Use the ground-truth items in Purchase History as references.
Product 2:
**Preference:** Pick the highest-rated option.
**Available Options:**
- A Toyvian rainbow unicorn cake topper.
- A Gyufise gold glitter wedding cake topper.
----------------------------------------------------------------
"""
    observations = [
        _webshop_detail(
            question,
            "TOYVIAN001",
            "Toyvian Rainbow Unicorn Cake Topper",
            9.59,
            "N.A.",
        ),
        _webshop_detail(
            question,
            "GYUFISE001",
            "Gyufise Gold Glitter Wedding Cake Topper",
            5.99,
            4.7,
        ),
    ]
    for asin, observation in zip(["TOYVIAN001", "GYUFISE001"], observations):
        mem.add_chunk(_memoryarena_turn(f"click[{asin}]", observation))
    mem.wrap_user_prompt(observations[-1], task_instruction=question)

    certificate = build_visible_decision_certificate(
        question=question,
        materialized_state=mem.storage.load_materialized_state(),
        selected_option_id="GYUFISE001",
    )

    assert certificate is not None
    assert certificate["uncertainties"] == []
    assert "budget" not in certificate
    toyvian = next(row for row in certificate["candidates"] if row["option_id"] == "TOYVIAN001")
    assert toyvian["eligible"] is False
    assert toyvian["ineligible_reason"] == "rating was visibly unavailable"

    selection = mem.audit_selection(question, "GYUFISE001")
    purchase = mem.guard_action(observations[-1], "click[Buy Now]")

    assert selection["status"] == "audited"
    assert selection["audit_cache"]["status"] == "bound"
    assert purchase["allow"] is True
    assert purchase["audit_cache"]["hit"] is True
    assert mem.llm.calls == 1


def test_unknown_prior_dependency_is_soft_for_deterministic_guard(tmp_path):
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = """You are shopping.
Product 2:
Compatibility notes: Alloy-X pairs well with Ceramic-Z.
Constraint: Must be compatible with the previous decision.
Preference: Pick the highest-priced compatible option.
**Available Options:**
- A shell made from Ceramic-Z.
- A shell made from Polymer-W.
----------------------------------------------------------------
"""
    result_page = (
        question
        + "[SEP] Back to Search [SEP] Page 1 "
        "[SEP] SHELL00001 [SEP] Protective shell made from Ceramic-Z [SEP] $100 "
        "[SEP] POLYM00001 [SEP] Protective shell made from Polymer-W [SEP] $20"
    )
    mem.add_chunk(_memoryarena_turn("search[protective shell]", result_page))
    mem.wrap_user_prompt(result_page, task_instruction=question)
    detail = (
        question
        + "[SEP] Back to Search [SEP] Protective shell made from Ceramic-Z "
        "[SEP] Price: $100 [SEP] Rating: 4.5 [SEP] Buy Now"
    )
    mem.add_chunk(_memoryarena_turn("click[SHELL00001]", detail))
    packet = mem.wrap_user_prompt(detail, task_instruction=question)
    certificate = build_visible_decision_certificate(
        question=detail,
        materialized_state=mem.storage.load_materialized_state(),
        selected_option_id="SHELL00001",
    )

    assert certificate is not None
    assert certificate["uncertainties"] == []
    assert certificate["soft_uncertainties"]
    assert certificate["grounding_receipt"]["validated_against_environment"] is True
    action = "click[Buy Now]\nDecisionCertificate: " + json.dumps(certificate)
    result = deterministic_guard_precheck(
        action=action,
        contract=mem.storage.load_contract_card(),
        state_packet=packet,
    )
    assert result["allow"] is True


def _webshop_question():
    return """You are shopping.
**Total Budget:** All items combined must not exceed $70.
### Select Cake Base
**Goal:** Buy the highest-priced one in available options.
**Available Options:**
- A Duncan Hines Signature Cake Mix with Strawberry Supreme flavor in 16.5 oz.
- A Sweet 'N Low lemon cake mix with sugar-free ingredients.
- A Simple Mills Almond Flour Baking Mix with gluten free vanilla cake mix, ready for muffin pans.
----------------------------------------------------------------
"""


def _webshop_detail(question, asin, title, price, rating):
    return (
        f"WebShop [SEP] Instruction: [SEP] {question} [SEP] Back to Search "
        f"[SEP] {title} [SEP] Price: ${price} [SEP] Rating: {rating} "
        "[SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
    )


def _memoryarena_turn(action, observation):
    return (
        "Step 1:\n"
        "User Instruction: repeated task text\n"
        f"Agent Action: {action}\n"
        f"Env Observation: {observation}\n"
        "Done: False"
    )


def _certificate(*, selected: str, grounded: bool = False):
    certificate = {
        "expected_option_ids": ["S1O1", "S1O2", "S1O3"],
        "evaluated_option_ids": ["S1O1", "S1O2", "S1O3"],
        "objective": {"metric": "price", "direction": "maximize"},
        "candidates": [
            {"option_id": "S1O1", "eligible": True, "objective_value": 4.49, "evidence": "visible"},
            {"option_id": "S1O2", "eligible": True, "objective_value": 6.99, "evidence": "visible"},
            {"option_id": "S1O3", "eligible": True, "objective_value": 9.99, "evidence": "visible"},
        ],
        "selected_option_id": selected,
        "budget": {"before": 0.0, "after": 9.99 if selected == "S1O3" else 6.99, "limit": 70.0},
        "uncertainties": [],
    }
    if grounded:
        certificate["grounding_receipt"] = {
            "validated_against_environment": True,
            "validated_option_ids": ["S1O1", "S1O2", "S1O3"],
        }
    return certificate


def _selection_event(certificate):
    return "DART_EVENT_JSON:\n" + json.dumps(
        {
            "event_type": "selection_committed",
            "action": "select[S1O3]",
            "observation": "selected option S1O3",
            "is_final_action": True,
            "boundary_signal": True,
            "entities": [
                {
                    "subject": "option S1O3",
                    "status": "final",
                    "properties": {
                        "selected": True,
                        "decision_certificate": certificate,
                    },
                }
            ],
        }
    )


class _FakeGuardLLM:
    provider = "openai"

    def __init__(self):
        self.calls = 0

    def call_json_prompt(self, prompt):
        self.calls += 1
        assert "Proposed action:" in prompt
        return {
            "allow": "true",
            "violations": [],
            "repair_instruction": None,
            "needs_backfill": False,
            "missing_evidence": [],
        }


class _BlockingGuardLLM:
    provider = "openai"

    def call_json_prompt(self, prompt):
        assert "DART-CSL state packet" in prompt
        return {
            "allow": False,
            "violations": [{"constraint": "missing evidence", "why_violated": "fake", "evidence": "fake"}],
            "repair_instruction": "fake repair",
            "needs_backfill": True,
            "missing_evidence": ["fake"],
        }


class _CountingGuardLLM:
    provider = "openai"

    def __init__(self):
        self.calls = 0

    def call_json_prompt(self, prompt):
        self.calls += 1
        return {
            "allow": True,
            "violations": [],
            "repair_instruction": None,
            "needs_backfill": False,
            "missing_evidence": [],
            "decision_audit": {
                "hard_constraints_satisfied": True,
                "all_candidates_evaluated": True,
                "selected_is_objective_optimal": True,
                "budget_satisfied": True,
            },
        }

    def usage_dict(self):
        return {
            "calls": self.calls,
            "prompt_tokens": self.calls * 100,
            "completion_tokens": self.calls * 20,
            "total_tokens": self.calls * 120,
            "latency_ms": self.calls * 10.0,
        }


class _CountingBlockingGuardLLM(_CountingGuardLLM):
    def call_json_prompt(self, prompt):
        self.calls += 1
        return {
            "allow": False,
            "violations": [
                {
                    "constraint": "all candidates evaluated",
                    "why_violated": "comparison evidence is incomplete",
                    "evidence": "state packet",
                }
            ],
            "repair_instruction": "Gather the missing comparison evidence.",
            "needs_backfill": True,
            "missing_evidence": ["candidate comparison"],
        }
