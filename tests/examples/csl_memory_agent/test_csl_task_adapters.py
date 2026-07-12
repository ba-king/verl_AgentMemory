import json

from examples.csl_memory_agent.csl_memory import DARTCSLMemorySystem
from examples.csl_memory_agent.csl_memory.csl_task_adapter import (
    ActionDirective,
    GenericTaskAdapter,
    NoProgressTracker,
    ProgressContext,
    ProgressSignal,
    RecoveryAction,
    TaskAdapterRegistry,
    TaskIR,
    is_declared_commit_action,
    validate_action_directive,
)
from examples.csl_memory_agent.csl_memory.csl_transition import (
    blocked_transition_result,
    enforce_required_transition,
)
from examples.csl_memory_agent.csl_memory.csl_visible_entities import WebShopTaskAdapter
from examples.csl_memory_agent.csl_memory.memory_arena_adapter import (
    DARTCSLMemoryArenaMemorySystem,
)
from examples.csl_memory_agent.scripts.run_memoryarena_shopping_with_usage import (
    _action_contract_violation,
    _extract_action_contract,
)


def test_no_progress_detector_is_domain_agnostic():
    tracker = NoProgressTracker(repeat_limit=2)

    first = tracker.observe(
        scope_id="itinerary-1",
        action="revise[route-a]",
        observation="The itinerary still exceeds the public budget.",
        evidence_items=["total_cost=1250"],
    )
    second = tracker.observe(
        scope_id="itinerary-1",
        action="revise[route-a]",
        observation="The itinerary still exceeds the public budget.",
        evidence_items=["total_cost=1250"],
    )

    assert first is not None and first.stale is False
    assert second is not None and second.stale is True
    guidance = GenericTaskAdapter().recommend_next_action(
        task=GenericTaskAdapter().compile_task("Minimize total itinerary cost."),
        question="Minimize total itinerary cost.",
        materialized_state=[],
        progress=tracker.context("itinerary-1"),
    )
    assert guidance is not None
    assert "revise[route-a]" in guidance
    assert "legal action" in guidance


def test_different_actions_returning_same_state_are_zero_progress():
    tracker = NoProgressTracker(repeat_limit=2)
    tracker.observe(
        scope_id="proof-1",
        action="expand[lemma-a]",
        observation="No additional obligations were discharged.",
        evidence_items=[],
    )
    second = tracker.observe(
        scope_id="proof-1",
        action="expand[lemma-b]",
        observation="No additional obligations were discharged.",
        evidence_items=[],
    )

    assert second is not None and second.stale is True
    assert tracker.context("proof-1").is_stale("expand[lemma-b]")


def test_packet_keeps_a_domain_agnostic_recent_execution_path(tmp_path):
    memory = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    observation = "The route matrix still has one unresolved feasibility branch."
    memory.add_chunk(_turn("revise[route-a]", observation))
    memory.add_chunk(_turn("inspect[constraint-b]", observation))

    wrapped = memory.wrap_user_prompt("Complete the route matrix before submission.")

    assert "Active execution path (recent transitions)" in wrapped
    assert "revise[route-a]" in wrapped
    assert "inspect[constraint-b]" in wrapped


def test_webshop_identity_resolution_is_invariant_to_option_order(tmp_path):
    result_page = (
        "[SEP] Back to Search [SEP] Page 1 "
        "[SEP] DRIVE00001 [SEP] Nova portable drive, Series QZ-82, 1 TB [SEP] $20 "
        "[SEP] DRIVE00002 [SEP] Nova portable drive, Series KX-41, 2 TB [SEP] $30"
    )
    orders = [
        [
            "A Nova portable drive, Series KX-41, 2 TB.",
            "A Nova portable drive, Series QZ-82, 1 TB.",
        ],
        [
            "A Nova portable drive, Series QZ-82, 1 TB.",
            "A Nova portable drive, Series KX-41, 2 TB.",
        ],
    ]

    recommendations = []
    for index, options in enumerate(orders):
        memory = DARTCSLMemorySystem({"state_dir": str(tmp_path / str(index))})
        question = _shopping_question(options, preference="lowest-priced")
        observation = question + result_page
        memory.add_chunk(_turn("search[Nova portable drives]", observation))
        wrapped = memory.wrap_user_prompt(observation, task_instruction=question)
        recommendations.append(
            next(line for line in wrapped.splitlines() if "Non-binding decision support" in line)
        )

    assert recommendations == [
        "Non-binding decision support (verify legality against the final current page): click[DRIVE00001] (open the verified objective winner before purchase)",
        "Non-binding decision support (verify legality against the final current page): click[DRIVE00001] (open the verified objective winner before purchase)",
    ]


def test_identity_dimensions_are_unit_normalized_and_recovery_keeps_target_values(tmp_path):
    question = _shopping_question(
        ["An insulated bottle, Model VX-72, 750 ml."],
        preference="lowest-priced",
    )
    mismatch = (
        question
        + "[SEP] Back to Search [SEP] Page 1 "
        "[SEP] BOTTLE0001 [SEP] Insulated bottle Model VX-72, 1 L [SEP] $25"
    )
    memory = DARTCSLMemorySystem({"state_dir": str(tmp_path / "mismatch")})
    memory.add_chunk(_turn("search[VX-72 bottle]", mismatch))
    mismatch_packet = memory.wrap_user_prompt(mismatch, task_instruction=question)

    assert "Required state change" in mismatch_packet
    assert "search[insulated bottle, Model VX-72, 750 ml]" in mismatch_packet
    assert "search[insulated bottle, Model VX-72, 750 ml 1" not in mismatch_packet

    equivalent = (
        question
        + "[SEP] Back to Search [SEP] Page 1 "
        "[SEP] BOTTLE0002 [SEP] Insulated bottle Model VX-72, 0.75 L [SEP] $24"
    )
    equivalent_memory = DARTCSLMemorySystem({"state_dir": str(tmp_path / "equivalent")})
    equivalent_memory.add_chunk(_turn("search[VX-72 bottle]", equivalent))
    equivalent_packet = equivalent_memory.wrap_user_prompt(equivalent, task_instruction=question)

    assert "Non-binding decision support" in equivalent_packet
    assert "click[BOTTLE0002]" in equivalent_packet


def test_stale_search_uses_a_different_public_recovery_operator(tmp_path):
    memory = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = _shopping_question(
        ["A Nova portable drive, Series KX-41, 2 TB."],
        preference="lowest-priced",
    )
    result_page = (
        question
        + "[SEP] Back to Search [SEP] Page 1 [SEP] Next > "
        "[SEP] DRIVE00001 [SEP] Nova portable drive, Series QZ-82, 1 TB [SEP] $20"
    )
    actions = [
        "search[Nova portable drive Series KX-41 2 TB]",
        "search[Nova portable drive Series KX-41 2 TB]",
        "search[Nova portable drive]",
        "search[Nova portable drive]",
    ]
    for action in actions:
        memory.add_chunk(_turn(action, result_page))

    wrapped = memory.wrap_user_prompt(result_page, task_instruction=question)

    assert "Required state change" in wrapped
    assert "click[Next >]" in wrapped
    assert memory.get_metrics()["no_progress_events"] == 3


def test_scope_switch_emits_global_decision_receipt_without_leaking_candidates(tmp_path):
    memory = DARTCSLMemoryArenaMemorySystem(user_id="receipt", state_dir=tmp_path)
    first = _shopping_question(
        ["A Nova portable drive, Series KX-41, 2 TB."],
        preference="highest-rated",
        product=1,
    )
    memory.wrap_user_prompt(first + "[SEP] Search")
    detail = (
        first
        + "[SEP] Back to Search [SEP] Nova portable drive, Series KX-41, 2 TB "
        "[SEP] Price: $30 [SEP] Rating: 4.8 [SEP] Buy Now"
    )
    memory.add_chunk(_turn("click[DRIVE00002]", detail))
    memory.add_chunk(
        _turn(
            "click[Buy Now]",
            "WEBSHOPPLUS TASK COMPLETED: purchase accepted",
            done=True,
        )
    )

    second = _shopping_question(
        ["A generic carrying case compatible with the previous decision."],
        preference="lowest-priced",
        product=2,
    )
    wrapped = memory.wrap_user_prompt(second + "[SEP] Search")

    assert "decision_receipt product-1" in wrapped
    assert "DRIVE00002" in wrapped
    assert memory.get_metrics()["decision_receipts_emitted"] == 1
    assert memory.get_metrics()["cross_scope_receipts_retrieved"] == 1
    current_products = [
        state.subject
        for state in memory.system._scoped_materialized_state()
        if state.subject.startswith("product ")
    ]
    assert current_products == []


def test_failed_scope_emits_provisional_semantic_summary_for_next_dependency(tmp_path):
    memory = DARTCSLMemoryArenaMemorySystem(user_id="provisional", state_dir=tmp_path)
    first = _shopping_question(
        ["A support frame made from Alloy-X."],
        preference="highest-rated",
        product=1,
    )
    memory.wrap_user_prompt(first + "[SEP] Search")
    detail = (
        first
        + "[SEP] Back to Search [SEP] Support frame made from Alloy-X "
        "[SEP] Price: $40 [SEP] Rating: 4.7 [SEP] Buy Now"
    )
    memory.add_chunk(_turn("click[FRAME00001]", detail))
    memory.wrap_user_prompt(detail)

    second = """You are shopping.
Product 2:
Compatibility notes: Alloy-X pairs well with Ceramic-Z.
Avoid: Alloy-X avoids Polymer-W.
Preference: Pick the highest-priced compatible exact option.
Constraint: Must be compatible with the previous decision.
**Available Options:**
- A shell made from Ceramic-Z.
- A shell made from Polymer-W.
----------------------------------------------------------------
"""
    result_page = (
        second
        + "[SEP] Back to Search [SEP] Page 1 "
        "[SEP] SHELL00001 [SEP] Protective shell made from Ceramic-Z [SEP] $20 "
        "[SEP] POLYM00001 [SEP] Protective shell made from Polymer-W [SEP] $100"
    )
    memory.wrap_user_prompt(second + "[SEP] Search")
    memory.add_chunk(_turn("search[protective shell]", result_page))
    memory.add_chunk(_turn("search[protective shell]", result_page))
    wrapped = memory.wrap_user_prompt(result_page)

    receipt = next(
        state.value
        for state in memory.system.storage.load_materialized_state()
        if state.variable == "decision_receipt"
        and state.attributes.get("source_scope_id") == "product-1"
    )
    assert receipt["status"] == "provisional"
    assert "Alloy-X" in receipt["committed_decision"]["semantic_option"]
    assert "click[SHELL00001]" in wrapped
    assert "prior_decision_summary" not in wrapped
    assert memory.get_metrics()["provisional_receipts_emitted"] == 1


def test_materialization_keeps_same_entity_separate_across_scopes(tmp_path):
    memory = DARTCSLMemoryArenaMemorySystem(user_id="scope-identity", state_dir=tmp_path)
    memory.wrap_user_prompt("Product 1:\nCompare the current candidates.")
    memory.add_chunk(_structured_price_event(10.0))
    memory.wrap_user_prompt("Product 1:\nUse the visible price.")

    memory.wrap_user_prompt("Product 2:\nCompare the current candidates.")
    memory.add_chunk(_structured_price_event(20.0))
    memory.wrap_user_prompt("Product 2:\nUse the visible price.")

    prices = [
        (state.attributes.get("scope_id"), state.value)
        for state in memory.system.storage.load_materialized_state()
        if state.subject == "product SHARED0001" and state.variable == "price"
    ]
    assert sorted(prices) == [("product-1", 10.0), ("product-2", 20.0)]


def test_adapter_compiles_arbitrary_public_relations_without_fixed_values():
    adapter = WebShopTaskAdapter()
    question = """You are shopping.
Product 2:
Compatibility notes: Alloy-X pairs well with Ceramic-Z. Fiber-Q pairs well with Glass-R.
Avoid: Alloy-X avoids Polymer-W.
Preference: Pick the lowest-priced option.
Constraint: Must be compatible with the previous decision.
**Available Options:**
- A shell made from Ceramic-Z.
- A shell made from Polymer-W.
----------------------------------------------------------------
"""

    task = adapter.compile_task(question)

    assert task.metadata["positive_relations"] == {
        "alloy-x": ["ceramic-z"],
        "fiber-q": ["glass-r"],
    }
    assert task.metadata["negative_relations"] == {"alloy-x": ["polymer-w"]}
    assert task.dependencies[0].source == "previous_decision"
    assert task.objective.metric == "price"
    assert task.objective.direction == "minimize"


def test_task_ir_reports_missing_prior_scope_summary(tmp_path):
    memory = DARTCSLMemoryArenaMemorySystem(user_id="missing-prior", state_dir=tmp_path)
    question = """You are shopping.
Product 2:
Compatibility notes: Alloy-X pairs well with Ceramic-Z.
Constraint: Must be compatible with the previous decision.
Preference: Pick the highest-priced compatible option.
**Available Options:**
- A shell made from Ceramic-Z.
----------------------------------------------------------------
"""

    wrapped = memory.wrap_user_prompt(question + "[SEP] Search")

    assert "prior_decision_summary" in wrapped
    assert memory.get_metrics()["missing_field_count"] == 1


def test_public_relation_and_prior_receipt_override_raw_objective_order(tmp_path):
    memory = DARTCSLMemoryArenaMemorySystem(user_id="dynamic-relation", state_dir=tmp_path)
    first = _shopping_question(
        ["A support frame made from Alloy-X."],
        preference="highest-rated",
        product=1,
    )
    memory.wrap_user_prompt(first + "[SEP] Search")
    detail = (
        first
        + "[SEP] Back to Search [SEP] Support frame made from Alloy-X "
        "[SEP] Price: $40 [SEP] Rating: 4.7 [SEP] Buy Now"
    )
    memory.add_chunk(_turn("click[FRAME00001]", detail))
    memory.add_chunk(
        _turn(
            "click[Buy Now]",
            "WEBSHOPPLUS TASK COMPLETED: purchase accepted",
            done=True,
        )
    )

    second = """You are shopping.
Product 2:
Compatibility notes: Alloy-X pairs well with Ceramic-Z.
Avoid: Alloy-X avoids Polymer-W.
Preference: Pick the highest-priced compatible exact option.
Constraint: Must be compatible with the previous decision.
**Available Options:**
- A shell made from Ceramic-Z.
- A shell made from Polymer-W.
----------------------------------------------------------------
"""
    result_page = (
        second
        + "[SEP] Back to Search [SEP] Page 1 "
        "[SEP] SHELL00001 [SEP] Protective shell made from Ceramic-Z [SEP] $20 "
        "[SEP] POLYM00001 [SEP] Protective shell made from Polymer-W [SEP] $100"
    )
    memory.wrap_user_prompt(second + "[SEP] Search")
    memory.add_chunk(_turn("search[protective shell]", result_page))
    wrapped = memory.wrap_user_prompt(result_page)

    assert "Alloy-X" in wrapped
    assert (
        "Non-binding decision support (verify legality against the final current page): click[SHELL00001] "
        "(open the verified objective winner before purchase)"
    ) in wrapped
    assert "click[POLYM00001] (open the verified objective winner" not in wrapped


def test_formal_problem_uses_generic_adapter():
    registry = TaskAdapterRegistry([WebShopTaskAdapter()])

    adapter = registry.resolve("Given assumptions A and B, prove the final inequality.")

    assert adapter.name == "generic"


def test_external_adapter_can_be_registered_without_core_edits(tmp_path):
    class RouteAdapter(GenericTaskAdapter):
        name = "route"

        def supports(self, question):
            return "route matrix" in question.lower()

    memory = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    memory.register_task_adapter(RouteAdapter())

    assert memory.adapter_registry.resolve("Optimize this route matrix.").name == "route"


def test_generic_required_directive_is_enforced_without_domain_logic(tmp_path):
    class RouteAdapter(GenericTaskAdapter):
        name = "route"

        def supports(self, question):
            return "route matrix" in question.lower()

        def build_action_directive(self, **_kwargs):
            return ActionDirective(
                action="revise[route-b]",
                reason="route-b is the only unresolved feasibility branch",
                mode="required",
                obligation_id="resolve-route-b",
            )

    memory = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    memory.register_task_adapter(RouteAdapter())
    prompt = memory.wrap_user_prompt("Optimize this route matrix before submission.")

    blocked = memory.validate_transition(
        "Optimize this route matrix before submission.",
        "revise[route-a]",
    )
    allowed = memory.validate_transition(
        "Optimize this route matrix before submission.",
        "revise[route-b]",
    )

    assert "Required evidence transition" in prompt
    assert blocked["allow"] is False
    assert blocked["status"] == "required_violation"
    assert allowed["allow"] is True
    assert allowed["status"] == "required_satisfied"


def test_state_change_obligation_accepts_a_recovery_set_and_expires_with_lease():
    directive = ActionDirective(
        action="inspect[branch-b]",
        reason="resolve the remaining feasibility branch",
        mode="required_state_change",
        obligation_id="resolve-branch-b",
        required_state_delta=("branch-b feasibility becomes known",),
        recovery_actions=(
            RecoveryAction(
                action="revise[route-b]",
                operator="revise",
                reason="repair the unresolved branch",
            ),
        ),
        forbidden_actions=("submit[*]",),
        lease_id="lease-1",
    )

    assert validate_action_directive(
        "inspect[branch-b]",
        directive,
        current_lease_id="lease-1",
    )[0] is True
    assert validate_action_directive(
        "revise[route-b]",
        directive,
        current_lease_id="lease-1",
    )[0] is True
    assert validate_action_directive(
        "submit[final]",
        directive,
        current_lease_id="lease-1",
    )[0] is False
    assert validate_action_directive(
        "inspect[branch-a]",
        directive,
        current_lease_id="lease-1",
    )[0] is False
    assert validate_action_directive(
        "submit[final]",
        directive,
        current_lease_id="lease-2",
    )[0] is True


def test_state_change_obligation_caps_consecutive_recovery_actions():
    signals = tuple(
        ProgressSignal(
            scope_id="proof-1",
            action="inspect[branch-b]",
            observation_fingerprint=f"state-{index}",
            repeat_count=1,
            evidence_novel=True,
            stale=False,
        )
        for index in range(2)
    )
    directive = ActionDirective(
        action="inspect[branch-b]",
        reason="resolve branch-b",
        mode="required_state_change",
        obligation_id="resolve-branch-b",
        max_consecutive_action_repeats=2,
    )

    allowed, violation = validate_action_directive(
        "inspect[branch-b]",
        directive,
        progress=ProgressContext(scope_id="proof-1", signals=signals),
    )

    assert allowed is False
    assert "consecutive" in str(violation)


def test_webshop_recovery_does_not_force_seven_identical_page_actions(tmp_path):
    memory = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = _shopping_question(
        ["A Nova portable drive, Series KX-41, 2 TB."],
        preference="lowest-priced",
    )
    page_2 = question + "[SEP] Back to Search [SEP] Page 2 [SEP] Next >"
    page_3 = question + "[SEP] Back to Search [SEP] Page 3 [SEP] Next >"
    memory.add_chunk(_turn("click[Next >]", page_2))
    memory.add_chunk(_turn("click[Next >]", page_3))

    repeated = memory.validate_transition(page_3, "click[Next >]")
    alternative = memory.validate_transition(
        page_3,
        "search[Nova portable drive Series KX-41 2 TB]",
    )

    assert repeated["allow"] is False
    assert repeated["status"] == "state_change_action_violation"
    assert "consecutive" in repeated["violation"]
    assert alternative["allow"] is True


def test_commit_detection_uses_adapter_declarations_instead_of_domain_verbs():
    task = TaskIR(
        adapter_name="formal",
        goal_id="proof-1",
        commit_actions=["submit[*]"],
    )

    assert is_declared_commit_action(task, "submit[proof-7]") is True
    assert is_declared_commit_action(task, "revise[proof-7]") is False


def test_shared_transition_bridge_repairs_a_domain_agnostic_action_once():
    memory = _RouteTransitionMemory()
    agent = _RouteRepairAgent("revise[route-b]")
    messages = [{"role": "user", "content": "Route matrix state: branch-b unresolved."}]

    outcome = enforce_required_transition(
        memory=memory,
        agent=agent,
        prompt_source="Route matrix state: branch-b unresolved.",
        action="revise[route-a]",
        input_messages=messages,
        normalize_message=lambda role, content: {"role": role, "content": content},
    )

    assert outcome.action == "revise[route-b]"
    assert outcome.repair_attempted is True
    assert outcome.blocked is False
    assert agent.calls == 1
    assert memory.actions == ["revise[route-a]", "revise[route-b]"]
    assert outcome.input_messages[-1]["role"] == "user"


def test_shared_transition_bridge_fails_closed_and_preserves_environment_state():
    page = "Route matrix state: branch-b unresolved."
    memory = _RouteTransitionMemory()
    agent = _RouteRepairAgent("revise[route-a]")

    outcome = enforce_required_transition(
        memory=memory,
        agent=agent,
        prompt_source=page,
        action="revise[route-a]",
        input_messages=[{"role": "user", "content": page}],
        normalize_message=lambda role, content: {"role": role, "content": content},
    )
    result = blocked_transition_result(prompt_source=page, outcome=outcome)

    assert outcome.blocked is True
    assert agent.calls == 1
    assert page in result["observation"]["state"]
    assert result["info"]["transition_blocked"] is True


def test_webshop_required_evidence_blocks_resolved_candidate_cycle(tmp_path):
    memory = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = _shopping_question(
        [
            "A Duncan strawberry cake mix in 16.5 oz.",
            "A Sweet Low lemon cake mix.",
            "A Simple Mills almond flour vanilla cake mix.",
        ],
        preference="highest-priced",
    )
    duncan_page = (
        question
        + "[SEP] Back to Search [SEP] Page 1 "
        "[SEP] DUNCAN0001 [SEP] Duncan strawberry cake mix, 16.5 oz [SEP] $3.59"
    )
    sweet_page = (
        question
        + "[SEP] Back to Search [SEP] Page 1 "
        "[SEP] SWEET00001 [SEP] Sweet Low lemon cake mix [SEP] $5.90"
    )
    memory.add_chunk(_turn("search[Duncan strawberry cake mix]", duncan_page))
    memory.wrap_user_prompt(duncan_page, task_instruction=question)
    memory.add_chunk(_turn("search[Sweet Low lemon cake mix]", sweet_page))
    wrapped = memory.wrap_user_prompt(sweet_page)

    blocked = memory.validate_transition(
        sweet_page,
        "search[Duncan strawberry cake mix]",
    )
    allowed = memory.validate_transition(
        sweet_page,
        "search[Simple Mills almond flour vanilla cake mix]",
    )

    assert "Required state change" in wrapped
    assert "Simple Mills" in wrapped
    assert blocked["allow"] is False
    assert blocked["directive"]["mode"] == "required_state_change"
    assert allowed["allow"] is True


def test_answer_adapter_extracts_only_an_unambiguous_action():
    assert _extract_action_contract("I will now use SEARCH[  query terms  ].") == "search[query terms]"
    assert _extract_action_contract("click[A] or click[B]") is None
    assert _extract_action_contract("") is None


def test_answer_adapter_rejects_historical_click_ids_from_memory_packet():
    messages = [
        {
            "role": "user",
            "content": (
                "<memory_context>historical candidate OLDID00001</memory_context>\n\n"
                "WebShop [SEP] Back to Search [SEP] NEWID00001 [SEP] Current item [SEP] $10"
            ),
        }
    ]

    assert _action_contract_violation("click[OLDID00001]", messages) is not None
    assert _action_contract_violation("click[NEWID00001]", messages) is None


def test_answer_adapter_uses_current_page_before_transition_repair_message():
    messages = [
        {
            "role": "user",
            "content": "WebShop [SEP] Back to Search [SEP] CURRENT001 [SEP] Current item [SEP] $10",
        },
        {"role": "assistant", "content": "search[stale branch]"},
        {
            "role": "user",
            "content": "The transition validator rejected that action; satisfy the required obligation.",
        },
    ]

    assert _action_contract_violation("click[CURRENT001]", messages) is None


def test_detail_page_prefers_visible_single_variant_over_search_loop(tmp_path):
    memory = DARTCSLMemorySystem({"state_dir": str(tmp_path)})
    question = _shopping_question(
        ["A Nova vanilla baking mix in 11.5 oz."],
        preference="highest-priced",
    )
    detail = (
        question
        + "[SEP] Back to Search [SEP] size "
        "[SEP] 11.5 Ounce (Pack of 1) [SEP] 11.5 Ounce (Pack of 3) "
        "[SEP] Nova Vanilla Baking Mix, 11.5 oz, 3 Count "
        "[SEP] Price: $24 [SEP] Rating: 4.5 [SEP] Buy Now"
    )
    memory.add_chunk(_turn("click[NOVAMIX001]", detail))
    wrapped = memory.wrap_user_prompt(detail, task_instruction=question)

    assert "click[11.5 Ounce (Pack of 1)]" in wrapped


def _shopping_question(options, *, preference, product=1):
    option_lines = "\n".join(f"- {option}" for option in options)
    return f"""You are shopping.
Product {product}:
Preference: Pick the {preference} exact option.
**Available Options:**
{option_lines}
----------------------------------------------------------------
"""


def _turn(action, observation, *, done=False):
    return (
        "Step 1:\n"
        f"Agent Action: {action}\n"
        f"Env Observation: {observation}\n"
        f"Done: {done}"
    )


class _RouteTransitionMemory:
    def __init__(self):
        self.actions = []

    def validate_transition(self, _prompt_source, action):
        self.actions.append(action)
        allowed = action == "revise[route-b]"
        return {
            "allow": allowed,
            "repair_instruction": "Resolve branch-b with revise[route-b].",
        }


class _RouteRepairAgent:
    def __init__(self, repaired_action):
        self.repaired_action = repaired_action
        self.calls = 0

    def act_with_messages(self, _messages):
        self.calls += 1
        return self.repaired_action


def _structured_price_event(price):
    return "DART_EVENT_JSON:\n" + json.dumps(
        {
            "event_type": "search_result",
            "action": "inspect[current entity]",
            "observation": "one public entity",
            "entities": [
                {
                    "subject": "product SHARED0001",
                    "properties": {"price": price},
                }
            ],
        }
    )
