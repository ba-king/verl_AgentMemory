from __future__ import annotations

import argparse
from pathlib import Path


FACTORY_MARKER = "# DART-CSL VTCC registration"
SERVER_GUARD_MARKER = "# DART-CSL guard endpoint"
CLIENT_GUARD_MARKER = "# DART-CSL guard client"
RUNNER_GUARD_MARKER = "# DART-CSL guarded action hook"
SERVER_TRANSITION_MARKER = "# DART-CSL transition-validation endpoint"
CLIENT_TRANSITION_MARKER = "# DART-CSL transition-validation client"
RUNNER_TRANSITION_MARKER = "# DART-CSL required-evidence transition hook"
RUNNER_TRANSITION_IMPORT_MARKER = "# DART-CSL shared transition helper import"
RUNNER_TRANSITION_HELPER_MARKER = "# DART-CSL shared transition helper v1"
FAST_GOALS_MARKER = "# MemoryArena fast lite-goal initialization"
SERVER_SELECTION_MARKER = "# DART-CSL selection-audit endpoint"
CLIENT_SELECTION_MARKER = "# DART-CSL selection-audit client"
RUNNER_SELECTION_MARKER = "# DART-CSL post-selection audit hook"
RUNNER_DEDUPE_MARKER = "# Avoid duplicating the latest observation"
RUNNER_REFRESH_MARKER = "# DART-CSL refresh memory on every action turn"
RUNNER_BOUNDED_CONTEXT_MARKER = "# DART-CSL bounded answer-agent context"
ANSWER_PROMPT_MARKER = "# DART-CSL current-page action discipline"
ANSWER_PROMPT_V2_MARKER = "# DART-CSL evidence-first action discipline v2"
ANSWER_PROMPT_V3_MARKER = "# DART-CSL evidence-workflow action discipline v3"
ANSWER_PROMPT_V4_MARKER = "# DART-CSL adapter-driven evidence discipline v4"
ANSWER_PROMPT_V5_MARKER = "# DART-CSL evidence-not-authority discipline v5"
ANSWER_PROMPT_V6_MARKER = "# DART-CSL typed transition discipline v6"
ANSWER_PROMPT_V7_MARKER = "# DART-CSL state-change action-set discipline v7"
FULL_SUMMARY_MARKER = "# DART-CSL parse WebShopPlus purchase-only summaries"
GUARD_CONTEXT_MARKER = "# Preserve the unchanged page after a guard block"
GUARD_CONTEXT_DEDUPE_MARKER = "# Strip earlier guard feedback before preserving the page"


def main() -> None:
    parser = argparse.ArgumentParser(description="Register DART-CSL in a MemoryArena checkout.")
    parser.add_argument("--memoryarena-dir", required=True, help="Path to the MemoryArena repository.")
    args = parser.parse_args()

    root = Path(args.memoryarena_dir).expanduser().resolve()
    targets = [
        root / "memory/server.py",
        root / "memory/client.py",
        root / "run_shopping.py",
        root
        / "env/env_systems/web_shopping_env/runtime/service/web_agent_site/engine/goal.py",
        root
        / "env/env_systems/web_shopping_env/runtime/service/web_agent_site/envs/web_agent_text_env.py",
        root / "agent/webshop.py",
        root / "env/env_systems/web_shopping_env/runtime/runner/summary_build.py",
    ]
    missing = [str(path) for path in targets if not path.exists()]
    if missing:
        raise FileNotFoundError(f"MemoryArena integration targets not found: {missing}")

    changed = [
        _register_factory(targets[0]),
        _register_guard_endpoint(targets[0]),
        _register_transition_endpoint(targets[0]),
        _register_selection_endpoint(targets[0]),
        _register_guard_client(targets[1]),
        _register_transition_client(targets[1]),
        _register_selection_client(targets[1]),
        _ensure_runner_re_import(targets[2]),
        _ensure_transition_helper_import(targets[2]),
        _register_shopping_guard(targets[2]),
        _register_transition_validation(targets[2]),
        _register_guard_block_context(targets[2]),
        _register_post_selection_audit(targets[2]),
        _register_observation_dedupe(targets[2]),
        _register_memory_refresh_and_bounded_context(targets[2]),
        _register_fast_lite_goals(targets[3], targets[4]),
        _register_answer_prompt(targets[5]),
        _register_full_summary_unmatched_purchases(targets[6]),
    ]
    status = "updated" if any(changed) else "already registered"
    print(f"DART-CSL {status} in {root}")


def _register_factory(server: Path) -> bool:
    text = server.read_text(encoding="utf-8")
    if FACTORY_MARKER in text:
        return False
    registration = f'''

{FACTORY_MARKER}
def _make_dart_csl_vtcc():
    import os
    from examples.csl_memory_agent.csl_memory.memory_arena_adapter import (
        DARTCSLMemoryArenaMemorySystem,
    )

    return DARTCSLMemoryArenaMemorySystem(config={{
        "state_dir": os.getenv("CSL_STATE_DIR", ".csl_state"),
        "llm": {{
            "provider": os.getenv("CSL_MEMORY_PROVIDER", "mock"),
            "model": os.getenv("CSL_MEMORY_MODEL", "gpt-4.1-mini"),
            "api_key": os.getenv("CSL_MEMORY_API_KEY"),
            "base_url": os.getenv("CSL_MEMORY_BASE_URL"),
            "temperature": 0,
            "max_retries": 3,
        }},
    }})


MEMORY_FACTORIES["dart_csl_vtcc"] = _make_dart_csl_vtcc
'''
    anchor = "MEMORY_FACTORIES: Dict[str, Callable[[], object]] = {"
    if anchor not in text:
        raise RuntimeError("Could not locate MEMORY_FACTORIES in memory/server.py")
    start = text.index(anchor)
    brace_start = text.index("{", start)
    depth = 0
    end = None
    for index in range(brace_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise RuntimeError("Could not parse MEMORY_FACTORIES dictionary")
    server.write_text(text[:end] + registration + text[end:], encoding="utf-8")
    return True


def _register_guard_endpoint(server: Path) -> bool:
    text = server.read_text(encoding="utf-8")
    if SERVER_GUARD_MARKER in text:
        return False
    endpoint = f'''

{SERVER_GUARD_MARKER}
class DARTCSLGuardRequest(BaseModel):
    user_id: str
    question: str
    action: str
    memory_system_name: str


@app.post("/memory/guard_action")
def dart_csl_guard_action(req: DARTCSLGuardRequest):
    memory_system = _get_memory(req.user_id, req.memory_system_name)
    guard = getattr(memory_system, "guard_action", None)
    if not callable(guard):
        return {{"status": "unsupported", "result": {{"allow": True}}}}
    return {{"status": "ok", "result": guard(req.question, req.action)}}
'''
    anchor = 'if __name__ == "__main__":'
    if anchor not in text:
        raise RuntimeError("Could not locate memory server main guard")
    server.write_text(text.replace(anchor, endpoint + "\n\n" + anchor, 1), encoding="utf-8")
    return True


def _register_guard_client(client: Path) -> bool:
    text = client.read_text(encoding="utf-8")
    if CLIENT_GUARD_MARKER in text:
        return False
    method = f'''    {CLIENT_GUARD_MARKER}
    def guard_action(self, question: str, action: str) -> dict:
        data = self._post(
            "/memory/guard_action",
            {{
                "user_id": self.user_id,
                "memory_system_name": self.memory_system_name,
                "question": question,
                "action": action,
            }},
        )
        return data["result"]

'''
    anchor = "    def _post(self, path: str, payload: dict) -> dict:"
    if anchor not in text:
        raise RuntimeError("Could not locate MemoryClient._post")
    client.write_text(text.replace(anchor, method + anchor, 1), encoding="utf-8")
    return True


def _register_transition_endpoint(server: Path) -> bool:
    text = server.read_text(encoding="utf-8")
    if SERVER_TRANSITION_MARKER in text:
        return False
    endpoint = f'''

{SERVER_TRANSITION_MARKER}
class DARTCSLTransitionRequest(BaseModel):
    user_id: str
    question: str
    action: str
    memory_system_name: str


@app.post("/memory/validate_transition")
def dart_csl_validate_transition(req: DARTCSLTransitionRequest):
    memory_system = _get_memory(req.user_id, req.memory_system_name)
    validator = getattr(memory_system, "validate_transition", None)
    if not callable(validator):
        return {{"status": "unsupported", "result": {{"allow": True, "status": "unsupported"}}}}
    return {{"status": "ok", "result": validator(req.question, req.action)}}
'''
    anchor = 'if __name__ == "__main__":'
    if anchor not in text:
        raise RuntimeError("Could not locate memory server main guard")
    server.write_text(text.replace(anchor, endpoint + "\n\n" + anchor, 1), encoding="utf-8")
    return True


def _register_transition_client(client: Path) -> bool:
    text = client.read_text(encoding="utf-8")
    if CLIENT_TRANSITION_MARKER in text:
        return False
    method = f'''    {CLIENT_TRANSITION_MARKER}
    def validate_transition(self, question: str, action: str) -> dict:
        data = self._post(
            "/memory/validate_transition",
            {{
                "user_id": self.user_id,
                "memory_system_name": self.memory_system_name,
                "question": question,
                "action": action,
            }},
        )
        return data["result"]

'''
    anchor = "    def _post(self, path: str, payload: dict) -> dict:"
    if anchor not in text:
        raise RuntimeError("Could not locate MemoryClient._post")
    client.write_text(text.replace(anchor, method + anchor, 1), encoding="utf-8")
    return True


def _register_selection_endpoint(server: Path) -> bool:
    text = server.read_text(encoding="utf-8")
    if SERVER_SELECTION_MARKER in text:
        return False
    endpoint = f'''

{SERVER_SELECTION_MARKER}
class DARTCSLSelectionAuditRequest(BaseModel):
    user_id: str
    question: str
    selected_option_id: str
    memory_system_name: str


@app.post("/memory/audit_selection")
def dart_csl_audit_selection(req: DARTCSLSelectionAuditRequest):
    memory_system = _get_memory(req.user_id, req.memory_system_name)
    audit = getattr(memory_system, "audit_selection", None)
    if not callable(audit):
        return {{"status": "unsupported", "result": {{"allow": True, "status": "unsupported"}}}}
    return {{"status": "ok", "result": audit(req.question, req.selected_option_id)}}
'''
    anchor = 'if __name__ == "__main__":'
    if anchor not in text:
        raise RuntimeError("Could not locate memory server main guard")
    server.write_text(text.replace(anchor, endpoint + "\n\n" + anchor, 1), encoding="utf-8")
    return True


def _register_selection_client(client: Path) -> bool:
    text = client.read_text(encoding="utf-8")
    if CLIENT_SELECTION_MARKER in text:
        return False
    method = f'''    {CLIENT_SELECTION_MARKER}
    def audit_selection(self, question: str, selected_option_id: str) -> dict:
        data = self._post(
            "/memory/audit_selection",
            {{
                "user_id": self.user_id,
                "memory_system_name": self.memory_system_name,
                "question": question,
                "selected_option_id": selected_option_id,
            }},
        )
        return data["result"]

'''
    anchor = "    def _post(self, path: str, payload: dict) -> dict:"
    if anchor not in text:
        raise RuntimeError("Could not locate MemoryClient._post")
    client.write_text(text.replace(anchor, method + anchor, 1), encoding="utf-8")
    return True


def _register_shopping_guard(runner: Path) -> bool:
    text = runner.read_text(encoding="utf-8")
    if RUNNER_GUARD_MARKER in text:
        return False
    original = '''            action = agent.act_with_messages(input_messages)
            logger.debug("turn %d: action=%r", turn_idx, action[:120] if action else action)
            result = env_client.step(
                action,
                ground_truth=task_def.get("target_products"),
                need_judge=True,
            )
'''
    replacement = f'''            action = agent.act_with_messages(input_messages)
            logger.debug("turn %d: action=%r", turn_idx, action[:120] if action else action)

            {RUNNER_GUARD_MARKER}
            guard_result = None
            action_lower = (action or "").strip().lower()
            guard_required = (
                "buy now" in action_lower
                or action_lower.startswith(("finish[", "submit[", "book[", "purchase["))
            )
            if memory is not None and guard_required and hasattr(memory, "guard_action"):
                guard_result = memory.guard_action(prompt_source, action)

            if guard_result is not None and guard_result.get("allow") is False:
                repair = guard_result.get("repair_instruction") or "Gather missing evidence and retry."
                result = {{
                    "observation": {{
                        "state": "DART-CSL guard blocked the proposed action. " + repair
                    }},
                    "reward": 0.0,
                    "done": False,
                    "info": {{"dart_csl_guard": guard_result, "guard_blocked": True}},
                }}
            else:
                result = env_client.step(
                    action,
                    ground_truth=task_def.get("target_products"),
                    need_judge=True,
                )
                if guard_result is not None:
                    result.setdefault("info", {{}})["dart_csl_guard"] = guard_result
'''
    if original not in text:
        raise RuntimeError("Could not locate the WebShop action execution block")
    runner.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    return True


def _register_transition_validation(runner: Path) -> bool:
    text = runner.read_text(encoding="utf-8")
    if RUNNER_TRANSITION_HELPER_MARKER in text:
        return False

    guard_anchor = f'''            {RUNNER_GUARD_MARKER}
            guard_result = None
'''
    transition_block = f'''            {RUNNER_TRANSITION_MARKER}
            {RUNNER_TRANSITION_HELPER_MARKER}
            transition_outcome = enforce_required_transition(
                memory=memory,
                agent=agent,
                prompt_source=prompt_source,
                action=action,
                input_messages=input_messages,
                normalize_message=_normalize_api_message,
            )
            action = transition_outcome.action
            input_messages = transition_outcome.input_messages
            transition_result = transition_outcome.validation
            transition_blocked = transition_outcome.blocked
            if transition_outcome.repair_attempted:
                logger.debug(
                    "turn %d: transition repair action=%r allow=%s",
                    turn_idx,
                    action[:120] if action else action,
                    not transition_blocked,
                )

            {RUNNER_GUARD_MARKER}
            guard_result = None
'''
    if RUNNER_TRANSITION_MARKER in text:
        start = text.index(f"            {RUNNER_TRANSITION_MARKER}\n")
        end = text.index(f"            {RUNNER_GUARD_MARKER}\n", start)
        text = text[:start] + transition_block + text[end + len(guard_anchor) :]
    else:
        if guard_anchor not in text:
            raise RuntimeError("Could not locate the registered DART-CSL guard block")
        text = text.replace(guard_anchor, transition_block, 1)

    guard_call = '''            if memory is not None and guard_required and hasattr(memory, "guard_action"):
                guard_result = memory.guard_action(prompt_source, action)

            if guard_result is not None and guard_result.get("allow") is False:
'''
    old_guarded_call = '''            if (
                not transition_blocked
                and memory is not None
                and guard_required
                and hasattr(memory, "guard_action")
            ):
                guard_result = memory.guard_action(prompt_source, action)

            if transition_blocked:
                final_transition = transition_result.get("repair") or {}
                repair = (
                    final_transition.get("repair_instruction")
                    or "Choose an action that satisfies the required evidence obligation."
                )
                result = {
                    "observation": {
                        "state": (
                            "DART-CSL transition validator blocked the proposed action. "
                            + repair
                            + "\\n\\nCurrent environment remains unchanged:\\n"
                            + prompt_source.rsplit(
                                "\\n\\nCurrent environment remains unchanged:\\n", 1
                            )[-1]
                        )
                    },
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "dart_csl_transition": transition_result,
                        "transition_blocked": True,
                    },
                }
            elif guard_result is not None and guard_result.get("allow") is False:
'''
    guarded_call = '''            if (
                not transition_blocked
                and memory is not None
                and guard_required
                and hasattr(memory, "guard_action")
            ):
                guard_result = memory.guard_action(prompt_source, action)

            if transition_blocked:
                result = blocked_transition_result(
                    prompt_source=prompt_source,
                    outcome=transition_outcome,
                )
            elif guard_result is not None and guard_result.get("allow") is False:
'''
    if old_guarded_call in text:
        text = text.replace(old_guarded_call, guarded_call, 1)
    elif guard_call in text:
        text = text.replace(guard_call, guarded_call, 1)
    else:
        raise RuntimeError("Could not locate guard call for transition integration")

    observation_anchor = '''            observation = result["observation"]
'''
    observation_replacement = '''            if transition_result is not None:
                result.setdefault("info", {})["dart_csl_transition"] = transition_result
            observation = result["observation"]
'''
    if observation_anchor not in text:
        raise RuntimeError("Could not locate runner observation assignment")
    if observation_replacement not in text:
        if observation_anchor not in text:
            raise RuntimeError("Could not locate runner observation assignment")
        text = text.replace(observation_anchor, observation_replacement, 1)
    runner.write_text(text, encoding="utf-8")
    return True


def _register_guard_block_context(runner: Path) -> bool:
    text = runner.read_text(encoding="utf-8")
    if GUARD_CONTEXT_DEDUPE_MARKER in text:
        return False
    if GUARD_CONTEXT_MARKER in text:
        original = '''                        + prompt_source
                    )
'''
        replacement = f'''                        {GUARD_CONTEXT_DEDUPE_MARKER}
                        + prompt_source.rsplit(
                            "\\n\\nCurrent environment remains unchanged:\\n", 1
                        )[-1]
                    )
'''
        if original not in text:
            raise RuntimeError("Could not locate the existing guarded page context")
        runner.write_text(text.replace(original, replacement, 1), encoding="utf-8")
        return True
    original = '''                    "state": "DART-CSL guard blocked the proposed action. " + repair
'''
    replacement = f'''                    {GUARD_CONTEXT_MARKER}
                    "state": (
                        "DART-CSL guard blocked the proposed action. "
                        + repair
                        + "\\n\\nCurrent environment remains unchanged:\\n"
                        {GUARD_CONTEXT_DEDUPE_MARKER}
                        + prompt_source.rsplit(
                            "\\n\\nCurrent environment remains unchanged:\\n", 1
                        )[-1]
                    )
'''
    if original not in text:
        raise RuntimeError("Could not locate guarded observation state")
    runner.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    return True


def _register_post_selection_audit(runner: Path) -> bool:
    text = runner.read_text(encoding="utf-8")
    if RUNNER_SELECTION_MARKER in text:
        return False
    original = '''            if use_step_memory and memory is not None:
                for entry in build_memory_entries(agent, task_instruction, [turn_record]):
                    memory.add(entry)
'''
    replacement = original + f'''                {RUNNER_SELECTION_MARKER}
                selected_match = re.fullmatch(
                    r"\\s*click\\[\\s*([A-Za-z0-9]{{10}})\\s*\\]\\s*",
                    action or "",
                    flags=re.I,
                )
                if selected_match is not None and hasattr(memory, "audit_selection"):
                    selection_audit = memory.audit_selection(
                        observation.get("state") or json.dumps(observation, ensure_ascii=False),
                        selected_match.group(1).upper(),
                    )
                    info["dart_csl_selection_audit"] = selection_audit
                    turn_record["info"] = info
'''
    if original not in text:
        raise RuntimeError("Could not locate per-turn memory write block")
    runner.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    return True


def _ensure_runner_re_import(runner: Path) -> bool:
    text = runner.read_text(encoding="utf-8")
    if "import re\n" in text:
        return False
    anchor = "import os\n"
    if anchor not in text:
        raise RuntimeError("Could not locate runner imports")
    runner.write_text(text.replace(anchor, anchor + "import re\n", 1), encoding="utf-8")
    return True


def _ensure_transition_helper_import(runner: Path) -> bool:
    text = runner.read_text(encoding="utf-8")
    if RUNNER_TRANSITION_IMPORT_MARKER in text:
        return False
    anchor = "from memory.client import MemoryClient\n"
    if anchor not in text:
        raise RuntimeError("Could not locate MemoryArena memory client import")
    helper_import = f'''{anchor}
{RUNNER_TRANSITION_IMPORT_MARKER}
from examples.csl_memory_agent.csl_memory.csl_transition import (
    blocked_transition_result,
    enforce_required_transition,
)
'''
    runner.write_text(text.replace(anchor, helper_import, 1), encoding="utf-8")
    return True


def _register_observation_dedupe(runner: Path) -> bool:
    text = runner.read_text(encoding="utf-8")
    if RUNNER_DEDUPE_MARKER in text:
        return False
    original = '''            input_messages = _clone_api_messages(conversation)
            input_messages.append(_normalize_api_message("user", prompt))
'''
    replacement = f'''            input_messages = _clone_api_messages(conversation)
            {RUNNER_DEDUPE_MARKER}
            latest = input_messages[-1] if input_messages else None
            latest_content = latest.get("content") if isinstance(latest, dict) else None
            if latest_content != prompt:
                input_messages.append(_normalize_api_message("user", prompt))
'''
    if original not in text:
        raise RuntimeError("Could not locate runner prompt append block")
    runner.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    return True


def _register_memory_refresh_and_bounded_context(runner: Path) -> bool:
    text = runner.read_text(encoding="utf-8")
    if RUNNER_REFRESH_MARKER in text and RUNNER_BOUNDED_CONTEXT_MARKER in text:
        return False
    original = f'''            if memory is not None and not memory_injected:
                prompt = memory.wrap_user_prompt(prompt_source)
                memory_injected = True
            else:
                prompt = prompt_source

            input_messages = _clone_api_messages(conversation)
            {RUNNER_DEDUPE_MARKER}
'''
    replacement = f'''            dart_csl_turn = (
                memory is not None
                and getattr(args, "memory_system", None) == "dart_csl_vtcc"
            )
            if memory is not None and (dart_csl_turn or not memory_injected):
                {RUNNER_REFRESH_MARKER}
                prompt = memory.wrap_user_prompt(prompt_source)
                memory_injected = True
            else:
                prompt = prompt_source

            if dart_csl_turn:
                {RUNNER_BOUNDED_CONTEXT_MARKER}
                input_messages = _clone_api_messages(
                    build_conversation_start_messages(args.action_format)
                )
            else:
                input_messages = _clone_api_messages(conversation)
            {RUNNER_DEDUPE_MARKER}
'''
    if original not in text:
        raise RuntimeError("Could not locate MemoryArena memory-injection message block")
    runner.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    return True


def _register_fast_lite_goals(goal_module: Path, env_module: Path) -> bool:
    goal_text = goal_module.read_text(encoding="utf-8")
    env_text = env_module.read_text(encoding="utf-8")
    changed = False

    if FAST_GOALS_MARKER in env_text and "import os\n" not in env_text:
        future_import = "from __future__ import annotations\n"
        if future_import not in env_text:
            raise RuntimeError("Could not locate future import in WebAgentTextEnv")
        env_text = env_text.replace(future_import, future_import + "\nimport os\n", 1)
        env_module.write_text(env_text, encoding="utf-8")
        changed = True

    if FAST_GOALS_MARKER in goal_text and FAST_GOALS_MARKER in env_text:
        return changed

    if FAST_GOALS_MARKER not in goal_text:
        old_signature = "def get_synthetic_goals(all_products, product_prices):"
        new_signature = f'''{FAST_GOALS_MARKER}
def get_synthetic_goals(all_products, product_prices, max_goals=None):'''
        if old_signature not in goal_text:
            raise RuntimeError("Could not locate get_synthetic_goals")
        goal_text = goal_text.replace(old_signature, new_signature, 1)

        old_append = """        goals += product_goals
    for goal in goals:
"""
        new_append = """        goals += product_goals
        if max_goals is not None and len(goals) >= max_goals:
            goals = goals[:max_goals]
            break
    for goal in goals:
"""
        if old_append not in goal_text:
            raise RuntimeError("Could not locate synthetic goal accumulation")
        goal_text = goal_text.replace(old_append, new_append, 1)
        goal_module.write_text(goal_text, encoding="utf-8")
        changed = True

    if FAST_GOALS_MARKER not in env_text:
        old_call = "self.goals = get_goals(self.all_products, self.product_prices, human_goals)"
        new_call = f'''{FAST_GOALS_MARKER}
        fast_goal_limit = None
        if (
            not human_goals
            and limit_goals > 0
            and os.getenv("MEMORYARENA_FAST_LITE_GOALS") == "1"
        ):
            fast_goal_limit = limit_goals
        if fast_goal_limit is None:
            self.goals = get_goals(self.all_products, self.product_prices, human_goals)
        else:
            from ..engine.goal import get_synthetic_goals
            self.goals = get_synthetic_goals(
                self.all_products,
                self.product_prices,
                max_goals=fast_goal_limit,
            )'''
        if old_call not in env_text:
            raise RuntimeError("Could not locate SimServer goal initialization")
        env_text = env_text.replace(old_call, new_call, 1)
        if "import os\n" not in env_text:
            future_import = "from __future__ import annotations\n"
            env_text = env_text.replace(future_import, future_import + "\nimport os\n", 1)
        env_module.write_text(env_text, encoding="utf-8")
        changed = True

    return changed


def _register_answer_prompt(agent: Path) -> bool:
    text = agent.read_text(encoding="utf-8")
    if ANSWER_PROMPT_V7_MARKER in text:
        return False
    original = '''DEFAULT_SYSTEM_PROMPT = (
    "You are a WebShop shopping agent. "
    "Reply with exactly one action in the format search[...] or click[...]. "
    "Do not add explanations, JSON, or multiple actions."
)
'''
    replacement = f'''{ANSWER_PROMPT_MARKER}
{ANSWER_PROMPT_V4_MARKER}
{ANSWER_PROMPT_V5_MARKER}
{ANSWER_PROMPT_V6_MARKER}
{ANSWER_PROMPT_V7_MARKER}
DEFAULT_SYSTEM_PROMPT = (
    "You are a WebShop shopping agent. Reply with exactly one action in the "
    "format search[...] or click[...]. The final user message is the only current "
    "page; all earlier product ids, options, and clickables are history unless they "
    "also appear in that final message. A click[X] is legal only when the exact X "
    "is visible as a clickable in the final message. If the current page offers only "
    "Search, use search[...] and never click a stale id. Treat exact package count, "
    "size, color, and variant as part of product identity. Compare every listed option "
    "using the least sufficient visible evidence: an exact result-page price can support "
    "a price objective, while rating or missing identity fields require detail evidence. "
    "Open the selected exact candidate before Buy Now. Derive compatibility only from "
    "public relation rules and committed or provisional prior-scope summaries. The memory "
    "packet is non-binding evidence except for a section explicitly labeled Required evidence "
    "transition or Required state change. A Required state change lists a hard evidence goal, "
    "temporarily prohibited actions, and several admissible recovery actions. Choose any one "
    "currently legal recovery; the first is a recommendation, not the unique path. The obligation "
    "expires after the next environment observation and must then be recomputed. Candidate "
    "ids elsewhere in memory are historical unless the same id occurs in the final current "
    "page. Do not repeat zero-gain searches "
    "or clicks, and do not retry an unchanged blocked action. "
    "After a guard block, gather "
    "only the requested missing evidence, then return to the chosen exact candidate. "
    "Keep internal reasoning concise and always emit a final action. Do not add "
    "explanations, JSON, or multiple actions."
)
'''
    if ANSWER_PROMPT_MARKER in text:
        start = text.index(ANSWER_PROMPT_MARKER)
        end_anchor = "ACTION_PATTERN ="
        end = text.find(end_anchor, start)
        if end < 0:
            raise RuntimeError("Could not locate ACTION_PATTERN after registered WebShop prompt")
        updated = text[:start] + replacement + "\n" + text[end:]
        agent.write_text(updated, encoding="utf-8")
        return True
    if original not in text:
        raise RuntimeError("Could not locate WebShop default system prompt")
    agent.write_text(text.replace(original, replacement, 1), encoding="utf-8")
    return True


def _register_full_summary_unmatched_purchases(summary_build: Path) -> bool:
    text = summary_build.read_text(encoding="utf-8")
    if FULL_SUMMARY_MARKER in text:
        return False

    targets_anchor = '''    step_rewards = parsed.get("step_rewards") or []
    targets = parsed.get("targets") or []

    reward_lookup = {entry.get("idx"): entry for entry in step_rewards if entry.get("idx")}
    target_lookup = {entry.get("idx"): entry for entry in targets if entry.get("idx")}
'''
    targets_replacement = f'''    step_rewards = parsed.get("step_rewards") or []
    targets = parsed.get("targets") or []
    {FULL_SUMMARY_MARKER}
    unmatched = parsed.get("unmatched") or []

    reward_lookup = {{entry.get("idx"): entry for entry in step_rewards if entry.get("idx")}}
    target_lookup = {{entry.get("idx"): entry for entry in targets if entry.get("idx")}}
    unmatched_lookup = {{entry.get("idx"): entry for entry in unmatched if entry.get("idx")}}
'''
    if targets_anchor not in text:
        raise RuntimeError("Could not locate full-mode parsed-summary lookup block")
    text = text.replace(targets_anchor, targets_replacement, 1)

    result_anchor = '''        elif target_entry:
            purchased_asin = target_entry.get("purchased_asin")
            purchased_price = target_entry.get("purchased_price")
            reward = target_entry.get("reward")
'''
    result_replacement = '''        elif target_entry:
            purchased_asin = target_entry.get("purchased_asin")
            purchased_price = target_entry.get("purchased_price")
            reward = target_entry.get("reward")
        elif unmatched_lookup.get(step_num):
            purchase_entry = unmatched_lookup[step_num]
            purchased_asin = purchase_entry.get("asin")
            purchased_price = purchase_entry.get("price")
'''
    if result_anchor not in text:
        raise RuntimeError("Could not locate full-mode target result block")
    summary_build.write_text(text.replace(result_anchor, result_replacement, 1), encoding="utf-8")
    return True


if __name__ == "__main__":
    main()
