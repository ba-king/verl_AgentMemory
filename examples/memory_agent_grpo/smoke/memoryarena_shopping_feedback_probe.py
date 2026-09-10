"""Fixed failure-prefix intervention; only the fifth tool feedback differs.

Both arms use the current executor for all continuation actions. This is not
an end-to-end comparison of old and new executor policies.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

from examples.memory_agent_grpo.adapters.memoryarena.shopping.executor_adapter import ShoppingToolExecutorAdapter
from examples.memory_agent_grpo.answer_agent.input import AnswerAgentInitialMessage, AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.prompts.builder import build_answer_agent_initial_messages
from examples.memory_agent_grpo.answer_agent.response import AnswerAgentResponse, AnswerAgentToolCall
from examples.memory_agent_grpo.answer_agent.runtime.conversation import (
    AnswerAgentConversation,
    AnswerAgentToolExchange,
)
from examples.memory_agent_grpo.answer_agent.runtime.subtask_runner import AnswerAgentSubtaskRunner
from examples.memory_agent_grpo.answer_agent.tool_call_gateway import validate_tool_call_for_execution
from examples.memory_agent_grpo.answer_agent.tool_protocol.observation import ToolObservation, ToolObservationStatus
from examples.memory_agent_grpo.smoke import memoryarena_shopping_diagnostic as base

PROJECT = Path(__file__).parents[1]
SOURCE = PROJECT / "runs/shopping32_prefix_probe_20260904_r2/repeat3_factual"
MODE = base.AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL


def load_prefix():
    source = json.loads((SOURCE / "answer_trace.json").read_text())
    data = json.loads((SOURCE / "initial_input.json").read_text())
    subtask = AnswerAgentSubtaskInput(**{**data, "retrieved_memories": tuple(data["retrieved_memories"])})
    initial = tuple(AnswerAgentInitialMessage(**m) for m in source["conversation"]["initial_messages"])
    assert initial == build_answer_agent_initial_messages(subtask, interaction_mode=MODE)
    exchanges = []
    for step in source["steps"][:5]:
        response = AnswerAgentResponse(
            text=step["response"]["text"],
            tool_calls=tuple(AnswerAgentToolCall(**c) for c in step["response"]["tool_calls"]),
        )
        observations = []
        for outcome in step["tool_outcomes"]:
            observation = outcome["tool_observation"]
            observations.append(
                ToolObservation(
                    **{
                        **observation,
                        "status": ToolObservationStatus(observation["status"]),
                    }
                )
            )
        exchanges.append(AnswerAgentToolExchange(response, tuple(observations)))
    prefix = AnswerAgentConversation(initial, tuple(exchanges))
    assert len(exchanges) == 5
    assert json.loads(exchanges[3].assistant_response.tool_calls[0].arguments_json) == {"item": "Description"}
    assert json.loads(exchanges[4].assistant_response.tool_calls[0].arguments_json) == {"item": "Buy Now"}
    assert exchanges[3].tool_observations[0].content == exchanges[4].tool_observations[0].content
    return subtask, prefix


class PrefixClient:
    """Add frozen exchanges to every request without changing the production runner."""

    def __init__(self, client, prefix):
        self.client = client
        self.prefix = prefix

    def generate(self, conversation, tool_definitions):
        if conversation.initial_messages != self.prefix.initial_messages:
            raise ValueError("Continuation initial messages differ from the frozen source.")
        merged = AnswerAgentConversation(
            conversation.initial_messages,
            self.prefix.tool_exchanges + conversation.tool_exchanges,
        )
        return self.client.generate(merged, tool_definitions)


def replace_feedback(prefix, observation):
    exchange = replace(prefix.tool_exchanges[-1], tool_observations=(observation,))
    return replace(prefix, tool_exchanges=prefix.tool_exchanges[:-1] + (exchange,))


def run_probe(args, recorder):
    recorder.add_secret_values([os.environ.get(args.answer_api_key_env, "")])
    subtask, prefix = load_prefix()
    root = base._expose_memoryarena(args.memoryarena_root)
    task_file = root / base.DEFAULT_TASK
    task = base._load_full_task(task_file)
    catalogs = base._resolve_catalog_sources(memoryarena_root=root, catalog_sources=None)
    hashes = [{"path": str(p), "sha256": base._sha256(p)} for p in catalogs]
    source_manifest = json.loads((SOURCE.parent / "run_manifest.json").read_text())
    assert source_manifest["task_sha256"] == base._sha256(task_file)
    assert source_manifest["catalogs"] == hashes
    schedule = [(r, arm) for r in range(1, 4) for arm in (("old", "new") if r % 2 else ("new", "old"))]
    recorder.write_json(
        "manifest.json",
        {
            "scope": "first_invalid_click_feedback_intervention_at_fixed_product2_prefix",
            "source_trace": str(SOURCE / "answer_trace.json"),
            "source_trace_sha256": base._sha256(SOURCE / "answer_trace.json"),
            "source_input_sha256": base._sha256(SOURCE / "initial_input.json"),
            "task_sha256": base._sha256(task_file),
            "catalogs": hashes,
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "thinking": "disabled",
            "answer_max_tokens": args.answer_max_tokens,
            "max_observation_chars": args.max_observation_chars,
            "frozen_decisions": 5,
            "continuation_decision_budget": 15,
            "maximum_answer_calls": 90,
            "memory_calls": 0,
            "reset_seed": 0,
            "schedule": schedule,
            "both_arms_use_new_executor_for_continuation": True,
            "both_arms_start_with_four_environment_rounds": True,
            "both_arms_start_with_zero_continuation_failure_counter": True,
            "repeats_are_same_case_not_independent_tasks": True,
            "source_files_sha256": {
                str(p.relative_to(PROJECT)): base._sha256(p)
                for p in sorted(PROJECT.rglob("*.py"))
                if "runs" not in p.parts and "__pycache__" not in p.parts
            },
        },
    )
    recorder.write_json("source_prefix.json", prefix)
    if args.preflight_only:
        return {"prepared": True, "model_calls": 0}
    summaries = []
    with tempfile.TemporaryDirectory(prefix="shopping-feedback-probe-") as temporary:
        temp = Path(temporary)
        catalog = temp / "catalog.json"
        recorder.write_json(
            "catalog_summary.json",
            base.build_shopping_catalog_bundle(
                source_files=catalogs,
                destination=catalog,
                required_target_asins=base._preflight_required_targets(task),
            ),
        )
        with base._upstream_service(
            memoryarena_root=root,
            base_url=args.upstream_url,
            python_executable=root / ".venv-runtime/bin/python",
            items_file=catalog,
            startup_timeout_seconds=args.upstream_startup_timeout_seconds,
            api_key_env=args.answer_api_key_env,
            additional_secret_env_names=(args.memory_api_key_env,),
            upstream_log_path=args.output_dir / "upstream.log",
        ):
            for repeat, arm in schedule:
                label = f"repeat{repeat}_{arm}"
                work = temp / label
                work.mkdir()
                factory = base.ShoppingAnswerOnlySessionFactory(
                    working_directory=work,
                    env_server_base=args.upstream_url,
                    product_catalog_dir=root / "data/shopping/product_catalog",
                    domain_data_path=root / "data/shopping/domain_data.json",
                    timeout_seconds=args.environment_timeout_seconds,
                    max_observation_chars=args.max_observation_chars,
                    max_rounds=20,
                )
                rec, answer, _ = base._build_clients(args, args.output_dir / label)
                with factory.open_session(
                    single_product_task=base.split_six_products(task)[1], product_ordinal=2
                ) as session:
                    public = base.ShoppingTaskInputAdapter(
                        max_observation_chars=args.max_observation_chars
                    ).build_subtask_input(
                        session.reset(seed=0).to_dict(),
                    )
                    assert replace(public, retrieved_memories=subtask.retrieved_memories) == subtask
                    executor = ShoppingToolExecutorAdapter(session=session)
                    for index, exchange in enumerate(prefix.tool_exchanges):
                        call = exchange.assistant_response.tool_calls[0]
                        gateway = validate_tool_call_for_execution(call, tool_definitions=base.tool_definitions())
                        assert gateway.validated_call is not None
                        execution = executor.execute(gateway.validated_call)
                        rec.append_event("prefix_replay.jsonl", {"step": index + 1, "execution": execution})
                        if index < 4:
                            assert execution.tool_observation == exchange.tool_observations[0]
                        else:
                            assert execution.tool_observation.status is ToolObservationStatus.ERROR
                            assert "< prev" in execution.tool_observation.content
                            assert session._upstream_client.round_count == 4
                    frozen = prefix if arm == "old" else replace_feedback(prefix, execution.tool_observation)
                    rec.write_json("frozen_conversation.json", frozen)
                    rec.write_json("initial_input.json", subtask)
                    result = AnswerAgentSubtaskRunner(
                        client=PrefixClient(answer, frozen),
                        executor=executor,
                        config=base.AnswerAgentSubtaskRunnerConfig(
                            max_inference_steps=15,
                            max_tool_calls_per_inference=1,
                            max_consecutive_tool_failures=2,
                            interaction_mode=MODE,
                        ),
                    ).run(subtask, tool_definitions=base.tool_definitions())
                    rec.write_json("answer_trace.json", result)
                    rec.write_json("termination_evidence.json", session.termination_evidence)
                    row = {
                        "repeat": repeat,
                        "arm": arm,
                        "path": label,
                        "status": result.status.value,
                        "purchase_completed": session.termination_evidence.kind.value == "PURCHASE_COMPLETED",
                        "target_matched": session.termination_evidence.purchased_target_matched,
                        "answer_calls": answer.call_count,
                        "error_code": None if result.error_code is None else result.error_code.value,
                    }
                    rec.write_json("summary.json", row)
                    summaries.append(row)
    summary = {"runs": summaries, "memory_calls": 0}
    recorder.write_json("summary.json", summary)
    return summary


def main(argv=None):
    args = base.parse_args(argv)
    if (args.answer_max_tokens, args.max_observation_chars, args.max_inference_steps) != (4096, 16000, 20):
        raise ValueError("This fixed-source probe requires the original 4096/16000/20 settings.")
    if args.task_file or args.catalog_source:
        raise ValueError("This probe requires the fixed task and catalogs.")
    recorder = base.ArtifactRecorder(args.output_dir)
    try:
        with base._credentials(args), open(os.devnull, "w") as discarded:
            with redirect_stdout(discarded), redirect_stderr(discarded):
                result = run_probe(args, recorder)
    except Exception as error:
        recorder.write_json("failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        print(json.dumps({"error_type": type(error).__name__, "output_dir": str(args.output_dir)}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
