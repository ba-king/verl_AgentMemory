"""Persist one matched Shopping diagnostic, including rejected Memory outputs.

This is a separate opt-in diagnostic entry; the content-free smoke commands
retain their existing behavior. Stored trajectories contain benchmark and
model content and are intended for local analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from examples.memory_agent_grpo.adapters.memoryarena.shopping.answer_only_episode_runtime import (
    ShoppingAnswerOnlyEpisodeEvaluationPolicy,
    ShoppingAnswerOnlyEpisodeRunner,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.answer_only_session_factory import (
    ShoppingAnswerOnlySessionFactory,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.catalog_bundle import build_shopping_catalog_bundle
from examples.memory_agent_grpo.adapters.memoryarena.shopping.memory_episode_runtime import ShoppingMemoryEpisodeRunner
from examples.memory_agent_grpo.adapters.memoryarena.shopping.memory_episode_validation import split_six_products
from examples.memory_agent_grpo.adapters.memoryarena.shopping.task_input_adapter import ShoppingTaskInputAdapter
from examples.memory_agent_grpo.adapters.memoryarena.shopping.tool_catalog import tool_definitions
from examples.memory_agent_grpo.answer_agent.interaction_mode import AnswerAgentInteractionMode
from examples.memory_agent_grpo.answer_agent.providers.deepseek import DeepSeekAnswerAgentClient, DeepSeekToolChoice
from examples.memory_agent_grpo.answer_agent.runtime import AnswerAgentSubtaskRunnerConfig
from examples.memory_agent_grpo.diagnostics import ArtifactRecorder, RecordingHttpTransport
from examples.memory_agent_grpo.memory_agent.decision.model_client import OpenAICompatibleMemoryAgentClient
from examples.memory_agent_grpo.model_io.http_transport import UrllibHttpTransport
from examples.memory_agent_grpo.smoke.deepseek_action_smoke import (
    DeepSeekV4ConservativeTokenCounter,
    build_deepseek_smoke_client,
)
from examples.memory_agent_grpo.smoke.deepseek_answer_agent_initial_smoke import DeepSeekAnswerConservativeTokenCounter
from examples.memory_agent_grpo.smoke.memoryarena_shopping_answer_only_episode_smoke import (
    _preflight_required_targets,
    _resolve_catalog_sources,
)
from examples.memory_agent_grpo.smoke.memoryarena_shopping_smoke import (
    DEFAULT_ANSWER_TOOL_CHOICE,
    DEFAULT_MEMORYARENA_ROOT,
    _build_deepseek_client,
    _expose_memoryarena,
    _load_full_task,
    _upstream_service,
)

DEFAULT_TASK = Path("benchmark_data/fixed100_v2/shopping_tasks/032_beauty_item_2.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    return getattr(code, "value", type(error).__name__)


class ObservedMemoryClient:
    """Record attempts even when provider validation rejects a completion."""

    def __init__(self, client, recorder: ArtifactRecorder):
        self.client = client
        self.recorder = recorder
        self.attempts: list[dict] = []

    def generate(self, model_input):
        attempt = {
            "attempt": len(self.attempts) + 1,
            "correction": "<correction_notice>" in model_input[1].content,
            "status": "started",
        }
        self.attempts.append(attempt)
        self.recorder.append_event(
            "memory_attempts.jsonl",
            {
                **attempt,
                "model_input": model_input,
            },
        )
        try:
            output = self.client.generate(model_input)
        except Exception as error:
            attempt.update(status="model_call_failed", error_code=_error_code(error))
            self.recorder.append_event("memory_attempts.jsonl", attempt)
            raise
        self.recorder.append_event(
            "memory_attempts.jsonl",
            {
                **attempt,
                "status": "output_received",
                "raw_output": output,
            },
        )
        return output

    def observe_decision(self, trace):
        validation = trace.gateway_result.validation
        raw_validation = validation if trace.output_repair is None else trace.output_repair.original_validation
        attempt = self.attempts[-1]
        attempt.update(
            status=("repaired" if trace.output_repair is not None else "valid") if validation.is_valid else "rejected",
            raw_output_valid=raw_validation.is_valid,
            raw_error_code=None if raw_validation.error_code is None else raw_validation.error_code.value,
            output_repair_rule=None if trace.output_repair is None else trace.output_repair.rule,
            phase=trace.context.phase.value,
            error_code=None if validation.error_code is None else validation.error_code.value,
        )
        self.recorder.append_event(
            "memory_decisions.jsonl",
            {
                **attempt,
                "trace": trace,
            },
        )


class ObservedAnswerClient:
    def __init__(self, client, recorder: ArtifactRecorder):
        self.client = client
        self.recorder = recorder
        self.call_count = 0

    def generate(self, conversation, tool_definitions=()):
        self.call_count += 1
        # This also preserves the most recent environment observation when a
        # local context-budget check prevents an HTTP request from being sent.
        self.recorder.append_event(
            "answer_inputs.jsonl",
            {
                "call": self.call_count,
                "conversation": conversation,
            },
        )
        try:
            return self.client.generate(conversation, tool_definitions)
        except Exception as error:
            self.recorder.append_event(
                "answer_errors.jsonl",
                {
                    "call": self.call_count,
                    "error_code": _error_code(error),
                },
            )
            raise


def summarize_memory_attempts(attempts: list[dict]) -> dict:
    first = [item for item in attempts if not item["correction"]]
    corrections = [item for item in attempts if item["correction"]]
    errors: dict[str, int] = {}
    for item in attempts:
        code = item.get("raw_error_code") or item.get("error_code")
        if code:
            errors[code] = errors.get(code, 0) + 1
    return {
        "first_attempt_count": len(first),
        "first_attempt_valid_count": sum(item["status"] == "valid" for item in first),
        "correction_attempt_count": len(corrections),
        "correction_valid_count": sum(item["status"] == "valid" for item in corrections),
        "deterministic_repair_count": sum(item["status"] == "repaired" for item in attempts),
        "first_attempt_accepted_count": sum(item["status"] in ("valid", "repaired") for item in first),
        "correction_accepted_count": sum(item["status"] in ("valid", "repaired") for item in corrections),
        "error_counts": errors,
        "attempts": attempts,
    }


def summarize_answer_episode(result) -> dict:
    products = result.product_results
    return {
        **result.to_dict(),
        # Answer-only's native COMPLETED requires correct purchases; normalize
        # the structural metric before comparing it with the Memory runner.
        "pipeline_completed": len(products) == 6
        and all(
            item.answer_result.status.value == "COMPLETED"
            and item.terminal_tool_execution_count == 1
            and item.termination_evidence.kind.value != "NOT_TERMINATED"
            for item in products
        ),
        "target_match_count": sum(item.termination_evidence.purchased_target_matched is True for item in products),
        "purchase_completed_count": sum(
            item.termination_evidence.kind.value == "PURCHASE_COMPLETED" for item in products
        ),
    }


@contextmanager
def _credentials(args):
    names = (args.answer_api_key_env, args.memory_api_key_env)
    previous = {name: os.environ.get(name) for name in names}
    try:
        if args.api_key_file is not None:
            key = args.api_key_file.read_text(encoding="utf-8").strip()
            if not key or any(character.isspace() for character in key):
                raise ValueError("API key file must contain one non-empty key.")
            for name in names:
                os.environ[name] = key
        if not args.preflight_only:
            required = names if args.mode in {"memory", "paired"} else names[:1]
            if not all(os.environ.get(name) for name in required):
                raise ValueError(
                    "Missing DeepSeek API key; supply --api-key-file or configured key environment variables."
                )
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _build_clients(args, directory: Path):
    recorder = ArtifactRecorder(
        directory,
        secret_values=[
            os.environ.get(args.answer_api_key_env, ""),
            os.environ.get(args.memory_api_key_env, ""),
        ],
    )
    answer_config = _build_deepseek_client(
        api_key_env=args.answer_api_key_env,
        answer_max_tokens=args.answer_max_tokens,
        answer_tool_choice=getattr(args, "answer_tool_choice", DEFAULT_ANSWER_TOOL_CHOICE),
    ).config
    memory_config = replace(
        build_deepseek_smoke_client(api_key_env=args.memory_api_key_env).config,
        max_tokens=args.memory_max_tokens,
    )
    answer = ObservedAnswerClient(
        DeepSeekAnswerAgentClient(
            answer_config,
            token_counter=DeepSeekAnswerConservativeTokenCounter(),
            transport=RecordingHttpTransport(UrllibHttpTransport(), directory, "answer", recorder=recorder),
        ),
        recorder,
    )
    memory = ObservedMemoryClient(
        OpenAICompatibleMemoryAgentClient(
            memory_config,
            token_counter=DeepSeekV4ConservativeTokenCounter(),
            transport=RecordingHttpTransport(UrllibHttpTransport(), directory, "memory", recorder=recorder),
        ),
        recorder,
    )
    recorder.write_json("client_config.json", {"answer": answer_config, "memory": memory_config})
    return recorder, answer, memory


def run_diagnostic(args, recorder: ArtifactRecorder) -> dict:
    recorder.add_secret_values(
        [
            os.environ.get(args.answer_api_key_env, ""),
            os.environ.get(args.memory_api_key_env, ""),
        ]
    )
    root = _expose_memoryarena(args.memoryarena_root)
    task_file = (root / DEFAULT_TASK if args.task_file is None else args.task_file).expanduser().resolve()
    task = _load_full_task(task_file)
    targets = _preflight_required_targets(task)
    single_tasks = split_six_products(task)
    catalogs = _resolve_catalog_sources(
        memoryarena_root=root,
        catalog_sources=None if args.catalog_source is None else tuple(args.catalog_source),
    )
    runtime_python = root / ".venv-runtime" / "bin" / "python"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "preflight_only": args.preflight_only,
        "task_file": str(task_file),
        "task_sha256": _sha256(task_file),
        "source_files_sha256": {
            str(path.relative_to(Path(__file__).parents[1])): _sha256(path)
            for path in sorted(Path(__file__).parents[1].rglob("*.py"))
            if "runs" not in path.parts and "__pycache__" not in path.parts
        },
        "catalogs": [{"path": str(path), "sha256": _sha256(path)} for path in catalogs],
        "reduced_catalog": True,
        "evaluation_scope": "single_task_pipeline_diagnostic",
        "answer_max_tokens": args.answer_max_tokens,
        "answer_tool_choice": getattr(args, "answer_tool_choice", DEFAULT_ANSWER_TOOL_CHOICE).value,
        "memory_max_tokens": args.memory_max_tokens,
        "temperature": 0,
        "thinking": "disabled",
        "max_observation_chars": args.max_observation_chars,
        "max_inference_steps_per_product": args.max_inference_steps,
        "memory_correction_budget_per_episode": args.memory_output_correction_budget,
        "reset_seed": 0,
        "expected_products": 6,
        "answer_history_reset_per_product": True,
        "bank_reset_per_episode": True,
        "evaluator_feedback_exposed_to_models": False,
        "comparison_note": (
            "Same current Answer runner and catalog; not directly comparable to historical full-catalog baselines."
        ),
    }
    recorder.write_json("run_manifest.json", manifest)
    summaries: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="shopping-diagnostic-") as temporary:
        temporary_root = Path(temporary)
        catalog_file = temporary_root / "catalog.json"
        catalog_summary = build_shopping_catalog_bundle(
            source_files=catalogs,
            destination=catalog_file,
            required_target_asins=targets,
        )
        recorder.write_json("catalog_summary.json", catalog_summary)
        with _upstream_service(
            memoryarena_root=root,
            base_url=args.upstream_url,
            python_executable=runtime_python,
            items_file=catalog_file,
            startup_timeout_seconds=args.upstream_startup_timeout_seconds,
            api_key_env=args.answer_api_key_env,
            additional_secret_env_names=(args.memory_api_key_env,),
            upstream_log_path=args.output_dir / "upstream.log",
        ):
            modes = (
                ("preflight",)
                if args.preflight_only
                else (("answer_only", "memory") if args.mode == "paired" else (args.mode,))
            )
            for mode in modes:
                task_directory = temporary_root / mode
                task_directory.mkdir()
                factory = ShoppingAnswerOnlySessionFactory(
                    working_directory=task_directory,
                    env_server_base=args.upstream_url,
                    product_catalog_dir=root / "data" / "shopping" / "product_catalog",
                    domain_data_path=root / "data" / "shopping" / "domain_data.json",
                    timeout_seconds=args.environment_timeout_seconds,
                    max_observation_chars=args.max_observation_chars,
                    max_rounds=args.max_inference_steps,
                )
                if mode == "preflight":
                    for ordinal, single_task in enumerate(single_tasks, 1):
                        with factory.open_session(single_product_task=single_task, product_ordinal=ordinal) as session:
                            observation = session.reset(seed=0)
                            recorder.append_event(
                                "preflight_observations.jsonl",
                                {
                                    "product_ordinal": ordinal,
                                    "public_observation": observation,
                                },
                            )
                    summaries[mode] = {"environment_reset_count": 6, "model_calls": 0, "passed": True}
                    continue
                arm_recorder, answer, memory = _build_clients(args, args.output_dir / mode)
                config = AnswerAgentSubtaskRunnerConfig(
                    max_inference_steps=args.max_inference_steps,
                    max_tool_calls_per_inference=1,
                    max_consecutive_tool_failures=2,
                    interaction_mode=AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL,
                )
                shared = dict(
                    answer_client=answer,
                    session_factory=factory,
                    task_input_adapter=ShoppingTaskInputAdapter(max_observation_chars=args.max_observation_chars),
                    tool_definitions=tool_definitions(),
                    config=config,
                )
                if mode == "memory":
                    runner = ShoppingMemoryEpisodeRunner(
                        **shared,
                        memory_client=memory,
                        memory_output_correction_budget=args.memory_output_correction_budget,
                        memory_decision_observer=memory.observe_decision,
                        product_observer=lambda trace, recorder=arm_recorder: recorder.append_event(
                            "products.jsonl", trace
                        ),
                    )
                    result = runner.run(task)
                    summary = {
                        **result.to_safe_dict(),
                        "protocol_diagnostics": summarize_memory_attempts(memory.attempts),
                    }
                else:
                    runner = ShoppingAnswerOnlyEpisodeRunner(
                        **shared,
                        evaluation_policy=ShoppingAnswerOnlyEpisodeEvaluationPolicy.CONTINUE_EVALUATION_FAILURES,
                    )
                    result = runner.run(task)
                    summary = summarize_answer_episode(result)
                # asdict preserves the complete nested traces; to_dict on some
                # result classes intentionally emits only a safe summary.
                arm_recorder.write_json("episode_trace.json", asdict(result))
                arm_recorder.write_json("summary.json", summary)
                summaries[mode] = summary
                recorder.append_event("progress.jsonl", {"arm": mode, "summary": summary})
    output = {"preflight_only": args.preflight_only, "saved_trajectory": True, "arms": summaries}
    recorder.write_json("summary.json", output)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("paired", "memory", "answer_only"), default="paired")
    parser.add_argument("--memoryarena-root", type=Path, default=DEFAULT_MEMORYARENA_ROOT)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--catalog-source", type=Path, action="append")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--answer-api-key-env", default="P2R_ANSWER_API_KEY")
    parser.add_argument("--memory-api-key-env", default="P2R_MEMORY_API_KEY")
    parser.add_argument("--answer-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--answer-tool-choice",
        type=DeepSeekToolChoice,
        choices=tuple(choice.value for choice in DeepSeekToolChoice),
        default=DEFAULT_ANSWER_TOOL_CHOICE,
    )
    parser.add_argument("--memory-max-tokens", type=int, default=4096)
    parser.add_argument("--max-observation-chars", type=int, default=16000)
    parser.add_argument("--max-inference-steps", type=int, default=20)
    parser.add_argument(
        "--memory-output-correction-budget",
        type=int,
        choices=(0, 1, 2, 3),
        default=3,
        help="Episode-wide correction limit; each Memory decision is retried at most once.",
    )
    parser.add_argument("--upstream-url", default="http://127.0.0.1:36044")
    parser.add_argument("--upstream-startup-timeout-seconds", type=float, default=180)
    parser.add_argument("--environment-timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--preflight-only", action="store_true", help="Reset all six real environments without model calls."
    )
    parser.add_argument("--confirm-live-api", action="store_true")
    args = parser.parse_args(argv)
    for field in ("answer_max_tokens", "memory_max_tokens", "max_observation_chars", "max_inference_steps"):
        if getattr(args, field) <= 0:
            parser.error(f"{field} must be positive")
    if not args.preflight_only and not args.confirm_live_api:
        parser.error("--confirm-live-api is required for model calls")
    args.output_dir = args.output_dir.expanduser().absolute()
    if args.output_dir.exists():
        parser.error("output-dir must be a new path; existing runs cannot be overwritten")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    recorder = ArtifactRecorder(args.output_dir)
    try:
        with _credentials(args), open(os.devnull, "w") as discarded:
            with redirect_stdout(discarded), redirect_stderr(discarded):
                summary = run_diagnostic(args, recorder)
    except Exception as error:
        failure = {"status": "failed", "error_code": _error_code(error), "exception_type": type(error).__name__}
        recorder.write_json(
            "failure.json",
            {
                **failure,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        print(json.dumps({**failure, "output_dir": str(args.output_dir)}))
        return 1
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, ensure_ascii=False, indent=2))
    return 0 if all(arm.get("pipeline_completed", arm.get("passed", False)) for arm in summary["arms"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
