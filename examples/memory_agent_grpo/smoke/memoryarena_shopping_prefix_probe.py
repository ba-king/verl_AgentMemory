"""Compare memory inputs at task 32 Product 2 after one recorded Product 1.

This diagnostic reuses the recorded Memory policy outcome. It does not train
or rerun Memory, and the factual intervention bypasses retrieval intentionally.
Use --prepare-only to freeze inputs without environment or model calls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from examples.memory_agent_grpo.adapters.memoryarena.shopping.executor_adapter import ShoppingToolExecutorAdapter
from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.prompts.builder import build_answer_agent_initial_messages
from examples.memory_agent_grpo.answer_agent.runtime.subtask_runner import AnswerAgentSubtaskRunner
from examples.memory_agent_grpo.smoke import memoryarena_shopping_diagnostic as base

DEFAULT_SOURCE = Path(__file__).parents[1] / "runs/shopping32_paired_20260904_r1"
MODE = base.AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL


def public_pages(product: dict) -> list[str]:
    return [
        observation["content"]
        for event in product["interaction_events"]
        if event["event_type"] == "ENVIRONMENT_OBSERVATION"
        for observation in json.loads(event["content"])["tool_observations"]
        if observation["status"] == "success"
    ]


def factual_memory(product: dict) -> tuple[str, dict]:
    """Extract only public Product 1 observations; fail on unsupported evidence."""
    if product["product_ordinal"] != 1:
        raise ValueError("The fixed prefix must end at Product 1.")
    pages = public_pages(product)
    purchase = re.search(r"Purchase #1: ([A-Z0-9]{10}) \(\$(\d+\.\d+)\)", pages[-1])
    if purchase is None or "Purchases Made: 1" not in pages[-1]:
        raise ValueError("A public single-purchase receipt is required.")
    asin, price = purchase.groups()
    search_name = None
    for page in pages[:-1]:
        parts = page.split(" [SEP] ")
        if asin in parts:
            position = parts.index(asin)
            if position + 2 < len(parts):
                search_name = parts[position + 1]
    if search_name is None:
        raise ValueError("Purchased ASIN must have a name in the observed search results.")
    detail = next((p for p in pages if search_name in p and " [SEP] Price: $" in p), None)
    if detail is None:
        raise ValueError("The purchased item's detail page must be observed.")
    detail_price = re.search(r"\[SEP\] Price: \$(\d+(?:\.\d+)?)", detail)
    rating = re.search(r"\[SEP\] Rating: (\d+(?:\.\d+)?)", detail)
    budget = re.search(r"must not exceed \$(\d+(?:\.\d+)?)", product["memory_task_input"])
    if detail_price is None or rating is None or budget is None:
        raise ValueError("Missing public price, rating, or budget evidence.")
    if Decimal(detail_price.group(1)) != Decimal(price):
        raise ValueError("Detail and receipt prices disagree.")
    if "Gel Cleanser" not in search_name:
        raise ValueError("This task-32 probe requires the observed Gel Cleanser prefix.")
    remaining = Decimal(budget.group(1)) - Decimal(price)
    content = (
        f"Observed purchase history, Product 1 (Cleanser): purchased {search_name}. "
        f"ASIN: {asin}. Paid: ${Decimal(price):.2f}. Observed rating: {rating.group(1)}. "
        "The observed product title identifies it as a Gel Cleanser. "
        f"Cumulative actual spend through Product 1: ${Decimal(price):.2f}. "
        f"The public total budget is ${Decimal(budget.group(1)):.2f}; "
        f"remaining budget for all later products combined is ${remaining:.2f}."
    )
    return content, {
        "source": "Product 1 public search page, detail page, purchase receipt, and public budget",
        "asin": asin,
        "name": search_name,
        "paid": price,
        "rating": rating.group(1),
        "remaining_budget": str(remaining),
        "derived_fields": {"Gel Cleanser": "literal observed title", "remaining_budget": "budget minus receipt price"},
        "public_pages": pages,
    }


def prepare_fixture(source: Path) -> dict:
    products_file = source / "memory/products.jsonl"
    # Read the fixed prefix and the already-recorded next input only. No later
    # product, evaluator result, reward, or target label is used for the memory.
    with products_file.open() as stream:
        product1 = json.loads(next(stream))
        product2 = json.loads(next(stream))
    if product2["product_ordinal"] != 2:
        raise ValueError("Expected Product 2 after the prefix.")
    content, evidence = factual_memory(product1)
    current = product2["answer_input"]["retrieved_memories"]
    if product1["bank_after"]["records"] or current:
        raise ValueError("This probe is scoped to the recorded null-memory prefix.")
    post_outputs = [decision["raw_output"] for decision in product1["post_task_decisions"]]
    if not post_outputs or any(json.loads(raw) != {"memory_action": None} for raw in post_outputs):
        raise ValueError("Recorded Product 1 Memory output must be null.")
    fixture = {
        "source_run": str(source.resolve()),
        "source_products_sha256": base._sha256(products_file),
        "prefix_product_ordinal": 1,
        "evaluated_product_ordinal": 2,
        "prefix_memory_raw_outputs": post_outputs,
        "factual_memory_evidence": evidence,
        "base_answer_input": {**product2["answer_input"], "retrieved_memories": []},
        "memories": {"empty": [], "current": current, "factual": [content]},
        "execution_aliases": {"empty": "empty", "current": "empty", "factual": "factual"},
        "current_is_recorded_policy_replay": True,
        "factual_is_manual_evidence_intervention": True,
        "factual_bypasses_memory_writer_and_retriever": True,
        "hidden_target_or_evaluator_feedback_in_memory": False,
    }
    fixture["initial_messages"] = {
        arm: [
            asdict(message)
            for message in build_answer_agent_initial_messages(
                AnswerAgentSubtaskInput(**{**fixture["base_answer_input"], "retrieved_memories": tuple(memories)}),
                interaction_mode=MODE,
            )
        ]
        for arm, memories in fixture["memories"].items()
    }
    if fixture["initial_messages"]["empty"] != fixture["initial_messages"]["current"]:
        raise ValueError("Cannot alias arms with different model inputs.")
    return fixture


def planned_runs(repeats: int) -> list[dict]:
    return [
        {"repeat": repeat, "arm": arm}
        for repeat in range(1, repeats + 1)
        for arm in (("empty", "factual") if repeat % 2 else ("factual", "empty"))
    ]


def build_probe_input(initial_observation, fixture: dict, arm: str, max_observation_chars: int):
    public_input = base.ShoppingTaskInputAdapter(
        max_observation_chars=max_observation_chars,
    ).build_subtask_input(initial_observation.to_dict())
    if public_input.to_dict() != fixture["base_answer_input"]:
        raise ValueError("Product 2 public input differs from the frozen prefix input.")
    return replace(public_input, retrieved_memories=tuple(fixture["memories"][arm]))


def run_probe(args, recorder):
    recorder.add_secret_values(
        [os.environ.get(args.answer_api_key_env, ""), os.environ.get(args.memory_api_key_env, "")]
    )
    fixture = prepare_fixture(args.source_run)
    recorder.write_json("fixture.json", fixture)
    root = base._expose_memoryarena(args.memoryarena_root)
    task_file = (root / base.DEFAULT_TASK if args.task_file is None else args.task_file).resolve()
    source_manifest = json.loads((args.source_run / "run_manifest.json").read_text())
    if base._sha256(task_file) != source_manifest["task_sha256"]:
        raise ValueError("Task changed since the recorded prefix.")
    task = base._load_full_task(task_file)
    product2 = base.split_six_products(task)[1]
    catalogs = base._resolve_catalog_sources(memoryarena_root=root, catalog_sources=None)
    catalog_hashes = [{"path": str(path), "sha256": base._sha256(path)} for path in catalogs]
    if catalog_hashes != source_manifest["catalogs"]:
        raise ValueError("Catalogs changed since the recorded prefix.")
    schedule = planned_runs(args.repeats)
    recorder.write_json(
        "run_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "task32_product1_to_product2_fixed_prefix_memory_input_intervention",
            "task_sha256": base._sha256(task_file),
            "catalogs": catalog_hashes,
            "reduced_catalog": True,
            "schedule": schedule,
            "reset_seed": 0,
            "requested_arms": ["empty", "current", "factual"],
            "execution_aliases": fixture["execution_aliases"],
            "memory_api_calls_planned": 0,
            "maximum_answer_api_calls": len(schedule) * args.max_inference_steps,
            "answer_max_tokens": args.answer_max_tokens,
            "temperature": 0,
            "thinking": "disabled",
            "max_observation_chars": args.max_observation_chars,
            "max_inference_steps": args.max_inference_steps,
            "repeats_are_same_case_not_independent_tasks": True,
            "source_files_sha256": {
                str(path.relative_to(Path(__file__).parents[1])): base._sha256(path)
                for path in sorted(Path(__file__).parents[1].rglob("*.py"))
                if "runs" not in path.parts and "__pycache__" not in path.parts
            },
        },
    )
    if args.prepare_only:
        return {"prepared": True, "environment_calls": 0, "model_calls": 0, "schedule": schedule}
    summaries = []
    with tempfile.TemporaryDirectory(prefix="shopping-prefix-probe-") as temporary:
        temporary_root = Path(temporary)
        catalog_file = temporary_root / "catalog.json"
        recorder.write_json(
            "catalog_summary.json",
            base.build_shopping_catalog_bundle(
                source_files=catalogs,
                destination=catalog_file,
                required_target_asins=base._preflight_required_targets(task),
            ),
        )
        with base._upstream_service(
            memoryarena_root=root,
            base_url=args.upstream_url,
            python_executable=root / ".venv-runtime/bin/python",
            items_file=catalog_file,
            startup_timeout_seconds=args.upstream_startup_timeout_seconds,
            api_key_env=args.answer_api_key_env,
            additional_secret_env_names=(args.memory_api_key_env,),
            upstream_log_path=args.output_dir / "upstream.log",
        ):
            for item in schedule:
                label = f"repeat{item['repeat']}_{item['arm']}"
                work = temporary_root / label
                work.mkdir()
                factory = base.ShoppingAnswerOnlySessionFactory(
                    working_directory=work,
                    env_server_base=args.upstream_url,
                    product_catalog_dir=root / "data/shopping/product_catalog",
                    domain_data_path=root / "data/shopping/domain_data.json",
                    timeout_seconds=args.environment_timeout_seconds,
                    max_observation_chars=args.max_observation_chars,
                    max_rounds=args.max_inference_steps,
                )
                arm_recorder, answer, _ = base._build_clients(args, args.output_dir / label)
                with factory.open_session(single_product_task=product2, product_ordinal=2) as session:
                    answer_input = build_probe_input(
                        session.reset(seed=0),
                        fixture,
                        item["arm"],
                        args.max_observation_chars,
                    )
                    arm_recorder.write_json("initial_input.json", answer_input)
                    result = AnswerAgentSubtaskRunner(
                        client=answer,
                        executor=ShoppingToolExecutorAdapter(session=session),
                        config=base.AnswerAgentSubtaskRunnerConfig(
                            max_inference_steps=args.max_inference_steps,
                            max_tool_calls_per_inference=1,
                            max_consecutive_tool_failures=2,
                            interaction_mode=MODE,
                        ),
                    ).run(answer_input, tool_definitions=base.tool_definitions())
                    arm_recorder.write_json("answer_trace.json", result)
                    arm_recorder.write_json("termination_evidence.json", session.termination_evidence)
                    summary = {
                        **item,
                        "path": label,
                        "runner_status": result.status.value,
                        "purchase_completed": session.termination_evidence.kind.value == "PURCHASE_COMPLETED",
                        "target_matched": session.termination_evidence.purchased_target_matched,
                        "answer_calls": answer.call_count,
                        "error_code": None if result.error_code is None else result.error_code.value,
                    }
                    arm_recorder.write_json("summary.json", summary)
                    recorder.append_event("progress.jsonl", summary)
                    summaries.append(summary)
    result = {"runs": summaries, "execution_aliases": fixture["execution_aliases"], "memory_api_calls": 0}
    recorder.write_json("summary.json", result)
    return result


def parse_args(argv=None):
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    extra.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    extra.add_argument("--prepare-only", action="store_true")
    additional, remaining = extra.parse_known_args(argv)
    if additional.prepare_only:
        remaining.append("--preflight-only")
    args = base.parse_args(remaining)
    if args.preflight_only and not additional.prepare_only:
        extra.error("Use --prepare-only for this probe's offline preparation.")
    if args.catalog_source is not None:
        extra.error("This probe freezes the source run's two catalogs; custom catalogs are unsupported.")
    args.source_run = additional.source_run.expanduser().resolve()
    args.repeats = additional.repeats
    args.prepare_only = additional.prepare_only
    return args


def main(argv=None):
    args = parse_args(argv)
    recorder = base.ArtifactRecorder(args.output_dir)
    try:
        with base._credentials(args), open(os.devnull, "w") as discarded:
            with redirect_stdout(discarded), redirect_stderr(discarded):
                result = run_probe(args, recorder)
    except Exception as error:
        recorder.write_json(
            "failure.json",
            {"error_code": base._error_code(error), "message": str(error), "traceback": traceback.format_exc()},
        )
        print(json.dumps({"output_dir": str(args.output_dir), "error_code": base._error_code(error)}))
        return 1
    print(json.dumps({"output_dir": str(args.output_dir), **result}, ensure_ascii=False, indent=2))
    return 0 if args.prepare_only or all(row["purchase_completed"] for row in result["runs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
