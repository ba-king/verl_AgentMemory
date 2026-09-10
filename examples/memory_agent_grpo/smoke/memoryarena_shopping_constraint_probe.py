"""Compare factual memory with the same facts plus explicit public-rule deductions."""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

from examples.memory_agent_grpo.adapters.memoryarena.shopping.executor_adapter import ShoppingToolExecutorAdapter
from examples.memory_agent_grpo.answer_agent.runtime.subtask_runner import AnswerAgentSubtaskRunner
from examples.memory_agent_grpo.smoke import memoryarena_shopping_diagnostic as base
from examples.memory_agent_grpo.smoke import memoryarena_shopping_prefix_probe as prefix

PROJECT = Path(__file__).parents[1]
SOURCE = PROJECT / "runs/shopping32_prefix_probe_20260904_r2/repeat3_factual"


def prepare_fixture():
    # Regenerate facts from public P1 observations, then verify the exact prior input.
    original = prefix.prepare_fixture(prefix.DEFAULT_SOURCE)
    factual = original["memories"]["factual"]
    prior = json.loads((SOURCE / "initial_input.json").read_text())
    assert prior == {**original["base_answer_input"], "retrieved_memories": factual}
    public = prior["subtask_input"]
    assert "Gel pairs well with Astringent." in public and "Gel avoids Milky, Rose." in public
    assert original["factual_memory_evidence"]["remaining_budget"] == "91.00"
    deduction = (
        "Derived constraints for current Product 2, from the observed previous purchase and public task rules: "
        "the previous cleanser category is Gel. Its compatible toner category is Astringent; "
        "its excluded toner categories are Milky and Rose. "
        "The remaining aggregate budget for all later products combined is $91.00 ($140.00 minus $49.00). "
        "A single later item priced above $91.00 cannot fit within the bundle budget."
    )
    return {
        "source_input_sha256": base._sha256(SOURCE / "initial_input.json"),
        "base_answer_input": original["base_answer_input"],
        "memories": {"factual": factual, "derived": [factual[0] + "\n\n" + deduction]},
        "added_text": deduction,
        "evidence": original["factual_memory_evidence"],
        "public_rules": ["Gel pairs well with Astringent.", "Gel avoids Milky, Rose."],
    }


def run_probe(args, recorder):
    recorder.add_secret_values([os.environ.get(args.answer_api_key_env, "")])
    fixture = prepare_fixture()
    recorder.write_json("fixture.json", fixture)
    root = base._expose_memoryarena(args.memoryarena_root)
    task_file = root / base.DEFAULT_TASK
    task = base._load_full_task(task_file)
    product2 = base.split_six_products(task)[1]
    catalogs = base._resolve_catalog_sources(memoryarena_root=root, catalog_sources=None)
    hashes = [{"path": str(p), "sha256": base._sha256(p)} for p in catalogs]
    original = json.loads((SOURCE.parent / "run_manifest.json").read_text())
    assert original["task_sha256"] == base._sha256(task_file) and original["catalogs"] == hashes
    schedule = [(r, a) for r in range(1, 4) for a in (("factual", "derived") if r % 2 else ("derived", "factual"))]
    recorder.write_json(
        "manifest.json",
        {
            "scope": "product2_initial_search_state_facts_vs_facts_plus_public_rule_deductions",
            "task_sha256": base._sha256(task_file),
            "catalogs": hashes,
            "schedule": schedule,
            "reset_seed": 0,
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "thinking": "disabled",
            "answer_max_tokens": 4096,
            "max_observation_chars": 16000,
            "max_decisions_per_run": 20,
            "maximum_answer_calls": 120,
            "memory_calls": 0,
            "reduced_catalog": True,
            "both_arms_use_current_executor": True,
            "manual_memory_intervention_bypasses_writer_and_retriever": True,
            "same_case_repeats_not_independent_tasks": True,
            "source_files_sha256": {
                str(p.relative_to(PROJECT)): base._sha256(p)
                for p in sorted(PROJECT.rglob("*.py"))
                if "runs" not in p.parts and "__pycache__" not in p.parts
            },
        },
    )
    if args.preflight_only:
        return {"prepared": True, "model_calls": 0}
    rows = []
    with tempfile.TemporaryDirectory(prefix="shopping-constraint-probe-") as temporary:
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
                    max_observation_chars=16000,
                    max_rounds=20,
                )
                rec, answer, _ = base._build_clients(args, args.output_dir / label)
                with factory.open_session(single_product_task=product2, product_ordinal=2) as session:
                    public = base.ShoppingTaskInputAdapter(max_observation_chars=16000).build_subtask_input(
                        session.reset(seed=0).to_dict(),
                    )
                    assert public.to_dict() == fixture["base_answer_input"]
                    subtask = replace(public, retrieved_memories=tuple(fixture["memories"][arm]))
                    rec.write_json("initial_input.json", subtask)
                    result = AnswerAgentSubtaskRunner(
                        client=answer,
                        executor=ShoppingToolExecutorAdapter(session=session),
                        config=base.AnswerAgentSubtaskRunnerConfig(
                            max_inference_steps=20,
                            max_tool_calls_per_inference=1,
                            max_consecutive_tool_failures=2,
                            interaction_mode=prefix.MODE,
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
                    rows.append(row)
    result = {"runs": rows, "memory_calls": 0}
    recorder.write_json("summary.json", result)
    return result


def main(argv=None):
    args = base.parse_args(argv)
    if (args.answer_max_tokens, args.max_observation_chars, args.max_inference_steps) != (4096, 16000, 20):
        raise ValueError("This fixed-source probe requires 4096/16000/20 settings.")
    if args.task_file or args.catalog_source:
        raise ValueError("This probe requires the original task and catalogs.")
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
