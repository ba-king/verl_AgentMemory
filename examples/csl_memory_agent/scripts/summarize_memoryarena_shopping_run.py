from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one complete official MemoryArena shopping item.")
    parser.add_argument("--task-result", required=True, type=Path)
    parser.add_argument("--reward-report", required=True, type=Path)
    parser.add_argument("--answer-usage-log", required=True, type=Path)
    parser.add_argument("--memory-state-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-name", default="memoryarena_task0_report")
    parser.add_argument("--report-title", default="Official MemoryArena Shopping Task 0 Report")
    parser.add_argument("--report-scope", default="single_official_memoryarena_item")
    parser.add_argument(
        "--scope-note",
        default="This is one complete official item, not the 150-item aggregate reported in the paper.",
    )
    args = parser.parse_args()

    task_result = _read_json(args.task_result)
    reward_report = _read_json(args.reward_report)
    answer_rows = _read_jsonl(args.answer_usage_log)
    memory_rows = [_read_json(path) for path in sorted(args.memory_state_root.rglob("metrics.json"))]

    expected_steps = int(task_result.get("total_steps") or 0)
    completed_steps = len(task_result.get("steps") or [])
    reward_item = _find_reward_item(reward_report, task_result.get("task_id"))
    reward_steps = reward_item.get("steps") or []
    hard_passes = sum(step.get("success") is True for step in reward_steps)
    reward_sum = sum(float(step.get("reward") or 0.0) for step in reward_steps)
    score_eligible = expected_steps > 0 and completed_steps == expected_steps and len(reward_steps) == expected_steps
    progress_score = (reward_sum / expected_steps) if score_eligible else None
    item_success = int(reward_item.get("item_success") is True) if score_eligible else None

    answer_usage = _sum_usage(answer_rows)
    memory_usage = _sum_memory_usage(memory_rows)
    metrics = {
        "scope": args.report_scope,
        "scope_note": args.scope_note,
        "task_id": task_result.get("task_id"),
        "expected_steps": expected_steps,
        "completed_steps": completed_steps,
        "scored_steps": len(reward_steps),
        "score_eligible": score_eligible,
        "hard_passed_steps": hard_passes,
        "hard_step_pass_rate": (hard_passes / expected_steps) if score_eligible else None,
        "PS_item": progress_score,
        "SR_item": item_success,
        "soft_average_reward": progress_score,
        "purchased_asins": [step.get("purchased_asin") for step in reward_steps],
        "target_asins": [step.get("target_asin") for step in reward_steps],
        "step_rewards": [step.get("reward") for step in reward_steps],
        "step_success": [step.get("success") for step in reward_steps],
        "token_scope": "answer_agent_plus_memory_agent; offline_reward_scorer_excluded",
        "tokens": {
            "answer_agent": answer_usage,
            "memory_agent": memory_usage,
            "combined_total_tokens": answer_usage["total_tokens"] + memory_usage["total_tokens"],
        },
        "official_reward_summary": reward_report.get("summary_results") or {},
        "source_files": {
            "task_result": str(args.task_result.resolve()),
            "reward_report": str(args.reward_report.resolve()),
            "answer_usage_log": str(args.answer_usage_log.resolve()),
            "memory_state_root": str(args.memory_state_root.resolve()),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.report_name}.json"
    md_path = args.output_dir / f"{args.report_name}.md"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(metrics, title=args.report_title), encoding="utf-8")
    print(json_path)
    print(md_path)


def _find_reward_item(report: dict[str, Any], task_id: str | None) -> dict[str, Any]:
    items = report.get("items") or []
    for item in items:
        if item.get("task_id") == task_id:
            return item
    return items[0] if items else {}


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    return {
        "calls": len(rows),
        "contract_repair_calls": sum(bool(row.get("contract_repair")) for row in rows),
        "transition_repair_calls": sum(bool(row.get("transition_repair")) for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        "latency_ms": sum(float(row.get("latency_ms") or 0.0) for row in rows),
    }


def _sum_memory_usage(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    metric_rows = [row.get("metrics") or {} for row in rows]
    return {
        "instances": len(metric_rows),
        "calls": sum(int(row.get("memory_llm_calls") or 0) for row in metric_rows),
        "prompt_tokens": sum(int(row.get("memory_llm_prompt_tokens") or 0) for row in metric_rows),
        "completion_tokens": sum(int(row.get("memory_llm_completion_tokens") or 0) for row in metric_rows),
        "total_tokens": sum(int(row.get("read_llm_tokens") or 0) for row in metric_rows),
        "latency_ms": sum(float(row.get("memory_llm_latency_ms") or 0.0) for row in metric_rows),
        "guard_cache_hits": sum(int(row.get("guard_audit_cache_hits") or 0) for row in metric_rows),
        "guard_llm_calls_avoided": sum(int(row.get("guard_llm_calls_avoided") or 0) for row in metric_rows),
        "transition_validation_calls": sum(
            int(row.get("transition_validation_calls") or 0) for row in metric_rows
        ),
        "transition_validation_blocks": sum(
            int(row.get("transition_validation_blocks") or 0) for row in metric_rows
        ),
        "transition_action_set_blocks": sum(
            int(row.get("transition_action_set_blocks") or 0) for row in metric_rows
        ),
        "transition_forbidden_action_blocks": sum(
            int(row.get("transition_forbidden_action_blocks") or 0) for row in metric_rows
        ),
        "transition_repeat_budget_blocks": sum(
            int(row.get("transition_repeat_budget_blocks") or 0) for row in metric_rows
        ),
        "no_progress_events": sum(int(row.get("no_progress_events") or 0) for row in metric_rows),
    }


def _to_markdown(metrics: dict[str, Any], *, title: str) -> str:
    tokens = metrics["tokens"]
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- Task: `{metrics['task_id']}`",
            f"- Complete: `{metrics['score_eligible']}` ({metrics['completed_steps']}/{metrics['expected_steps']} sessions)",
            f"- Hard passed sessions: `{metrics['hard_passed_steps']}/{metrics['expected_steps']}`",
            f"- Item PS (mean official step reward): `{metrics['PS_item']}`",
            f"- Item SR (all sessions receive reward 1): `{metrics['SR_item']}`",
            f"- Hard step pass rate (diagnostic): `{metrics['hard_step_pass_rate']}`",
            f"- Answer-agent tokens: `{tokens['answer_agent']['total_tokens']}`",
            f"- Transition repair calls: `{tokens['answer_agent']['transition_repair_calls']}`",
            f"- Memory-agent tokens: `{tokens['memory_agent']['total_tokens']}`",
            f"- Transition validations / blocks: `{tokens['memory_agent']['transition_validation_calls']}` / `{tokens['memory_agent']['transition_validation_blocks']}`",
            "- Transition block breakdown (action-set / forbidden / repeat-budget): "
            f"`{tokens['memory_agent']['transition_action_set_blocks']}` / "
            f"`{tokens['memory_agent']['transition_forbidden_action_blocks']}` / "
            f"`{tokens['memory_agent']['transition_repeat_budget_blocks']}`",
            f"- Combined tokens: `{tokens['combined_total_tokens']}`",
            "- Token scope: answer agent + memory agent; offline reward-scorer calls are excluded",
            "",
            str(metrics["scope_note"]),
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
