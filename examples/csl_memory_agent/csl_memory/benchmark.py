from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .csl_memory import DARTCSLMemorySystem


@dataclass
class BenchmarkCase:
    case_id: str
    history_chunks: list[str]
    final_query: str
    expected_terms: set[str] = field(default_factory=set)


@dataclass
class BenchmarkResult:
    case_id: str
    baseline: str
    prompt: str
    metrics: dict[str, Any]


class ToyCSLBenchmarkAdapter:
    """Small local benchmark shaped like MemoryArena's memory loop."""

    def load_cases(self) -> list[BenchmarkCase]:
        return [
            BenchmarkCase(
                case_id="abstract_final_decision",
                history_chunks=[
                    "Action: Inspect option A. Observation: Option A has attribute alpha, score 4.7, id A1.",
                    "Action: Select option A as the final choice. Observation: The choice was accepted.",
                ],
                final_query="Continue the task and choose the next compatible option. Prefer the highest score among valid options.",
                expected_terms={"option a", "alpha", "4.7", "final", "before final action"},
            ),
            BenchmarkCase(
                case_id="abstract_negative_constraint",
                history_chunks=[
                    "Action: Try candidate B. Observation: candidate B failed validation and should be avoided.",
                    "Action: Inspect candidate C. Observation: candidate C has status usable, score 8.1.",
                ],
                final_query="Select a usable candidate while avoiding prior failed states.",
                expected_terms={"candidate b", "failed", "avoid", "candidate c"},
            ),
        ]

    def run(self, baseline: str = "dart_csl_vtcc") -> list[BenchmarkResult]:
        results = []
        for case in self.load_cases():
            if baseline == "path_summary_only":
                mem = DARTCSLMemorySystem({"ablation": {"use_path_summary_only": True}}, user_id=case.case_id)
            elif baseline == "no_memory":
                prompt = f"<memory_context>\nNone\n</memory_context>\n\n{case.final_query}"
                results.append(
                    BenchmarkResult(
                        case_id=case.case_id,
                        baseline=baseline,
                        prompt=prompt,
                        metrics=_metrics(prompt, case.expected_terms),
                    )
                )
                continue
            else:
                mem = DARTCSLMemorySystem(user_id=case.case_id)
            for chunk in case.history_chunks:
                mem.add_chunk(chunk)
            prompt = mem.wrap_user_prompt(case.final_query)
            results.append(
                BenchmarkResult(
                    case_id=case.case_id,
                    baseline=baseline,
                    prompt=prompt,
                    metrics=_metrics(prompt, case.expected_terms),
                )
            )
        return results


def run_baselines(baselines: list[str] | None = None) -> list[dict[str, Any]]:
    adapter = ToyCSLBenchmarkAdapter()
    rows = []
    for baseline in baselines or ["no_memory", "path_summary_only", "dart_csl_vtcc"]:
        for result in adapter.run(baseline):
            rows.append(
                {
                    "case_id": result.case_id,
                    "baseline": result.baseline,
                    "metrics": result.metrics,
                }
            )
    return rows


def _metrics(prompt: str, expected_terms: set[str]) -> dict[str, Any]:
    lowered = prompt.lower()
    hits = sorted(term for term in expected_terms if term.lower() in lowered)
    recall = len(hits) / max(len(expected_terms), 1)
    return {
        "expected_term_recall": recall,
        "hits": hits,
        "num_expected_terms": len(expected_terms),
        "prompt_chars": len(prompt),
        "has_memory_context": "<memory_context>" in prompt,
        "has_soft_verifier": "[DART-CSL SOFT VERIFIER]" in prompt or "[CSL SOFT VERIFIER]" in prompt,
    }
