from examples.p2r_cmc.memory_system.eval.toy_memory_arena import ToyMemoryArenaAdapter, run_baselines


def test_toy_benchmark_runs_baselines_and_outputs_metrics():
    output = run_baselines(["no_memory", "p2r_cmc_prompt"])
    assert len(output) == 2
    assert all("metrics" in row for row in output)


def test_p2r_cmc_prompt_selects_correct_cluster():
    result = ToyMemoryArenaAdapter().run("p2r_cmc_prompt")[0]
    assert result.metrics["cluster_hit_rate"] > 0
    assert result.packet.evidence_items
