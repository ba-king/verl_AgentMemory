from examples.p2r_cmc.memory_system.clustering.maintenance_trigger import (
    MaintenanceTriggerPolicy,
    run_cluster_maintenance_if_needed,
)
from examples.p2r_cmc.memory_system.controller.remember_pipeline import RememberPipeline
from examples.p2r_cmc.memory_system.schemas import ActionTrace, RawEpisode
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


def test_trigger_runs_for_dirty_cluster_and_wrapper_executes():
    store = SQLiteMemoryStore()
    RememberPipeline(store).ingest_episode(
        RawEpisode(
            user_id="u1",
            project_id="p1",
            content="Remember: I prefer quiet hotels near public transit.",
        )
    )
    cluster = store.list_clusters(user_id="u1", project_id="p1")[0]
    assert cluster.dirty is True

    result = run_cluster_maintenance_if_needed(store, user_id="u1", project_id="p1")
    refreshed = store.get_cluster(cluster.cluster_id)

    assert result["triggered"] is True
    assert "dirty_clusters>=1" in result["decision"]["reasons"]
    assert result["maintenance"]["results"][0]["action"] == "REFRESH_CLUSTER_CARD"
    assert refreshed is not None and refreshed.dirty is False
    assert [trace.action for trace in store.list_action_traces()[-2:]] == [
        "MAINTENANCE_TRIGGER_CHECK",
        "CLUSTER_MAINTENANCE",
    ]
    store.close()


def test_trigger_skips_clean_store_without_new_signal():
    store = SQLiteMemoryStore()
    RememberPipeline(store).ingest_episode(
        RawEpisode(
            user_id="u1",
            project_id="p1",
            content="Remember: I avoid red-eye flights when traveling.",
        )
    )
    first = run_cluster_maintenance_if_needed(store, user_id="u1", project_id="p1")
    second = run_cluster_maintenance_if_needed(store, user_id="u1", project_id="p1")

    assert first["triggered"] is True
    assert second["triggered"] is False
    assert second["maintenance"] is None
    store.close()


def test_trigger_runs_on_recent_verifier_issue_after_previous_maintenance():
    store = SQLiteMemoryStore()
    RememberPipeline(store).ingest_episode(
        RawEpisode(
            user_id="u1",
            project_id="p1",
            content="Remember: I prefer one major attraction per travel day.",
        )
    )
    run_cluster_maintenance_if_needed(store, user_id="u1", project_id="p1")
    store.append_action_trace(
        ActionTrace(
            phase="verify",
            action="CHECK_CONFLICT",
            tool_result_json={
                "results": [
                    {
                        "memory_id": "mem_old",
                        "verdict": "contradicted",
                        "reason": "conflicts with newer selected memory",
                        "keep": False,
                    }
                ]
            },
        )
    )

    decision = MaintenanceTriggerPolicy().decide(store, user_id="u1", project_id="p1")

    assert decision.should_run is True
    assert "recent_verifier_issues>=1" in decision.reasons
    store.close()
