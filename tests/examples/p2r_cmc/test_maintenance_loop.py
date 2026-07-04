from examples.p2r_cmc.memory_system.clustering.maintenance_loop import run_cluster_maintenance
from examples.p2r_cmc.memory_system.clustering.assigner import create_cluster_from_entry
from examples.p2r_cmc.memory_system.controller.remember_pipeline import RememberPipeline
from examples.p2r_cmc.memory_system.schemas import ClusterMembership, MemoryEntry, RawEpisode
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


class FakeMaintenancePolicy:
    def __init__(self, proposals):
        self.proposals = proposals
        self.observation = None

    def propose_cluster_maintenance(self, observation):
        self.observation = observation
        return self.proposals


def test_cluster_maintenance_refreshes_dirty_cluster_card():
    store = SQLiteMemoryStore()
    RememberPipeline(store).ingest_episode(
        RawEpisode(
            user_id="u1",
            project_id="p1",
            content="Remember: P2R-CMC Stage 1 uses MemoryArena before RL.",
        )
    )
    cluster = store.list_clusters(user_id="u1", project_id="p1")[0]
    assert cluster.dirty is True

    result = run_cluster_maintenance(store, user_id="u1", project_id="p1")
    refreshed = store.get_cluster(cluster.cluster_id)
    assert result["results"][0]["action"] == "REFRESH_CLUSTER_CARD"
    assert refreshed is not None and refreshed.dirty is False
    assert store.list_action_traces()[-1].phase == "maintenance"
    store.close()


def test_policy_cluster_maintenance_refreshes_card_with_model_params():
    store = SQLiteMemoryStore()
    RememberPipeline(store).ingest_episode(
        RawEpisode(
            user_id="u1",
            project_id="p1",
            content="Remember: I prefer quiet travel days with one major attraction.",
        )
    )
    cluster = store.list_clusters(user_id="u1", project_id="p1")[0]
    policy = FakeMaintenancePolicy(
        [
            {
                "action": "REFRESH_CLUSTER_CARD",
                "cluster_ids": [cluster.cluster_id],
                "confidence": 0.9,
                "reason": "Make routing title clearer.",
                "params": {
                    "title": "Quiet travel pace preferences",
                    "summary": "Preferences about relaxed travel days and limited attractions.",
                    "keywords": ["travel", "pace", "quiet"],
                },
            }
        ]
    )

    result = run_cluster_maintenance(store, user_id="u1", project_id="p1", policy=policy)
    refreshed = store.get_cluster(cluster.cluster_id)

    assert result["proposal_source"] == "policy"
    assert policy.observation["mode"] == "cluster_index_only"
    assert refreshed is not None
    assert refreshed.title == "Quiet travel pace preferences"
    assert refreshed.summary == "Preferences about relaxed travel days and limited attractions."
    assert refreshed.keywords == ["travel", "pace", "quiet"]
    store.close()


def test_policy_cluster_maintenance_splits_with_model_groups():
    store = SQLiteMemoryStore()
    entries = [
        MemoryEntry(
            memory_id="mem_pace",
            user_id="u1",
            project_id="p1",
            content="User prefers slow travel pace.",
            summary="Slow travel pace.",
            facets=["preference"],
            keywords=["pace"],
        ),
        MemoryEntry(
            memory_id="mem_food",
            user_id="u1",
            project_id="p1",
            content="Partner avoids raw fish and beef.",
            summary="Food restrictions.",
            facets=["preference"],
            keywords=["food"],
        ),
    ]
    for entry in entries:
        store.upsert_memory(entry)
    cluster = create_cluster_from_entry(entries[0])
    cluster.member_memory_ids = [entry.memory_id for entry in entries]
    cluster.dirty = True
    store.upsert_cluster(cluster)
    for entry in entries:
        store.upsert_membership(ClusterMembership(cluster_id=cluster.cluster_id, memory_id=entry.memory_id, role="primary"))

    policy = FakeMaintenancePolicy(
        [
            {
                "action": "SPLIT_CLUSTER",
                "cluster_ids": [cluster.cluster_id],
                "confidence": 0.95,
                "reason": "Cluster mixes pace and food constraints.",
                "params": {
                    "groups": [
                        {
                            "title": "Travel pace",
                            "summary": "Preferences about travel speed and daily schedule.",
                            "keywords": ["pace", "schedule"],
                            "member_memory_ids": ["mem_pace"],
                        },
                        {
                            "title": "Food restrictions",
                            "summary": "Dietary restrictions for travel companions.",
                            "keywords": ["food", "raw fish", "beef"],
                            "member_memory_ids": ["mem_food"],
                        },
                    ]
                },
            }
        ]
    )

    result = run_cluster_maintenance(store, user_id="u1", project_id="p1", policy=policy)
    old_cluster = store.get_cluster(cluster.cluster_id)
    active_titles = {cluster.title for cluster in store.list_clusters(user_id="u1", project_id="p1", statuses={"active"})}

    assert result["results"][0]["action"] == "SPLIT_CLUSTER"
    assert old_cluster is not None and old_cluster.status == "split"
    assert {"Travel pace", "Food restrictions"}.issubset(active_titles)
    store.close()
