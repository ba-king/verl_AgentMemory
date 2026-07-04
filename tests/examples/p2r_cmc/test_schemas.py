from examples.p2r_cmc.memory_system.schemas import (
    ActionTrace,
    ClusterCard,
    ClusterHealthReport,
    ClusterMembership,
    ClusterNode,
    EvidenceItem,
    EvidencePacket,
    ExposureEvent,
    MemoryAction,
    MemoryEdge,
    MemoryEntry,
    QueryAspect,
    RawEpisode,
    RewardTrace,
)


def test_required_schemas_construct_and_serialize():
    episode = RawEpisode(user_id="u1", project_id="p1", content="Remember: use Chinese.")
    entry = MemoryEntry(user_id="u1", project_id="p1", content="Use Chinese.", summary="Use Chinese.")
    cluster = ClusterNode(user_id="u1", project_id="p1", title="Language preference", summary="Use Chinese.")
    card = ClusterCard(
        cluster_id=cluster.cluster_id,
        title=cluster.title,
        summary=cluster.summary,
        representative_memory=entry.summary,
        size=1,
        user_id="u1",
        project_id="p1",
    )
    membership = ClusterMembership(cluster_id=cluster.cluster_id, memory_id=entry.memory_id)
    edge = MemoryEdge(src_memory_id=entry.memory_id, dst_memory_id=entry.memory_id, relation_type="SUPPORTS")
    evidence = EvidenceItem(memory_id=entry.memory_id, content=entry.content, summary=entry.summary)
    aspect = QueryAspect(text="language preference", keywords=["language", "preference"])
    packet = EvidencePacket(query="How should you answer?", evidence_items=[evidence], required_aspects=[aspect])
    action = ActionTrace(phase="test", action=MemoryAction.WRITE_NEW)
    exposure = ExposureEvent(query="q", selected_memory_ids=[entry.memory_id])
    reward = RewardTrace(total_reward=1.0)
    health = ClusterHealthReport(num_clusters=1)

    for model in [episode, entry, cluster, card, membership, edge, evidence, aspect, packet, action, exposure, reward, health]:
        assert model.model_dump(mode="json")
