from examples.p2r_cmc.memory_system.clustering.assigner import assign_entry_to_cluster, create_cluster_from_entry
from examples.p2r_cmc.memory_system.schemas import MemoryEntry


def entry(content: str, keywords: list[str]) -> MemoryEntry:
    return MemoryEntry(
        user_id="u1",
        project_id="p1",
        content=content,
        summary=content,
        facets=["preference"],
        keywords=keywords,
    )


def test_similar_entry_joins_existing_cluster():
    first = entry("User avoids caffeine after 3 PM.", ["caffeine", "sleep", "drink"])
    cluster = create_cluster_from_entry(first)
    result = assign_entry_to_cluster(entry("User avoids caffeine late in the day.", ["caffeine", "sleep", "drink"]), [cluster])
    assert not result.created_new_cluster
    assert result.primary_cluster.cluster_id == cluster.cluster_id


def test_dissimilar_entry_creates_new_cluster():
    first = entry("User avoids caffeine after 3 PM.", ["caffeine", "sleep", "drink"])
    cluster = create_cluster_from_entry(first)
    other = MemoryEntry(user_id="u1", project_id="p1", content="Project uses GRPO.", summary="Project uses GRPO.", facets=["project_state"], keywords=["grpo"])
    result = assign_entry_to_cluster(other, [cluster])
    assert result.created_new_cluster
    assert result.primary_cluster.cluster_id != cluster.cluster_id
