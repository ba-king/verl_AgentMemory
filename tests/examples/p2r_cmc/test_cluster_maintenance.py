from examples.p2r_cmc.memory_system.clustering.assigner import create_cluster_from_entry
from examples.p2r_cmc.memory_system.clustering.maintenance import maintain_clusters
from examples.p2r_cmc.memory_system.schemas import MemoryEntry


def make_entry(idx: int, project_id="p1", keywords=None) -> MemoryEntry:
    keywords = keywords or ["memory", "cluster"]
    return MemoryEntry(
        memory_id=f"m_{idx}",
        user_id="u1",
        project_id=project_id,
        content=f"Memory cluster note {idx}",
        summary=f"Memory cluster note {idx}",
        facets=["project_state"],
        keywords=keywords,
    )


def test_large_cluster_produces_split_proposal():
    entries = [make_entry(idx) for idx in range(101)]
    cluster = create_cluster_from_entry(entries[0])
    cluster.member_memory_ids = [entry.memory_id for entry in entries]
    report = maintain_clusters([cluster], entries)
    assert cluster.cluster_id in report.split_candidates


def test_similar_small_clusters_merge_but_different_project_does_not():
    left_entry = make_entry(1, keywords=["memoryarena", "cluster"])
    right_entry = make_entry(2, keywords=["memoryarena", "cluster"])
    other_project_entry = make_entry(3, project_id="p2", keywords=["memoryarena", "cluster"])
    left = create_cluster_from_entry(left_entry)
    right = create_cluster_from_entry(right_entry)
    other = create_cluster_from_entry(other_project_entry)
    report = maintain_clusters([left, right, other], [left_entry, right_entry, other_project_entry], merge_threshold=0.8)
    assert (left.cluster_id, right.cluster_id) in report.merge_candidates
    assert all(other.cluster_id not in pair for pair in report.merge_candidates)
