from examples.p2r_cmc.memory_system.clustering.assigner import create_cluster_from_entry
from examples.p2r_cmc.memory_system.clustering.metrics import refresh_cluster_stats
from examples.p2r_cmc.memory_system.clustering.router import build_cluster_card, search_cluster_cards
from examples.p2r_cmc.memory_system.schemas import MemoryEntry


def test_cluster_card_contains_facet_distribution_and_routes():
    entry = MemoryEntry(
        user_id="u1",
        project_id="p1",
        content="P2R-CMC uses MemoryArena.",
        summary="P2R-CMC uses MemoryArena.",
        facets=["project_state"],
        keywords=["p2r", "memoryarena"],
        utility=0.5,
    )
    cluster = refresh_cluster_stats(create_cluster_from_entry(entry), [entry])
    card = build_cluster_card(cluster, [entry])
    assert card.facet_distribution["project_state"] == 1
    results = search_cluster_cards("How to evaluate P2R-CMC with MemoryArena?", [card], top_k=1, filters={"project_id": "p1"})
    assert results and results[0].cluster_id == cluster.cluster_id
