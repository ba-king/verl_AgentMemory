from examples.p2r_cmc.memory_system.controller.remember_pipeline import RememberPipeline
from examples.p2r_cmc.memory_system.controller.use_pipeline import UsePipeline
from examples.p2r_cmc.memory_system.schemas import RawEpisode
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


class FakeEmbedder:
    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]

    def embed_text(self, text):
        lowered = text.lower()
        if "memoryarena" in lowered or "p2r-cmc" in lowered:
            return [1.0, 0.0, 0.0]
        if "latte" in lowered or "coffee" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_embedding_path_sets_entry_cluster_and_dense_scores():
    store = SQLiteMemoryStore()
    embedder = FakeEmbedder()
    remember = RememberPipeline(store, embedder=embedder)
    remember.ingest_episode(
        RawEpisode(
            user_id="u1",
            project_id="p1",
            content="Remember: P2R-CMC should use MemoryArena for evaluation.",
        )
    )

    entries = store.list_memories(user_id="u1", project_id="p1")
    clusters = store.list_clusters(user_id="u1", project_id="p1")
    assert entries[0].embedding == [1.0, 0.0, 0.0]
    assert clusters[0].centroid_embedding == [1.0, 0.0, 0.0]

    packet = UsePipeline(store, embedder=embedder).run_sync("How do we evaluate P2R-CMC?", user_id="u1", project_id="p1")
    assert packet.evidence_items
    assert any(item.rationale and "dense_embedding" in item.rationale for item in packet.evidence_items)
    store.close()
