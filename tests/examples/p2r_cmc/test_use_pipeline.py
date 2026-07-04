from examples.p2r_cmc.memory_system.controller.remember_pipeline import RememberPipeline
from examples.p2r_cmc.memory_system.controller.use_pipeline import UsePipeline
from examples.p2r_cmc.memory_system.executor import MemoryExecutor
from examples.p2r_cmc.memory_system.schemas import MemoryAction, RawEpisode
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


def test_use_pipeline_logs_exposure_and_skips_deprecated_hard_fact():
    store = SQLiteMemoryStore()
    remember = RememberPipeline(store)
    remember.ingest_episode(
        RawEpisode(user_id="u1", project_id="p1", content="Remember: P2R-CMC first stage is prompt-based and tested with MemoryArena.")
    )
    remember.ingest_episode(
        RawEpisode(user_id="u1", project_id="p1", content="Remember: This deprecated note says use RL in stage one.")
    )
    deprecated = [entry for entry in store.list_memories(user_id="u1", project_id="p1") if "deprecated note" in entry.content][0]
    MemoryExecutor(store).execute(MemoryAction.DEPRECATE, {"memory_id": deprecated.memory_id})

    packet = UsePipeline(store).run_sync("How should P2R-CMC stage one be tested?", user_id="u1", project_id="p1")
    assert packet.evidence_items
    assert all(item.status == "active" for item in packet.evidence_items)
    assert store.list_exposure_events()
    store.close()
