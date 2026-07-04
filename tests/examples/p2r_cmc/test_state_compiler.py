from examples.p2r_cmc.memory_system.clustering.assigner import assign_and_persist_entry
from examples.p2r_cmc.memory_system.executor import MemoryExecutor
from examples.p2r_cmc.memory_system.controller.state_compiler import StateCompiler
from examples.p2r_cmc.memory_system.schemas import MemoryAction, MemoryEntry
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


def test_use_observation_exposes_cards_not_full_memory_bank():
    store = SQLiteMemoryStore()
    entry = MemoryEntry(
        user_id="u1",
        project_id="p1",
        content="P2R-CMC uses MemoryArena.",
        summary="P2R-CMC uses MemoryArena.",
        facets=["project_state"],
        keywords=["p2r", "memoryarena"],
    )
    result = MemoryExecutor(store).execute(MemoryAction.WRITE_NEW, {"entry": entry.model_dump(mode="json")})
    assign_and_persist_entry(store, store.get_memory(result["memory_id"]))

    observation = StateCompiler(store).compile(query="MemoryArena for P2R-CMC", user_id="u1", project_id="p1")
    payload = observation.model_dump(mode="json")
    assert payload["cluster_cards"]
    assert "member_memory_ids" not in str(payload)
    assert "MemoryEntry" not in str(payload)
    store.close()
