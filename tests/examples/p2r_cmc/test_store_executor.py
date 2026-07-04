from examples.p2r_cmc.memory_system.executor import MemoryExecutor
from examples.p2r_cmc.memory_system.schemas import MemoryAction, MemoryEntry
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


def make_entry(content="Use concise Chinese.", summary="Use concise Chinese.") -> MemoryEntry:
    return MemoryEntry(user_id="u1", project_id="p1", content=content, summary=summary, keywords=["chinese"])


def test_executor_write_update_deprecate_merge_and_trace():
    store = SQLiteMemoryStore()
    executor = MemoryExecutor(store)

    write = executor.execute(MemoryAction.WRITE_NEW, {"entry": make_entry().model_dump(mode="json")})
    memory_id = write["memory_id"]
    assert store.get_memory(memory_id) is not None

    update = executor.execute(
        MemoryAction.UPDATE_EXISTING,
        {"memory_id": memory_id, "updates": {"content": "Use concise Chinese bullets.", "summary": "Use Chinese bullets."}},
    )
    updated = store.get_memory(memory_id)
    assert update["version"] == 2
    assert updated is not None and updated.version == 2
    assert len(store.list_memory_versions(memory_id)) == 1

    dep = executor.execute(MemoryAction.DEPRECATE, {"memory_id": memory_id, "reason": "stale"})
    deprecated = store.get_memory(memory_id)
    assert dep["status"] == "deprecated"
    assert deprecated is not None and deprecated.status == "deprecated"

    second = executor.execute(MemoryAction.WRITE_NEW, {"entry": make_entry("P2R-CMC uses clusters.", "P2R uses clusters.").model_dump(mode="json")})
    merge = executor.execute(
        MemoryAction.MERGE,
        {"source_memory_ids": [memory_id, second["memory_id"]], "content": "Use Chinese bullets for P2R-CMC clusters.", "summary": "Chinese P2R-CMC cluster reporting."},
    )
    assert store.get_memory(merge["memory_id"]) is not None
    assert len(store.list_action_traces()) == 5
    store.close()
