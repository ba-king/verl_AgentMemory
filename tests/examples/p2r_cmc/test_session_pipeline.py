from examples.p2r_cmc.memory_system.controller.remember_pipeline import RememberPipeline
from examples.p2r_cmc.memory_system.controller.session_pipeline import SessionBuffer, SessionMemoryPipeline
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


def test_session_pipeline_writes_only_at_session_end():
    store = SQLiteMemoryStore()
    session = SessionBuffer(user_id="u1", project_id="p1", session_id="s1")
    session.add_turn("user", "Can you explain BM25?")
    session.add_turn("assistant", "BM25 is a sparse retrieval method.")
    session.add_turn("user", "Remember: For this project, Stage 1 should stay prompt-based before RL.")

    assert store.list_memories(user_id="u1", project_id="p1") == []

    results = SessionMemoryPipeline(RememberPipeline(store)).consolidate_session(session)
    memories = store.list_memories(user_id="u1", project_id="p1")
    assert results
    assert len(memories) == 1
    assert "Stage 1 should stay prompt-based" in memories[0].content
    assert results[0]["session_id"] == "s1"
    store.close()
