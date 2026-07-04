from examples.p2r_cmc.memory_system.controller.verifier import MemoryVerifier
from examples.p2r_cmc.memory_system.executor import MemoryExecutor
from examples.p2r_cmc.memory_system.schemas import EvidenceItem, EvidencePacket, MemoryAction, MemoryEdge, MemoryEntry
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


def write_entry(store: SQLiteMemoryStore, entry: MemoryEntry) -> str:
    return MemoryExecutor(store).execute(MemoryAction.WRITE_NEW, {"entry": entry.model_dump(mode="json")})["memory_id"]


def test_verifier_filters_unsafe_memory():
    store = SQLiteMemoryStore()
    memory_id = write_entry(
        store,
        MemoryEntry(user_id="u1", project_id="p1", content="The API key is sk-secret123456789.", summary="API key."),
    )
    packet = EvidencePacket(
        query="What is remembered?",
        evidence_items=[EvidenceItem(memory_id=memory_id, content="secret", summary="secret")],
    )
    verified, results = MemoryVerifier(store).verify_packet(packet)
    assert verified.evidence_items == []
    assert results[0].verdict == "unsafe"
    assert store.list_action_traces()[-1].phase == "verify"
    store.close()


def test_verifier_filters_older_conflicting_memory():
    store = SQLiteMemoryStore()
    old_id = write_entry(
        store,
        MemoryEntry(user_id="u1", project_id="p1", content="Stage 1 includes RL.", summary="Stage 1 includes RL."),
    )
    new_id = write_entry(
        store,
        MemoryEntry(user_id="u1", project_id="p1", content="Stage 1 is prompt-based; RL is Stage 2.", summary="RL is Stage 2."),
    )
    store.upsert_edge(MemoryEdge(src_memory_id=old_id, dst_memory_id=new_id, relation_type="CONFLICTS_WITH"))
    packet = EvidencePacket(
        query="What should Stage 1 include?",
        evidence_items=[
            EvidenceItem(memory_id=old_id, content="old", summary="old"),
            EvidenceItem(memory_id=new_id, content="new", summary="new"),
        ],
    )
    verified, results = MemoryVerifier(store).verify_packet(packet)
    assert [item.memory_id for item in verified.evidence_items] == [new_id]
    assert {result.verdict for result in results} == {"contradicted", "valid"}
    store.close()
