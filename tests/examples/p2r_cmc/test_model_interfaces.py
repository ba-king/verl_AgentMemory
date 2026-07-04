from examples.p2r_cmc.memory_system.controller.prompt_policy import OpenAICompatiblePolicy
from examples.p2r_cmc.memory_system.model_clients import OpenAICompatibleConfig
from examples.p2r_cmc.memory_system.controller.remember_pipeline import RememberPipeline
from examples.p2r_cmc.memory_system.model_clients import MainAnswerAgent
from examples.p2r_cmc.memory_system.schemas import (
    EvidenceItem,
    EvidencePacket,
    MemoryAction,
    MemoryEntry,
    PolicyObservation,
)
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


class FakeMemoryChatClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def chat_json(self, messages, **kwargs):
        self.messages = messages
        return self.payload


class FakeMainChatClient:
    def __init__(self):
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return "mock answer"


class SequenceMemoryChatClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def chat_json(self, messages, **kwargs):
        self.calls.append(messages)
        return self.payloads.pop(0)


def test_openai_compatible_memory_policy_extracts_candidates_with_fake_client():
    policy = OpenAICompatiblePolicy(
        chat_client=FakeMemoryChatClient(
            {
                "candidates": [
                    {
                        "content": "User wants concise Chinese summaries.",
                        "summary": "User wants concise Chinese.",
                        "facets": ["preference"],
                        "keywords": ["chinese", "concise"],
                        "scope": "project",
                        "confidence": 0.8,
                        "importance": 0.6,
                    }
                ]
            }
        )
    )
    candidates = policy.extract_candidates("Remember: concise Chinese.", user_id="u1", project_id="p1")
    assert len(candidates) == 1
    assert candidates[0].facets == ["preference"]
    assert "Memory Candidate Extractor" in policy.chat_client.messages[0]["content"]


def test_openai_compatible_memory_policy_decides_remember_with_fake_client():
    policy = OpenAICompatiblePolicy(
        chat_client=FakeMemoryChatClient(
            {
                "action": "WRITE_NEW",
                "memory_ids": [],
                "cluster_ids": [],
                "reason": "Durable project preference.",
                "params": {},
            }
        )
    )
    action = policy.decide_remember(
        candidate=MemoryEntry(
            content="Use concise Chinese.",
            summary="Use concise Chinese.",
            user_id="u1",
            project_id="p1",
        ),
        related_memories=[],
        related_clusters=[],
        assignment_scores={},
        thresholds={"duplicate": 0.9, "update": 0.72},
    )
    assert action is not None
    assert action.action == MemoryAction.WRITE_NEW
    assert "Remember Action Policy" in policy.chat_client.messages[0]["content"]


def test_remember_pipeline_uses_openai_policy_remember_action():
    store = SQLiteMemoryStore()
    policy = OpenAICompatiblePolicy(
        chat_client=SequenceMemoryChatClient(
            [
                {
                    "candidates": [
                        {
                            "content": "The user prefers concise Chinese.",
                            "summary": "User prefers concise Chinese.",
                            "facets": ["preference"],
                            "keywords": ["concise", "Chinese"],
                            "scope": "project",
                            "confidence": 0.8,
                            "importance": 0.6,
                        }
                    ]
                },
                {
                    "action": "NO_OP",
                    "memory_ids": [],
                    "cluster_ids": [],
                    "reason": "Already covered by session context.",
                    "params": {},
                },
            ]
        )
    )
    result = RememberPipeline(store, policy=policy).remember_text(
        "Remember: concise Chinese.",
        user_id="u1",
        project_id="p1",
    )
    assert result[0]["action"] == MemoryAction.NO_OP
    assert store.list_memories(user_id="u1", project_id="p1") == []
    assert "Memory Candidate Extractor" in policy.chat_client.calls[0][0]["content"]
    assert "Remember Action Policy" in policy.chat_client.calls[1][0]["content"]
    store.close()


def test_openai_compatible_memory_policy_decides_use_with_fake_client():
    policy = OpenAICompatiblePolicy(
        chat_client=FakeMemoryChatClient(
            {
                "action": "EXPAND_CLUSTER",
                "cluster_ids": ["c1"],
                "memory_ids": [],
                "reason": "Relevant card.",
            }
        )
    )
    action = policy.decide_use(PolicyObservation(query="q"))
    assert action.action == MemoryAction.EXPAND_CLUSTER
    assert action.cluster_ids == ["c1"]
    assert "Memory Use Controller" in policy.chat_client.messages[0]["content"]


def test_openai_compatible_memory_policy_proposes_cluster_maintenance():
    policy = OpenAICompatiblePolicy(
        chat_client=FakeMemoryChatClient(
            {
                "proposals": [
                    {
                        "action": "REFRESH_CLUSTER_CARD",
                        "cluster_ids": ["c1"],
                        "memory_ids": [],
                        "confidence": 0.9,
                        "reason": "Dirty card.",
                        "params": {"title": "Clean card"},
                    },
                    {
                        "action": "DEPRECATE",
                        "memory_ids": ["m1"],
                        "confidence": 0.9,
                        "reason": "Illegal in cluster-index-only maintenance.",
                        "params": {},
                    },
                ]
            }
        )
    )

    proposals = policy.propose_cluster_maintenance({"clusters": [{"cluster_id": "c1"}]})

    assert len(proposals) == 1
    assert proposals[0]["action"] == "REFRESH_CLUSTER_CARD"
    assert proposals[0]["cluster_ids"] == ["c1"]
    assert "Cluster Maintenance Policy" in policy.chat_client.messages[0]["content"]


def test_main_answer_agent_uses_evidence_packet_with_fake_client():
    client = FakeMainChatClient()
    agent = MainAnswerAgent(client)
    packet = EvidencePacket(
        query="q",
        evidence_items=[
            EvidenceItem(memory_id="m1", content="Use Chinese.", summary="Use Chinese.", score=1.0)
        ],
    )
    assert agent.answer("How to answer?", packet) == "mock answer"
    assert "Use Chinese." in client.messages[1]["content"]


def test_main_answer_agent_prompt_preserves_hard_constraints():
    client = FakeMainChatClient()
    agent = MainAnswerAgent(client)
    packet = EvidencePacket(
        query="q",
        evidence_items=[
            EvidenceItem(
                memory_id="m_food",
                content="Travel partner does not eat raw fish and avoids beef.",
                summary="Food restrictions for travel.",
                facets=["preference"],
                score=1.0,
            )
        ],
        conflicts=["m_old: contradicted - conflicts with newer selected memory m_food"],
    )

    assert agent.answer("Plan meals.", packet) == "mock answer"
    assert "Preserve hard constraints exactly" in client.messages[0]["content"]
    assert "raw fish" in client.messages[1]["content"]
    assert "avoids beef" in client.messages[1]["content"]
    assert "Verifier notes" in client.messages[1]["content"]


def test_main_answer_agent_prompt_preserves_query_structure_constraints():
    client = FakeMainChatClient()
    agent = MainAnswerAgent(client)
    packet = EvidencePacket(
        query="q",
        evidence_items=[
            EvidenceItem(
                memory_id="m_pace",
                content="User prefers one major attraction per day.",
                summary="One major attraction per day.",
                facets=["preference"],
                score=1.0,
            )
        ],
    )

    query = "Give exactly 3 options and choose either batch mode or streaming mode."
    assert agent.answer(query, packet) == "mock answer"
    assert "current user query as hard constraints" in client.messages[0]["content"]
    assert "Respect exclusivity and cardinality" in client.messages[0]["content"]
    assert "exactly N items" in client.messages[0]["content"]
    assert query in client.messages[1]["content"]


def test_openai_compatible_config_reads_timeout_env(monkeypatch):
    monkeypatch.setenv("P2R_MAIN_API_KEY", "sk-test")
    monkeypatch.setenv("P2R_MAIN_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("P2R_MAIN_TIMEOUT", "180")

    config = OpenAICompatibleConfig.from_env(prefix="P2R_MAIN")

    assert config.timeout == 180
