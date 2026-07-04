from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.p2r_cmc.memory_system.clustering.assigner import assign_and_persist_entry
from examples.p2r_cmc.memory_system.clustering.maintenance_trigger import run_cluster_maintenance_if_needed
from examples.p2r_cmc.memory_system.clustering.metrics import refresh_cluster_stats
from examples.p2r_cmc.memory_system.controller.prompt_policy import OpenAICompatiblePolicy
from examples.p2r_cmc.memory_system.controller.remember_pipeline import RememberPipeline
from examples.p2r_cmc.memory_system.controller.session_pipeline import SessionBuffer, SessionMemoryPipeline
from examples.p2r_cmc.memory_system.controller.use_pipeline import UsePipeline
from examples.p2r_cmc.memory_system.export import export_store_json
from examples.p2r_cmc.memory_system.model_clients import LocalQwenEmbeddingClient, MainAnswerAgent
from examples.p2r_cmc.memory_system.schemas import EvidencePacket, MemoryEdge, MemoryEntry
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="A longer P2R-CMC travel-planning memory demo.")
    parser.add_argument("--embedding", choices=["none", "local-qwen"], default="local-qwen")
    parser.add_argument("--memory-agent", choices=["rule", "openai"], default="rule")
    parser.add_argument("--main-agent", choices=["none", "openai"], default="none")
    parser.add_argument("--db-path", default=":memory:", help="SQLite path. Use a file path to persist the memory store.")
    parser.add_argument("--reset", action="store_true", help="Clear an existing SQLite store before running the demo.")
    parser.add_argument("--dump-json", default=None, help="Optional path to export a readable JSON snapshot after the run.")
    parser.add_argument("--include-embeddings", action="store_true", help="Include full embedding vectors in --dump-json.")
    args = parser.parse_args()

    embedder = LocalQwenEmbeddingClient.from_env() if args.embedding == "local-qwen" else None
    memory_policy = OpenAICompatiblePolicy.from_env() if args.memory_agent == "openai" else None

    if args.db_path != ":memory:":
        Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteMemoryStore(args.db_path)
    if args.reset:
        store.reset()

    user_id = "travel_demo_user"
    project_id = "life-travel-planning"
    remember = RememberPipeline(store, policy=memory_policy, embedder=None)

    session = build_travel_session(user_id=user_id, project_id=project_id)

    print("P2R-CMC travel planning demo")
    print("=" * 80)
    print("STEP 1: A realistic multi-turn travel session is buffered, but long-term memory is still empty.")
    print(json.dumps({"session_id": session.session_id, "turns": [turn.__dict__ for turn in session.turns]}, ensure_ascii=False, indent=2))
    print("memory_count_before_session_end:", len(store.list_memories(user_id=user_id, project_id=project_id)))

    session_results = SessionMemoryPipeline(remember).consolidate_session(session)
    print("\nSTEP 2: Session-end extraction writes durable travel memories.")
    print(json.dumps(session_results, ensure_ascii=False, indent=2))
    print("memory_count_after_session_end:", len(store.list_memories(user_id=user_id, project_id=project_id)))
    if embedder is not None:
        print("embedding_index_update:", json.dumps(embed_existing_memories(store, user_id, project_id, embedder), ensure_ascii=False))

    seeded = seed_stale_travel_conflict(store, user_id=user_id, project_id=project_id, embedder=embedder)
    print("\nSTEP 3: Seeded stale conflicting travel memory to test verifier filtering.")
    print(json.dumps(seeded, ensure_ascii=False, indent=2))

    maintenance = run_cluster_maintenance_if_needed(
        store,
        user_id=user_id,
        project_id=project_id,
        policy=memory_policy,
    )
    print("\nSTEP 4: Trigger policy decides whether to run cluster maintenance.")
    print(json.dumps(compact_maintenance_result(maintenance), ensure_ascii=False, indent=2))

    user_query = (
        "请根据我之前说过的旅行偏好，帮我设计一个 6 天关西旅行初版方案："
        "京都 3 天、大阪 2 天、奈良或宇治 1 天。请同时考虑航班时间、住宿区域、"
        "每日节奏、饮食限制、妈妈膝盖不适、下雨备用方案，以及为什么不要安排红眼航班"
        "或每天塞满景点。回答用中文，结构化但不要像论文。"
    )
    retrieval_query = (
        f"{user_query}\n"
        "Retrieval hints: relaxed pace, no departures before 9:30 AM, one major attraction per day, "
        "quiet cafe, bookstore, mother knee pain, avoid stairs, mid-range hotels near transit, "
        "simple local meals, raw fish, avoids beef, no red-eye flights, no tight transfers, "
        "rainy days, museums, covered shopping arcades, tea houses, craft workshops."
    )
    packet = UsePipeline(store, policy=memory_policy, embedder=embedder).run_sync(
        retrieval_query,
        user_id=user_id,
        project_id=project_id,
        budget={
            "verify": True,
            "max_clusters": 8,
            "top_m": 12,
            "max_items": 8,
            "top_k_per_cluster": 8,
            "token_budget": 2400,
        },
    )
    packet.query = user_query
    print("\nSTEP 5: Query-time retrieval expands relevant clusters, applies RRF, then verifies evidence.")
    print(json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2))

    if args.main_agent == "openai":
        answer = MainAnswerAgent.from_env().answer(user_query, packet)
        print("\n" + "=" * 80)
        print("MAIN_AGENT_ANSWER")
        print("=" * 80)
        print(answer)
    else:
        print("\n" + "=" * 80)
        print("DETERMINISTIC_FALLBACK_ANSWER")
        print("=" * 80)
        print(render_fallback_travel_answer(packet))

    if args.dump_json:
        dump_path = Path(args.dump_json)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            json.dumps(export_store_json(store, include_embeddings=args.include_embeddings), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote JSON dump: {dump_path}")
    store.close()


def build_travel_session(*, user_id: str, project_id: str) -> SessionBuffer:
    session = SessionBuffer(user_id=user_id, project_id=project_id, session_id="travel_demo_session")
    session.add_turn(
        "user",
        "I am thinking about a Kansai trip in Japan, probably Kyoto, Osaka, and one quieter day trip.",
    )
    session.add_turn(
        "assistant",
        "That sounds like a good route. We can plan around pace, mobility, food, hotels, and weather backup.",
    )
    session.add_turn(
        "user",
        (
            "Remember: For leisure trips, I prefer a relaxed pace: no departures before 9:30 AM, "
            "one major attraction per day, and at least one quiet cafe or bookstore break."
        ),
    )
    session.add_turn(
        "user",
        (
            "Remember: For this Japan trip, my mother has mild knee pain, so avoid long stair-heavy routes; "
            "elevators, buses, and short taxis are acceptable when they reduce walking."
        ),
    )
    session.add_turn(
        "assistant",
        "I will avoid routes that depend on too many stairs or long uphill walks.",
    )
    session.add_turn(
        "user",
        (
            "Remember: I am budgeting carefully for this trip: prefer mid-range hotels near transit, "
            "simple local meals, and only one or two splurge experiences."
        ),
    )
    session.add_turn(
        "user",
        (
            "Remember: Food preferences for travel: I like seafood, noodles, matcha desserts, and izakaya snacks, "
            "but my travel partner does not eat raw fish and avoids beef."
        ),
    )
    session.add_turn(
        "user",
        (
            "Remember: When flying internationally, I avoid red-eye flights and tight transfers; "
            "I would rather choose daytime flights with at least two hours between connections."
        ),
    )
    session.add_turn(
        "user",
        (
            "Remember: For rainy days while traveling, I prefer indoor culture plans such as museums, "
            "covered shopping arcades, tea houses, craft workshops, and department-store food halls."
        ),
    )
    session.add_turn(
        "assistant",
        "Great. For Kansai, that points toward a slower Kyoto base, easy transit, and flexible indoor backup days.",
    )
    session.add_turn(
        "user",
        "Would Nara or Uji be easier as the quieter day trip if we do not want a stair-heavy day?",
    )
    return session


def seed_stale_travel_conflict(
    store: SQLiteMemoryStore,
    *,
    user_id: str,
    project_id: str,
    embedder: LocalQwenEmbeddingClient | None,
) -> dict[str, Any]:
    flight_memory = find_memory(
        store,
        user_id,
        project_id,
        ["avoid red-eye flights", "tight transfers"],
    )
    pace_memory = find_memory(
        store,
        user_id,
        project_id,
        ["relaxed pace", "one major attraction"],
    )
    anchor = flight_memory or pace_memory
    if anchor is None:
        return {"ok": False, "reason": "No fresh travel preference memory found."}

    stale_time = anchor.updated_at - timedelta(days=21)
    stale = MemoryEntry(
        user_id=user_id,
        project_id=project_id,
        content=(
            "For Japan trips, the user prefers the cheapest red-eye flights, tight layovers, "
            "early departures, and packed itineraries with several major attractions each day."
        ),
        summary="Old stale note: choose cheap red-eye flights, tight layovers, and packed early itineraries.",
        scope="project",
        facets=["preference"],
        keywords=[
            "Japan",
            "travel",
            "red-eye",
            "tight",
            "layovers",
            "early",
            "packed",
            "itineraries",
            "attractions",
        ],
        confidence=0.72,
        importance=0.46,
        created_at=stale_time,
        updated_at=stale_time,
    )
    if embedder is not None:
        stale.embedding = embedder.embed_text(stale.summary)
    store.upsert_memory(stale)
    assign_and_persist_entry(store, stale)

    edge_targets = [memory for memory in [flight_memory, pace_memory] if memory is not None]
    for target in edge_targets:
        store.upsert_edge(
            MemoryEdge(
                src_memory_id=stale.memory_id,
                dst_memory_id=target.memory_id,
                relation_type="CONFLICTS_WITH",
                evidence="Travel demo conflict: newer preferences reject red-eye flights, tight transfers, or packed days.",
            )
        )
    return {
        "ok": True,
        "stale_memory_id": stale.memory_id,
        "conflicts_with": [target.memory_id for target in edge_targets],
        "stale_summary": stale.summary,
    }


def embed_existing_memories(
    store: SQLiteMemoryStore,
    user_id: str,
    project_id: str,
    embedder: LocalQwenEmbeddingClient,
) -> dict[str, Any]:
    embedded_memory_ids: list[str] = []
    for entry in store.list_memories(user_id=user_id, project_id=project_id, statuses={"active"}):
        if entry.embedding is None:
            entry.embedding = embedder.embed_text(entry.summary or entry.content)
            store.upsert_memory(entry)
            embedded_memory_ids.append(entry.memory_id)

    refreshed_cluster_ids: list[str] = []
    for cluster in store.list_clusters(user_id=user_id, project_id=project_id, statuses={"active"}):
        entries = store.get_memories(cluster.member_memory_ids)
        store.upsert_cluster(refresh_cluster_stats(cluster, entries))
        refreshed_cluster_ids.append(cluster.cluster_id)

    return {
        "embedded_memories": len(embedded_memory_ids),
        "refreshed_clusters": len(refreshed_cluster_ids),
    }


def find_memory(
    store: SQLiteMemoryStore,
    user_id: str,
    project_id: str,
    needles: list[str],
) -> MemoryEntry | None:
    lowered_needles = [needle.lower() for needle in needles]
    for entry in store.list_memories(user_id=user_id, project_id=project_id, statuses={"active"}):
        haystack = f"{entry.content}\n{entry.summary}".lower()
        if all(needle in haystack for needle in lowered_needles):
            return entry
    for entry in store.list_memories(user_id=user_id, project_id=project_id, statuses={"active"}):
        haystack = f"{entry.content}\n{entry.summary}".lower()
        if any(needle in haystack for needle in lowered_needles):
            return entry
    return None


def compact_maintenance_result(result: dict[str, object]) -> dict[str, object]:
    if "decision" in result:
        maintenance = result.get("maintenance")
        return {
            "triggered": result.get("triggered"),
            "decision": result.get("decision"),
            "maintenance": compact_maintenance_result(maintenance) if isinstance(maintenance, dict) else None,
        }
    report = result["report"]
    return {
        "report": report.model_dump(mode="json") if hasattr(report, "model_dump") else report,
        "proposal_source": result.get("proposal_source"),
        "proposals": result.get("proposals"),
        "results": result["results"],
    }


def render_fallback_travel_answer(packet: EvidencePacket) -> str:
    evidence_summaries = [item.summary for item in packet.evidence_items]
    evidence_block = "\n".join(f"- {summary}" for summary in evidence_summaries) or "- 没有检索到足够记忆"
    conflicts = "\n".join(f"- {conflict}" for conflict in packet.conflicts) or "- 无"
    return (
        "下面是一个不用 API 的模板化答案，用来检查 verified evidence 是否足够支撑主回答。\n\n"
        "已使用的记忆证据：\n"
        f"{evidence_block}\n\n"
        "被 verifier 标记或过滤的风险：\n"
        f"{conflicts}\n\n"
        "旅行方案初版：\n"
        "1. 航班：优先选白天航班，转机至少留 2 小时。不要为了省钱选红眼或极紧转机，"
        "因为这和已验证的出行偏好冲突，也会影响第一天体力。\n"
        "2. 住宿：京都 3 晚建议住四条、乌丸、京都站附近；大阪 2 晚建议住难波或梅田附近。"
        "核心标准是靠近地铁或 JR，选择中档酒店，把预算留给 1 到 2 个真正想体验的项目。\n"
        "3. 节奏：每天只放一个主景点，上午 9:30 以后出发，中间安排咖啡馆、书店、茶屋或商场休息。"
        "不要做每天多景点冲刺式路线。\n"
        "4. 京都 3 天：一天东山或祇园慢逛，一天岚山但减少爬坡和台阶，一天锦市场、茶屋、书店或博物馆。"
        "如果下雨，把户外寺社换成京都国立博物馆、商场、茶屋或手作体验。\n"
        "5. 大阪 2 天：一天难波、心斋桥、道顿堀附近吃饭和购物，一天梅田、展望台或中之岛美术馆。"
        "餐食避开纯生食和牛肉主导餐厅，多选乌冬、拉面、天妇罗、熟海鲜、居酒屋小菜。\n"
        "6. 一日游：如果担心膝盖负担，宇治通常比把奈良大范围走满更稳；可以安排平等院、宇治川、抹茶甜品，"
        "路线短、节奏慢，也更容易临时找室内休息点。\n"
    )


if __name__ == "__main__":
    main()
