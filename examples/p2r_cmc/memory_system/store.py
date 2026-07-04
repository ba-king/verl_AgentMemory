from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel

from .schemas import (
    ActionTrace,
    ClusterMembership,
    ClusterNode,
    ExposureEvent,
    MemoryEdge,
    MemoryEntry,
    RewardTrace,
    utc_now,
)

T = TypeVar("T", bound=BaseModel)


def _dump(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def _load(data: str, model_type: type[T]) -> T:
    return model_type.model_validate(json.loads(data))


class SQLiteMemoryStore:
    """Small SQLite-backed store for the prompt-based P2R-CMC prototype."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def reset(self) -> None:
        for table in (
            "memories",
            "memory_versions",
            "clusters",
            "memberships",
            "edges",
            "action_traces",
            "exposure_events",
            "reward_traces",
        ):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                snapshot TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                changed_by_action TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_id TEXT,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memberships (
                membership_id TEXT PRIMARY KEY,
                cluster_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                role TEXT NOT NULL,
                score REAL NOT NULL,
                data TEXT NOT NULL,
                UNIQUE(cluster_id, memory_id, role)
            );
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                src_memory_id TEXT NOT NULL,
                dst_memory_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS action_traces (
                trace_id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS exposure_events (
                exposure_id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reward_traces (
                reward_id TEXT PRIMARY KEY,
                query_id TEXT,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def upsert_memory(self, entry: MemoryEntry) -> None:
        self.conn.execute(
            """
            INSERT INTO memories(memory_id, user_id, project_id, status, updated_at, data)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                user_id=excluded.user_id,
                project_id=excluded.project_id,
                status=excluded.status,
                updated_at=excluded.updated_at,
                data=excluded.data
            """,
            (
                entry.memory_id,
                entry.user_id,
                entry.project_id,
                entry.status,
                entry.updated_at.isoformat(),
                _dump(entry),
            ),
        )
        self.conn.commit()

    def get_memory(self, memory_id: str) -> MemoryEntry | None:
        row = self.conn.execute("SELECT data FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        return _load(row["data"], MemoryEntry) if row else None

    def get_memories(self, memory_ids: Iterable[str]) -> list[MemoryEntry]:
        return [entry for memory_id in memory_ids if (entry := self.get_memory(memory_id)) is not None]

    def list_memories(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        statuses: set[str] | None = None,
    ) -> list[MemoryEntry]:
        rows = self.conn.execute("SELECT data FROM memories ORDER BY updated_at DESC").fetchall()
        entries = [_load(row["data"], MemoryEntry) for row in rows]
        if user_id is not None:
            entries = [entry for entry in entries if entry.user_id == user_id]
        if project_id is not None:
            entries = [entry for entry in entries if entry.project_id == project_id]
        if statuses is not None:
            entries = [entry for entry in entries if entry.status in statuses]
        return entries

    def add_memory_version(self, entry: MemoryEntry, changed_by_action: str) -> None:
        self.conn.execute(
            """
            INSERT INTO memory_versions(memory_id, version, snapshot, changed_at, changed_by_action)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry.memory_id,
                entry.version,
                _dump(entry),
                utc_now().isoformat(),
                changed_by_action,
            ),
        )
        self.conn.commit()

    def list_memory_versions(self, memory_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT version, snapshot, changed_at, changed_by_action FROM memory_versions WHERE memory_id = ? ORDER BY version_id",
            (memory_id,),
        ).fetchall()
        return [
            {
                "version": row["version"],
                "snapshot": json.loads(row["snapshot"]),
                "changed_at": row["changed_at"],
                "changed_by_action": row["changed_by_action"],
            }
            for row in rows
        ]

    def upsert_cluster(self, cluster: ClusterNode) -> None:
        self.conn.execute(
            """
            INSERT INTO clusters(cluster_id, user_id, project_id, status, updated_at, data)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cluster_id) DO UPDATE SET
                user_id=excluded.user_id,
                project_id=excluded.project_id,
                status=excluded.status,
                updated_at=excluded.updated_at,
                data=excluded.data
            """,
            (
                cluster.cluster_id,
                cluster.user_id,
                cluster.project_id,
                cluster.status,
                cluster.updated_at.isoformat(),
                _dump(cluster),
            ),
        )
        self.conn.commit()

    def get_cluster(self, cluster_id: str) -> ClusterNode | None:
        row = self.conn.execute("SELECT data FROM clusters WHERE cluster_id = ?", (cluster_id,)).fetchone()
        return _load(row["data"], ClusterNode) if row else None

    def list_clusters(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        statuses: set[str] | None = None,
    ) -> list[ClusterNode]:
        rows = self.conn.execute("SELECT data FROM clusters ORDER BY updated_at DESC").fetchall()
        clusters = [_load(row["data"], ClusterNode) for row in rows]
        if user_id is not None:
            clusters = [cluster for cluster in clusters if cluster.user_id == user_id]
        if project_id is not None:
            clusters = [cluster for cluster in clusters if cluster.project_id == project_id]
        if statuses is not None:
            clusters = [cluster for cluster in clusters if cluster.status in statuses]
        return clusters

    def upsert_membership(self, membership: ClusterMembership) -> None:
        self.conn.execute(
            """
            INSERT INTO memberships(membership_id, cluster_id, memory_id, role, score, data)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cluster_id, memory_id, role) DO UPDATE SET
                score=excluded.score,
                data=excluded.data
            """,
            (
                membership.membership_id,
                membership.cluster_id,
                membership.memory_id,
                membership.role,
                membership.score,
                _dump(membership),
            ),
        )
        self.conn.commit()

    def list_memberships(self, *, cluster_id: str | None = None, memory_id: str | None = None) -> list[ClusterMembership]:
        query = "SELECT data FROM memberships"
        params: list[str] = []
        clauses: list[str] = []
        if cluster_id is not None:
            clauses.append("cluster_id = ?")
            params.append(cluster_id)
        if memory_id is not None:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        rows = self.conn.execute(query, params).fetchall()
        return [_load(row["data"], ClusterMembership) for row in rows]

    def upsert_edge(self, edge: MemoryEdge) -> None:
        self.conn.execute(
            """
            INSERT INTO edges(edge_id, src_memory_id, dst_memory_id, relation_type, data)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET data=excluded.data
            """,
            (edge.edge_id, edge.src_memory_id, edge.dst_memory_id, edge.relation_type, _dump(edge)),
        )
        self.conn.commit()

    def list_edges(self, *, memory_id: str | None = None) -> list[MemoryEdge]:
        if memory_id is None:
            rows = self.conn.execute("SELECT data FROM edges").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT data FROM edges WHERE src_memory_id = ? OR dst_memory_id = ?",
                (memory_id, memory_id),
            ).fetchall()
        return [_load(row["data"], MemoryEdge) for row in rows]

    def append_action_trace(self, trace: ActionTrace) -> None:
        self.conn.execute(
            """
            INSERT INTO action_traces(trace_id, phase, action, created_at, data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (trace.trace_id, trace.phase, str(trace.action), trace.created_at.isoformat(), _dump(trace)),
        )
        self.conn.commit()

    def list_action_traces(self) -> list[ActionTrace]:
        rows = self.conn.execute("SELECT data FROM action_traces ORDER BY created_at").fetchall()
        return [_load(row["data"], ActionTrace) for row in rows]

    def append_exposure_event(self, event: ExposureEvent) -> None:
        self.conn.execute(
            """
            INSERT INTO exposure_events(exposure_id, query_id, created_at, data)
            VALUES (?, ?, ?, ?)
            """,
            (event.exposure_id, event.query_id, event.created_at.isoformat(), _dump(event)),
        )
        self.conn.commit()

    def list_exposure_events(self) -> list[ExposureEvent]:
        rows = self.conn.execute("SELECT data FROM exposure_events ORDER BY created_at").fetchall()
        return [_load(row["data"], ExposureEvent) for row in rows]

    def append_reward_trace(self, trace: RewardTrace) -> None:
        self.conn.execute(
            """
            INSERT INTO reward_traces(reward_id, query_id, created_at, data)
            VALUES (?, ?, ?, ?)
            """,
            (trace.reward_id, trace.query_id, trace.created_at.isoformat(), _dump(trace)),
        )
        self.conn.commit()
