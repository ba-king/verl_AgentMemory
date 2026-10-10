"""Memora adapter, single-branch replay, and content-verified provenance.

The Store's IDs are execution handles only. Logical roots and evidence predicates
live in an audit sidecar; no production protocol or MemoryRecord is extended.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ...memory_agent.action.protocol import (
    AddArguments,
    DeleteArguments,
    MemoryAction,
    MemoryActionName,
    UpdateArguments,
)
from ...memory_agent.bank.bank_state import MemoryBankState
from ...memory_agent.bank.store import MemoryBankStore
from ...memory_agent.executor.engine import MemoryActionExecutor


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def leaves(obj, prefix=()):
    """Lists represent sets of facts, not index-sensitive slots."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, (*prefix, key))
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (list, dict)):
                raise ValueError("unsupported_nested_list")
            yield (prefix, value)
    else:
        yield (prefix, obj)


def contains(payload, path, value):
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return value in current if isinstance(current, list) else current == value


class ReplayError(ValueError):
    def __init__(self, category, session, detail=""):
        super().__init__(f"{category}: session={session} {detail}")
        self.category, self.session, self.detail = category, session, detail

    def to_dict(self):
        return {"category": self.category, "session_id": self.session, "detail": self.detail}


@dataclass
class State:
    bank: MemoryBankState = field(default_factory=MemoryBankState.empty)
    roots: dict = field(default_factory=dict)  # runtime ID -> original logical item
    lineage: dict = field(default_factory=dict)  # runtime ID -> actual source sessions
    shadows: set = field(default_factory=set)  # duplicate stale records have no gold execution target

    def clone(self):
        return copy.deepcopy(self)

    def records(self):
        return [
            {
                "memory_id": r.memory_id,
                "content": r.content,
                "logical_root": self.roots[r.memory_id],
                "source_sessions": self.lineage[r.memory_id],
            }
            for r in self.bank.records
        ]

    def by_root(self, root):
        records = [r for r in self.bank.records if self.roots[r.memory_id] == root and r.memory_id not in self.shadows]
        if len(records) != 1:
            raise ReplayError("ambiguous_or_missing_logical_target", -1, root)
        return records[0]


def mutate(state, operation, payload, root, sid, memory_id=None):
    """Every mutation passes through the current atomic executor."""
    args = (
        AddArguments(canonical(payload))
        if operation == "ADD"
        else UpdateArguments(memory_id, canonical(payload))
        if operation == "UPDATE"
        else DeleteArguments(memory_id)
    )
    executor = MemoryActionExecutor(store=MemoryBankStore(state.bank))
    result = executor.execute(MemoryAction(MemoryActionName(operation), args))
    if result.status.value == "REJECTED":
        raise ReplayError("executor_rejected", sid, result.code.value)
    state.bank = executor.state
    mid = result.record.memory_id
    if operation == "DELETE":
        state.roots.pop(mid)
        state.lineage.pop(mid)
    else:
        if operation == "ADD" and mid in state.roots:
            raise ReplayError("duplicate_add_identity_collision", sid, root)
        state.roots[mid] = root
        state.lineage[mid] = sorted(set(state.lineage.get(mid, []) + [sid]))
    return mid


def normalized(s):
    d, kind = s["operation_details"], s["session_type"]
    if "content_data" in d:
        return {"kind": "content", "item": d["item"], "data": d["content_data"]}
    if kind == "preference":
        return {"kind": kind, "subcategory": d["subcategory"], "item": d["item"], "preference": d["preference"]}
    if kind == "activity" and isinstance(d.get("item"), dict):
        return {"kind": kind, "category": d["category"], "data": d["item"]}
    if kind == "goal":
        return {"kind": kind, "subcategory": d["subcategory"], "value": d["item"]}
    raise ReplayError("unsupported_operation_schema", s["session_id"], canonical(d))


def identity(payload):
    if payload["kind"] == "content":
        return {"kind": "content", "item": payload["item"]}
    if payload["kind"] == "preference":
        return {k: payload[k] for k in ("kind", "subcategory", "item")}
    if payload["kind"] == "goal":
        return {k: payload[k] for k in ("kind", "subcategory")}
    d = payload["data"]
    keys = [k for k in ("description", "event_name", "created_at") if k in d]
    if not keys:
        keys = list(d)
    return {"kind": payload["kind"], "category": payload["category"], "data": {k: d[k] for k in keys}}


def matches(payload, predicate):
    return all(contains(payload, p, v) for p, v in leaves(predicate))


def changes(before, after, path=()):
    result = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        p = (*path, key)
        if old == new:
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            result.extend(changes(old, new, p))
        elif isinstance(old, list) and isinstance(new, list):
            result.append(
                {
                    "path": p,
                    "mode": "list_delta",
                    "remove": [x for x in old if x not in new],
                    "add": [x for x in new if x not in old],
                }
            )
        else:
            result.append({"path": p, "mode": "unset" if key not in after else "set", "value": new})
    return result


def patch_payload(payload, patches, sid):
    result = copy.deepcopy(payload)
    for patch in patches:
        node = result
        for key in patch["path"][:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise ReplayError("inconsistent_state", sid, "missing parent field")
            node = node[key]
        key = patch["path"][-1]
        if patch["mode"] == "set":
            node[key] = copy.deepcopy(patch["value"])
        elif patch["mode"] == "unset":
            node.pop(key, None)
        else:
            if not isinstance(node.get(key), list):
                raise ReplayError("inconsistent_state", sid, "missing list")
            node[key] = [v for v in node[key] if v not in patch["remove"]]
            node[key].extend(v for v in patch["add"] if v not in node[key])
    return result


def compile_transition(s, state):
    sid, op = s["session_id"], s["operation"]
    row = {
        "session_id": sid,
        "dataset_operation": op,
        "atomic_operation": None,
        "root": None,
        "before": None,
        "after": None,
        "patches": [],
    }
    if op is None:
        return row
    if op not in {"add", "update", "delete"}:
        raise ReplayError("unsupported_operation_replay", sid, str(op))
    payload = normalized(s)
    if op == "add":
        row.update(atomic_operation="ADD", root=f"session:{sid}:item", after=payload)
        return row
    target = identity(payload)
    if payload["kind"] == "preference":
        target["item"] = s["operation_details"].get("old_item", payload["item"])
    found = [r for r in state.bank.records if matches(json.loads(r.content), target)]
    if len(found) != 1:
        raise ReplayError("target_memory_missing" if not found else "ambiguous_target", sid, canonical(target))
    record = found[0]
    old = json.loads(record.content)
    # Content 'delete' means field deletion, NOT deleting the entire record.
    atomic = "DELETE" if op == "delete" and payload["kind"] != "content" else "UPDATE"
    row.update(
        atomic_operation=atomic,
        root=state.roots[record.memory_id],
        before=old,
        after=None if atomic == "DELETE" else payload,
        target_identity=identity(old),
        target_memory_id=record.memory_id,
    )
    if atomic == "UPDATE":
        row["patches"] = changes(old, payload)
    return row


def apply_transition(state, row):
    op, sid = row["atomic_operation"], row["session_id"]
    if op is None:
        return
    if op == "ADD":
        mutate(state, op, row["after"], row["root"], sid)
        return
    try:
        record = state.by_root(row["root"])
    except ReplayError as error:
        raise ReplayError("required_prior_add_removed_or_target_missing", sid, row["root"]) from error
    old = json.loads(record.content)
    if not matches(old, row["target_identity"]):
        raise ReplayError("target_identity_dependency_broken", sid, row["root"])
    payload = None if op == "DELETE" else patch_payload(old, row["patches"], sid)
    mutate(state, op, payload, row["root"], sid, record.memory_id)


def replay(state, transitions, start, end):
    state = state.clone()
    for row in transitions:
        if start < row["session_id"] <= end:
            apply_transition(state, row)
    return state


def load_dataset(root, persona, setting, max_sessions):
    directory = Path(root) / "data" / setting / persona
    if not (Path(root) / "data_generation" / "session_simulator.py").is_file():
        raise ValueError("Expected the official geniesinc/Memora schema")
    sessions = sorted(
        [json.loads(p.read_text()) for p in (directory / "conversations").glob("*.json")], key=lambda s: s["session_id"]
    )
    if not sessions:
        raise ValueError("No sessions found")
    sessions = sessions[:max_sessions]
    if len({s["session_id"] for s in sessions}) != len(sessions):
        raise ValueError("Duplicate session IDs")
    qpath = directory / f"evaluation_questions_{persona}.json"
    raw = json.loads(qpath.read_text())
    qas, out_of_horizon = [], []
    for group, questions in raw["questions"].items():
        for q in questions:
            q = copy.deepcopy(q)
            # Official IDs put the evaluation session after the question prefix;
            # content item suffixes can also contain numbers, hence FIRST numeric token.
            tokens = re.findall(r"(?:^|_)(\d+)(?=_|$)", q["question_id"])
            if not tokens:
                raise ValueError("Unrecognized QA horizon: " + q["question_id"])
            q["at_session"] = int(tokens[0])
            q["group"] = group
            (qas if q["at_session"] <= sessions[-1]["session_id"] else out_of_horizon).append(q)
    return sessions, qas, out_of_horizon


def build_gold(sessions):
    state, transitions, snapshots, errors = State(), [], {0: State()}, []
    for s in sessions:
        try:
            row = compile_transition(s, state)
            apply_transition(state, row)
            if row["after"] is not None:
                actual = json.loads(state.by_root(row["root"]).content)
                if actual != row["after"]:
                    raise ReplayError("gold_snapshot_mismatch", s["session_id"])
            transitions.append(row)
            snapshots[s["session_id"]] = state.clone()
        except (ReplayError, KeyError, TypeError) as error:
            errors.append(
                error.to_dict()
                if isinstance(error, ReplayError)
                else {"category": "unsupported_schema", "session_id": s["session_id"], "detail": str(error)}
            )
            break  # never silently skip broken gold evolution
    return transitions, snapshots, errors


def evidence_id(root, predicates):
    return "fact:" + digest([root, predicates])[:24]


def register_evidence(registry, root, predicates, raw, rows, *, stale_session=None):
    predicates = sorted([[list(p), v] for p, v in predicates], key=canonical)
    eid = evidence_id(root, predicates)
    supports = lambda obj: obj is not None and all(contains(obj, p, v) for p, v in predicates)
    origins = [
        r["session_id"] for r in rows if r["root"] == root and supports(r["after"]) and not supports(r["before"])
    ]
    entry = registry.setdefault(
        eid,
        {
            "evidence_id": eid,
            "root": root,
            "predicates": predicates,
            "introduced_sessions": origins,
            "raw_refs": [],
            "stale_event_sessions": [],
        },
    )
    if raw not in entry["raw_refs"]:
        entry["raw_refs"].append(raw)
    if stale_session is not None and stale_session not in entry["stale_event_sessions"]:
        entry["stale_event_sessions"].append(stale_session)
    return eid


def map_evidence(qas, rows, snapshots):
    """Validate official evidence against actual replayed values, never session ID alone."""
    registry, mapped, errors, rowmap = {}, [], [], {r["session_id"]: r for r in rows}
    for q in qas:
        q = copy.deepcopy(q)
        q["positive_evidence_ids"], q["stale_evidence_ids"] = [], []
        local_errors = []
        if q["at_session"] not in snapshots:
            errors.append({"question_id": q["question_id"], "error": "gold_horizon_unavailable"})
            continue
        state = snapshots[q["at_session"]]
        evidence = q["memory_evidence"]

        def add(root, predicates, raw, *, state=state, q=q, local_errors=local_errors):
            try:
                r = state.by_root(root)
                obj = json.loads(r.content)
                if not all(contains(obj, p, v) for p, v in predicates):
                    raise ValueError("official_positive_disagrees_with_replayed_values")
                eid = register_evidence(registry, root, predicates, raw, rows)
                if not registry[eid]["introduced_sessions"]:
                    raise ValueError("positive_has_no_introduction")
                q["positive_evidence_ids"].append(eid)
            except (ReplayError, ValueError) as error:
                local_errors.append({"raw": raw, "error": str(error)})

        if "content_data" in evidence:
            roots = {rowmap[s]["root"] for s in evidence["session_history"] if s in rowmap}
            if len(roots) != 1:
                local_errors.append({"error": "content_history_ambiguous"})
            else:
                root = next(iter(roots))
                for p, v in leaves(evidence["content_data"], ("data",)):
                    add(root, [(p, v)], {"path": list(p), "value": v, "history": evidence["session_history"]})
        else:

            def visit(value, path=(), *, local_errors=local_errors, evidence=evidence, add=add):
                if isinstance(value, dict):
                    if "session_id" in value:
                        sid = value["session_id"]
                        if sid not in rowmap or rowmap[sid]["root"] is None:
                            local_errors.append({"raw": value, "error": "positive_session_unmapped"})
                            return
                        root = rowmap[sid]["root"]
                        old = rowmap[sid]["after"]
                        if old is None:
                            local_errors.append({"raw": value, "error": "positive_points_to_delete"})
                            return
                        if old["kind"] == "preference":
                            polarity = "dislike" if "dislikes" in path else "like"
                            # specific_items duplicates subcategory_data; use the latter's polarity.
                            if "specific_items" in path:
                                return
                            add(root, [(("item",), value["item"]), (("preference",), polarity)], value)
                        elif old["kind"] == "goal":
                            add(root, [(("value",), evidence["goal_value"])], value)
                        else:
                            predicates = []
                            for key, val in value.items():
                                if key in {"session_id", "session_date"}:
                                    continue
                                key = "description" if key == "value" and "remaining_tasks" in path else key
                                predicates.append((("data", key), val))
                            if not predicates:
                                local_errors.append({"raw": value, "error": "empty_positive_predicate"})
                            else:
                                add(root, predicates, value)
                        return
                    for key, item in value.items():
                        visit(item, (*path, key))
                elif isinstance(value, list):
                    for item in value:
                        visit(item, path)

            visit(evidence)

        for raw in q.get("forgetting_evidence", {}).get("forgotten_items", []):
            sid = raw["session_id"]
            row = rowmap.get(sid)
            if row is None or row["before"] is None:
                local_errors.append({"raw": raw, "error": "stale_event_unmapped"})
                continue
            before = row["before"]
            if "field" in raw:
                predicates = [(("data", raw["field"]), raw["value"])]
            elif before["kind"] == "preference":
                predicates = [(("item",), raw["value"]), (("preference",), before["preference"])]
            else:
                predicates = [(("data", "description"), raw["value"])]
            if not all(contains(before, p, v) for p, v in predicates):
                local_errors.append({"raw": raw, "error": "stale_value_not_in_prestate"})
                continue
            if row["after"] is not None and all(contains(row["after"], p, v) for p, v in predicates):
                local_errors.append({"raw": raw, "error": "stale_value_not_removed"})
                continue
            eid = register_evidence(registry, row["root"], predicates, raw, rows, stale_session=sid)
            q["stale_evidence_ids"].append(eid)
        q["positive_evidence_ids"] = sorted(set(q["positive_evidence_ids"]))
        q["stale_evidence_ids"] = sorted(set(q["stale_evidence_ids"]))
        q["mapping_errors"] = local_errors
        q["mapping_valid"] = not local_errors and bool(q["positive_evidence_ids"])
        errors.extend({"question_id": q["question_id"], **e} for e in local_errors)
        mapped.append(q)
    return mapped, registry, errors


def source_refs(state, registry):
    refs = {}
    for r in state.bank.records:
        try:
            payload = json.loads(r.content)
        except (ValueError, TypeError):
            refs[r.memory_id] = []  # no general free-text grounding verifier
            continue
        refs[r.memory_id] = [
            eid
            for eid, e in registry.items()
            if state.roots[r.memory_id] == e["root"] and all(contains(payload, p, v) for p, v in e["predicates"])
        ]
    return refs


def annotated_state(state, registry):
    refs = source_refs(state, registry)
    return [{**r, "source_refs": refs[r["memory_id"]]} for r in state.records()]


def raw_session_ids(value):
    result = set()
    if isinstance(value, dict):
        if isinstance(value.get("session_id"), int):
            result.add(value["session_id"])
        if "session_history" in value:
            result.update(x for x in value["session_history"] if isinstance(x, int))
        for v in value.values():
            result.update(raw_session_ids(v))
    elif isinstance(value, list):
        for v in value:
            result.update(raw_session_ids(v))
    return result


def distribution(values):
    import statistics

    values = list(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "histogram": dict(sorted(Counter(values).items())),
    }
