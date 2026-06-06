# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal agent-memory demo.

This script intentionally keeps memory rule-based and training-free. It mirrors
the simplest useful slice of an agent-memory flow:

1. Before answering: retrieve memories and expose them as context.
2. Main agent answers with that context.
3. After answering: extract candidate memories and write them to the memory bank.

Run:
    python examples/agent_memory_demo/simple_memory_demo.py --reset --demo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


@dataclass
class MemoryEntry:
    """The minimal memory schema needed for a runnable demo."""

    id: str
    content: str
    source: str
    created_at: float = field(default_factory=time.time)


@dataclass
class MemoryCandidate:
    content: str
    source: str


@dataclass
class MemoryAction:
    name: str
    candidate: MemoryCandidate | None = None
    reason: str = ""


class MemoryBank:
    """JSON-backed memory bank with simple lexical retrieval."""

    def __init__(self, path: Path):
        self.path = path
        self.memories: list[MemoryEntry] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.memories = []
            return
        raw_memories = json.loads(self.path.read_text(encoding="utf-8"))
        self.memories = [MemoryEntry(**item) for item in raw_memories]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(memory) for memory in self.memories]
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def retrieve(self, query: str, *, top_k: int = 3) -> list[MemoryEntry]:
        scored = []
        query_terms = tokenize(query)
        for memory in self.memories:
            memory_terms = tokenize(memory.content)
            overlap = len(query_terms & memory_terms)
            if overlap == 0:
                continue
            scored.append((overlap, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[:top_k]]

    def write_new(self, candidate: MemoryCandidate) -> MemoryEntry:
        memory = MemoryEntry(
            id=uuid4().hex,
            content=candidate.content,
            source=candidate.source,
            created_at=time.time(),
        )
        self.memories.append(memory)
        return memory


class MemoryAgent:
    """Rule-based memory agent for retrieve/expose and write."""

    def __init__(self, bank: MemoryBank, *, top_k: int = 3):
        self.bank = bank
        self.top_k = top_k

    def prepare_context(self, query: str) -> tuple[str, list[MemoryEntry]]:
        selected = self.bank.retrieve(query, top_k=self.top_k)
        if not selected:
            return "", []

        lines = ["Relevant memories:"]
        for idx, memory in enumerate(selected, start=1):
            lines.append(f"{idx}. {memory.content}")
        return "\n".join(lines), selected

    def maintain_after_turn(self, user_query: str, answer: str) -> list[MemoryAction]:
        candidates = extract_memory_candidates(user_query, answer)
        actions = [
            MemoryAction(name="WRITE_NEW", candidate=candidate, reason="User explicitly asked to remember this.")
            for candidate in candidates
        ]
        for action in actions:
            if action.name == "WRITE_NEW" and action.candidate is not None:
                self.bank.write_new(action.candidate)
        self.bank.save()
        return actions or [MemoryAction(name="NO_WRITE", reason="No durable candidate memory was extracted.")]


class MainAgent(Protocol):
    def answer(self, query: str, memory_context: str) -> str:
        """Generate an answer for the user query."""


class MockMainAgent:
    """Deterministic main agent for local smoke tests and demos."""

    def answer(self, query: str, memory_context: str) -> str:
        if memory_context:
            return (
                "I found relevant memory and will use it before answering.\n\n"
                f"{memory_context}\n\n"
                f"Answer: for this demo query, I would respond to: {query}"
            )
        return f"Answer: I have no relevant memory yet, so I will handle the query directly: {query}"


class OpenAICompatibleMainAgent:
    """Tiny stdlib client for OpenAI-compatible chat completions."""

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def answer(self, query: str, memory_context: str) -> str:
        system_prompt = "You are the main agent. Use provided memories only when they are relevant."
        user_content = query if not memory_context else f"{memory_context}\n\nUser query: {query}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat completion failed with HTTP {exc.code}: {detail}") from exc
        return body["choices"][0]["message"]["content"]


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", text.lower()))


def extract_memory_candidates(user_query: str, answer: str) -> list[MemoryCandidate]:
    """Extract explicit memories from the user turn.

    This keeps the demo honest: memory writes happen only when the user marks
    something durable with "remember:" or "memory:".
    """

    del answer  # Future learned extractors can use the answer too.
    candidates: list[MemoryCandidate] = []
    pattern = re.compile(r"(?:remember|memory|记住|记忆)[:：]\s*(.+)", re.IGNORECASE)
    for match in pattern.finditer(user_query):
        content = match.group(1).strip()
        if content:
            candidates.append(
                MemoryCandidate(
                    content=content,
                    source=f"user_query:{int(time.time())}",
                )
            )
    return candidates


def run_turn(query: str, memory_agent: MemoryAgent, main_agent: MainAgent) -> dict[str, Any]:
    memory_context, selected_memories = memory_agent.prepare_context(query)
    answer = main_agent.answer(query, memory_context)
    actions = memory_agent.maintain_after_turn(query, answer)
    return {
        "query": query,
        "selected_memories": [asdict(memory) for memory in selected_memories],
        "memory_context": memory_context,
        "answer": answer,
        "actions": [asdict(action) for action in actions],
    }


def build_main_agent(args: argparse.Namespace) -> MainAgent:
    if args.backend == "mock":
        return MockMainAgent()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before using --backend openai.")
    return OpenAICompatibleMainAgent(base_url=args.base_url, api_key=api_key, model=args.model)


def print_turn(result: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"USER: {result['query']}")
    print("-" * 80)
    if result["memory_context"]:
        print(result["memory_context"])
    else:
        print("Relevant memories: <none>")
    print("-" * 80)
    print(result["answer"])
    print("-" * 80)
    print("Memory actions:")
    for action in result["actions"]:
        print(f"- {action['name']}: {action['reason']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=Path("examples/agent_memory_demo/memory_bank.json"),
        help="Path to the JSON memory bank.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of memories exposed to the main agent.")
    parser.add_argument("--reset", action="store_true", help="Delete the memory bank before running.")
    parser.add_argument("--demo", action="store_true", help="Run a two-turn demo that writes then reuses memory.")
    parser.add_argument("--query", action="append", help="User query. Can be passed multiple times.")
    parser.add_argument("--backend", choices=["mock", "openai"], default="mock", help="Main-agent backend.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reset and args.memory_path.exists():
        args.memory_path.unlink()

    queries = args.query or []
    if args.demo:
        queries.extend(
            [
                "Remember: In this project, answer in concise Chinese unless the user asks otherwise.",
                "For this project, how should you answer implementation questions?",
            ]
        )
    if not queries:
        raise SystemExit("Pass --demo or at least one --query.")

    bank = MemoryBank(args.memory_path)
    memory_agent = MemoryAgent(bank, top_k=args.top_k)
    main_agent = build_main_agent(args)

    print(
        textwrap.dedent(
            f"""
            Minimal agent-memory demo
            memory_agent: rule-based retrieve/expose/extract/write
            main_agent: {args.backend}
            memory_bank: {args.memory_path}
            """
        ).strip()
    )

    for query in queries:
        result = run_turn(query, memory_agent, main_agent)
        print_turn(result)


if __name__ == "__main__":
    main()
