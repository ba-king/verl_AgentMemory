from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.csl_memory_agent.csl_memory import DARTCSLMemorySystem


def main() -> None:
    mem = DARTCSLMemorySystem()
    mem.add_chunk("Action: Inspect option A. Observation: Option A has attribute alpha, score 4.7, id A1.")
    mem.add_chunk("Action: Select option A as the final choice. Observation: The choice was accepted.")
    prompt = mem.wrap_user_prompt(
        "Continue the task and choose the next compatible option. Prefer the highest score among valid options."
    )
    print(prompt)


if __name__ == "__main__":
    main()
