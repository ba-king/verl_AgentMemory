from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.csl_memory_agent.csl_memory.benchmark import run_baselines


def main() -> None:
    print(json.dumps(run_baselines(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
