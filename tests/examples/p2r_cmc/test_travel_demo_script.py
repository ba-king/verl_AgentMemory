from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_travel_demo_script_runs_without_api(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[3]
    db_path = tmp_path / "travel_demo.sqlite"
    dump_path = tmp_path / "travel_demo_dump.json"

    result = subprocess.run(
        [
            sys.executable,
            "examples/p2r_cmc/scripts/run_p2r_cmc_travel_demo.py",
            "--embedding",
            "none",
            "--db-path",
            str(db_path),
            "--dump-json",
            str(dump_path),
            "--reset",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "P2R-CMC travel planning demo" in result.stdout
    assert "memory_count_before_session_end: 0" in result.stdout
    assert "STEP 5: Query-time retrieval expands relevant clusters" in result.stdout
    assert "DETERMINISTIC_FALLBACK_ANSWER" in result.stdout
    assert db_path.exists()
    assert dump_path.exists()

    payload = json.loads(dump_path.read_text(encoding="utf-8"))
    assert payload["counts"]["memories"] >= 6
    assert payload["counts"]["clusters"] >= 1
    assert payload["counts"]["exposure_events"] == 1
