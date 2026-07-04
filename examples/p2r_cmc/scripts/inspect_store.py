from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.p2r_cmc.memory_system.export import export_store_json
from examples.p2r_cmc.memory_system.store import SQLiteMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--dump-json", default=None)
    parser.add_argument("--include-embeddings", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"SQLite database not found: {db_path}")

    store = SQLiteMemoryStore(db_path)
    payload = export_store_json(store, include_embeddings=args.include_embeddings)
    store.close()

    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    if args.dump_json:
        dump_path = Path(args.dump_json)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON dump: {dump_path}")


if __name__ == "__main__":
    main()
