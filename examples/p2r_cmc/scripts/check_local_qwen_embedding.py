from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.p2r_cmc.memory_system.model_clients import LocalQwenEmbeddingClient


def main() -> None:
    client = LocalQwenEmbeddingClient.from_env()
    texts = [
        "P2R-CMC first stage is prompt-based.",
        "第一阶段先跑通 prompt-based memory controller。",
    ]
    vectors = client.embed_texts(texts)
    print(
        json.dumps(
            {
                "model_path": str(client.model_path),
                "device": client.device,
                "num_vectors": len(vectors),
                "dim": len(vectors[0]) if vectors else 0,
                "preview": vectors[0][:5] if vectors else [],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
