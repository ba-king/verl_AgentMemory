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

import importlib.util
import sys
from pathlib import Path


DEMO_PATH = Path(__file__).resolve().parents[2] / "examples" / "agent_memory_demo" / "simple_memory_demo.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("simple_memory_demo", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_memory_demo_writes_then_reuses_memory(tmp_path):
    demo = load_demo_module()
    bank = demo.MemoryBank(tmp_path / "memory_bank.json")
    memory_agent = demo.MemoryAgent(bank)
    main_agent = demo.MockMainAgent()

    first = demo.run_turn(
        "Remember: In this project, answer in concise Chinese unless told otherwise.",
        memory_agent,
        main_agent,
    )
    assert first["actions"][0]["name"] == "WRITE_NEW"

    second = demo.run_turn(
        "How should you answer implementation questions in this project?",
        memory_agent,
        main_agent,
    )
    assert len(second["selected_memories"]) == 1
    assert "concise Chinese" in second["memory_context"]
    assert "I found relevant memory" in second["answer"]
