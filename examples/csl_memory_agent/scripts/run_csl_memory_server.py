from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from examples.csl_memory_agent.csl_memory import DARTCSLMemoryArenaMemorySystem


class InitializeRequest(BaseModel):
    user_id: str
    memory_system_name: str


class AddRequest(InitializeRequest):
    chunk: str


class QueryRequest(InitializeRequest):
    question: str


class GuardRequest(QueryRequest):
    action: str


class SelectionAuditRequest(QueryRequest):
    selected_option_id: str


app = FastAPI(title="DART-CSL Memory Server")
systems: dict[str, DARTCSLMemoryArenaMemorySystem] = {}


@app.post("/memory/initialize")
def initialize(request: InitializeRequest) -> dict[str, Any]:
    if request.memory_system_name != "dart_csl_vtcc":
        raise HTTPException(status_code=400, detail="This server only provides dart_csl_vtcc")
    systems[request.user_id] = DARTCSLMemoryArenaMemorySystem(
        user_id=request.user_id,
        config={
            "state_dir": os.getenv("CSL_STATE_DIR", ".csl_state"),
            "llm": {
                "provider": os.getenv("CSL_MEMORY_PROVIDER", "mock"),
                "model": os.getenv("CSL_MEMORY_MODEL", "mock"),
                "api_key": os.getenv("CSL_MEMORY_API_KEY"),
                "base_url": os.getenv("CSL_MEMORY_BASE_URL"),
                "temperature": 0,
                "max_retries": 3,
            },
        },
    )
    return {"status": "ok", "user_id": request.user_id, "memory_system_name": request.memory_system_name}


@app.post("/memory/add")
def add(request: AddRequest) -> dict[str, Any]:
    system = _get(request.user_id)
    return {"status": "ok", "user_id": request.user_id, "response": system.add_chunk(request.chunk)}


@app.post("/memory/wrap_user_prompt")
def wrap_user_prompt(request: QueryRequest) -> dict[str, Any]:
    system = _get(request.user_id)
    return {"status": "ok", "user_id": request.user_id, "prompt": system.wrap_user_prompt(request.question)}


@app.post("/memory/guard_action")
def guard_action(request: GuardRequest) -> dict[str, Any]:
    system = _get(request.user_id)
    return {"status": "ok", "result": system.guard_action(request.question, request.action)}


@app.post("/memory/validate_transition")
def validate_transition(request: GuardRequest) -> dict[str, Any]:
    system = _get(request.user_id)
    return {
        "status": "ok",
        "result": system.validate_transition(request.question, request.action),
    }


@app.post("/memory/audit_selection")
def audit_selection(request: SelectionAuditRequest) -> dict[str, Any]:
    system = _get(request.user_id)
    return {
        "status": "ok",
        "result": system.audit_selection(request.question, request.selected_option_id),
    }


@app.get("/memory/metrics/{user_id}")
def metrics(user_id: str) -> dict[str, Any]:
    return {"status": "ok", "user_id": user_id, "metrics": _get(user_id).get_metrics()}


def _get(user_id: str) -> DARTCSLMemoryArenaMemorySystem:
    if user_id not in systems:
        raise HTTPException(status_code=404, detail="User not initialized")
    return systems[user_id]


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("CSL_MEMORY_HOST", "127.0.0.1"),
        port=int(os.getenv("CSL_MEMORY_PORT", "8000")),
    )
