"""Localhost server mirroring TypeSafe's `POST /v1/systemone` request/response shape.

The model is loaded once, when the server starts, and shared by every request.

    .venv/bin/jevlite-serve --model models/decision-model --port 8000
    JEVLITE_MODEL=models/decision-model .venv/bin/uvicorn jevlite.api:app --host 127.0.0.1 --port 8000

Environment:
    JEVLITE_MODEL   path to the model folder (required)
    JEVLITE_DEVICE  cuda | mps | cpu (optional; picked automatically)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .model import SystemOne
from .schema import validate_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("JEVLITE_MODEL")
    if not path:
        raise RuntimeError("set JEVLITE_MODEL to the model folder, e.g. JEVLITE_MODEL=models/decision-model")
    started = time.perf_counter()
    app.state.engine = SystemOne(path, device=os.environ.get("JEVLITE_DEVICE") or None)
    engine = app.state.engine
    print(f"jevlite: loaded {engine.name} on {engine.device} in {time.perf_counter() - started:.1f}s", flush=True)
    yield


app = FastAPI(title="jevlite", lifespan=lifespan)


class SystemOneRequest(BaseModel):
    state: str | dict | list
    model: str = "jevlite"
    questions: dict[str, dict[str, Any]]


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest, request: Request) -> dict:
    if not req.questions:
        raise HTTPException(status_code=422, detail=["questions must contain at least one question"])
    errors = [e for qid, q in req.questions.items() for e in validate_question(qid, q)]
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return request.app.state.engine.system_one(req.state, req.questions)


@app.get("/v1/models")
def models(request: Request) -> dict:
    engine = request.app.state.engine
    return {"models": [{"name": engine.name, "description": "Jev-style System One decision model (Choice / Score / Noul)",
                        "device": engine.device}]}


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model": request.app.state.engine.name}


def main() -> None:
    """`jevlite-serve --model models/decision-model --port 8000` (installed with `pip install -e '.[serve]'`)."""
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Run the Jev-like /v1/systemone server")
    ap.add_argument("--model", default=os.environ.get("JEVLITE_MODEL"), help="model folder (or set JEVLITE_MODEL)")
    ap.add_argument("--device", default=os.environ.get("JEVLITE_DEVICE"), choices=["cuda", "mps", "cpu"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not args.model:
        ap.error("--model is required (or set JEVLITE_MODEL)")
    os.environ["JEVLITE_MODEL"] = args.model
    if args.device:
        os.environ["JEVLITE_DEVICE"] = args.device
    uvicorn.run(app, host=args.host, port=args.port)
