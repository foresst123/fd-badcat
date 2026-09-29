"""Model-agnostic HTTP/NDJSON gateway for Kaggle or another GPU server."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from model_providers.runtime import get_mllm_capabilities


def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except (RuntimeError, ValueError):
            pass


@dataclass
class _ActiveResponse:
    cancelled: Event = field(default_factory=Event)
    stream: Any | None = None
    lock: Lock = field(default_factory=Lock)

    def cancel(self) -> None:
        self.cancelled.set()
        with self.lock:
            stream = self.stream
        if stream is not None:
            _close_stream(stream)


def create_mllm_gateway(
    provider: Any,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
) -> FastAPI:
    """Expose any conforming in-process provider over a stable API."""

    if provider is None:
        raise ValueError("MLLM gateway yêu cầu provider")
    expected_key = (
        os.getenv("MLLM_GATEWAY_API_KEY", "")
        if api_key is None
        else str(api_key)
    ).strip()
    app = FastAPI(title="FD-BADCAT MLLM Gateway", version="1")
    active: dict[str, _ActiveResponse] = {}
    active_lock = Lock()

    def authorize(request: Request) -> None:
        if not expected_key:
            return
        authorization = request.headers.get("authorization", "")
        supplied = (
            authorization[7:]
            if authorization.lower().startswith("bearer ")
            else request.headers.get("x-api-key", "")
        )
        if not secrets.compare_digest(supplied, expected_key):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        authorize(request)
        return {
            "status": "ready",
            "provider": str(getattr(provider, "provider_name", "unknown")),
            "model": model_name,
            "capabilities": get_mllm_capabilities(provider).as_dict(),
            "active_responses": len(active),
        }

    @app.get("/v1/capabilities")
    def capabilities(request: Request) -> dict[str, Any]:
        authorize(request)
        return get_mllm_capabilities(provider).as_dict()

    @app.post("/v1/decision")
    def decision(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        authorize(request)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=422, detail="messages must be a list")
        decide = getattr(provider, "decide", None)
        value = (
            decide(messages) if callable(decide) else provider.generate(messages)
        )
        value = str(value).strip().lower()
        if value not in {"continue", "switch"}:
            raise HTTPException(
                status_code=502,
                detail=f"provider returned invalid decision: {value!r}",
            )
        return {
            "type": "decision.result",
            "request_id": payload.get("request_id"),
            "decision": value,
        }

    @app.post("/v1/generate")
    def generate(payload: dict[str, Any], request: Request):
        authorize(request)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=422, detail="messages must be a list")
        request_id = str(payload.get("request_id") or f"response-{uuid4().hex}")
        item = _ActiveResponse()
        with active_lock:
            if request_id in active:
                raise HTTPException(status_code=409, detail="duplicate request_id")
            active[request_id] = item

        def events():
            try:
                stream = provider.stream_generate(messages)
                with item.lock:
                    item.stream = stream
                for delta in stream:
                    if item.cancelled.is_set():
                        break
                    if not isinstance(delta, str) or not delta:
                        continue
                    yield json.dumps({
                        "type": "response.delta",
                        "request_id": request_id,
                        "text": delta,
                    }, ensure_ascii=False) + "\n"
                if not item.cancelled.is_set():
                    yield json.dumps({
                        "type": "response.done",
                        "request_id": request_id,
                    }, ensure_ascii=False) + "\n"
            except GeneratorExit:
                item.cancelled.set()
                raise
            except Exception as exc:
                yield json.dumps({
                    "type": "response.error",
                    "request_id": request_id,
                    "message": str(exc),
                }, ensure_ascii=False) + "\n"
            finally:
                item.cancelled.set()
                with item.lock:
                    stream = item.stream
                    item.stream = None
                if stream is not None:
                    _close_stream(stream)
                with active_lock:
                    active.pop(request_id, None)

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post("/v1/cancel/{request_id}")
    def cancel(request_id: str, request: Request) -> dict[str, Any]:
        authorize(request)
        with active_lock:
            item = active.get(request_id)
        if item is None:
            return {"request_id": request_id, "cancelled": False}
        item.cancel()
        return {"request_id": request_id, "cancelled": True}

    app.state.mllm_provider = provider
    app.state.active_responses = active
    return app
