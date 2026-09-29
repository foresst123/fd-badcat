#!/usr/bin/env python3
"""Run Vi-FDB audio cases through the fd-badcat realtime websocket.

This runner deliberately keeps the benchmark clock separate from model
inference time: input frames are sent at their absolute audio timestamps and
the returned assistant audio is placed on the same timeline using the
``tts_done`` event emitted by ``src/backend.py``.

It produces the artifact contract needed by the Vi-FDB handoff document:

    <run-root>/<task>/<id>/output.wav
    <run-root>/<task>/<id>/output_timing.json
    <run-root>/<task>/<id>/clean_output.wav       # paired controls
    <run-root>/<task>/<id>/clean_output_timing.json

Semantic judging remains a separate step.  The runner never sends metadata,
expected actions, or event annotations to the model.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import websockets


SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 256  # 16 ms, matching fd-badcat/src/frontend.py


def _manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("records") or payload.get("items") or payload.get("data")
    else:
        records = None
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Unsupported manifest shape: {manifest_path}")
    return records


def _resolve_audio(dataset_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = dataset_root / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_audio(path: Path) -> np.ndarray:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate}")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32)


def _decode_audio_message(message: bytes) -> np.ndarray:
    """Decode either legacy WAV bytes or the MiniCPM branch's raw PCM16."""
    try:
        data, sample_rate = sf.read(
            io.BytesIO(message), dtype="float32", always_2d=False
        )
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"assistant audio has unexpected sample rate {sample_rate}"
            )
        if data.ndim == 2:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32)
    except Exception:
        if len(message) % 2:
            raise ValueError("raw PCM16 audio has an odd byte count")
        return (
            np.frombuffer(message, dtype="<i2").astype(np.float32) / 32768.0
        )


def _event_timestamp(obj: dict[str, Any]) -> float | None:
    value = obj.get("data", {}).get("timestamp")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def _run_case(
    *,
    ws_url: str,
    dataset_root: Path,
    run_root: Path,
    record: dict[str, Any],
    condition: str,
    exp_name: str,
    tail_seconds: float,
    close_grace_seconds: float,
) -> dict[str, Any]:
    task = str(record.get("task", "unknown_task"))
    case_id = str(record.get("id", "unknown_id"))
    audio_key = "input" if condition == "event" else "clean_input"
    audio_path = _resolve_audio(dataset_root, record.get(audio_key))
    if audio_path is None:
        return {"task": task, "id": case_id, "condition": condition, "status": "skipped", "reason": f"missing {audio_key}"}

    input_audio = _read_audio(audio_path)
    input_duration = len(input_audio) / SAMPLE_RATE
    case_dir = run_root / task / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    stem = "output" if condition == "event" else "clean_output"
    output_path = case_dir / f"{stem}.wav"
    timing_path = case_dir / f"{stem}_timing.json"

    events: list[dict[str, Any]] = []
    pending_tts_timestamps: deque[float] = deque()
    active_stream_segment: dict[str, Any] | None = None
    assistant_segments: list[dict[str, Any]] = []
    receiver_error: str | None = None

    async with websockets.connect(ws_url, max_size=None) as websocket:
        await websocket.send(json.dumps({
            "event": "config",
            "data": {"lang": task, "exp": exp_name},
        }))

        async def receiver() -> None:
            nonlocal receiver_error, active_stream_segment

            def finish_stream_segment() -> None:
                nonlocal active_stream_segment
                segment = active_stream_segment
                active_stream_segment = None
                if not segment or not segment["chunks"]:
                    return
                audio = np.concatenate(segment["chunks"])
                assistant_segments.append({
                    "start": float(segment["start"]),
                    "duration": len(audio) / SAMPLE_RATE,
                    "audio": audio,
                })

            try:
                while True:
                    message = await websocket.recv()
                    if isinstance(message, bytes):
                        if active_stream_segment is not None:
                            if len(message) % 2:
                                raise ValueError("raw PCM16 chunk has an odd byte count")
                            active_stream_segment["chunks"].append(
                                np.frombuffer(message, dtype="<i2").astype(
                                    np.float32
                                ) / 32768.0
                            )
                            continue
                        if not pending_tts_timestamps:
                            events.append({"event": "audio_without_tts_done"})
                            continue
                        start_time = pending_tts_timestamps.popleft()
                        audio = _decode_audio_message(message)
                        assistant_segments.append({
                            "start": start_time,
                            "duration": len(audio) / SAMPLE_RATE,
                            "audio": audio,
                        })
                        continue

                    try:
                        obj = json.loads(message)
                    except json.JSONDecodeError:
                        events.append({"event": "unparsed_message", "message": str(message)})
                        continue
                    event_name = obj.get("event", "unknown")
                    events.append(obj)
                    timestamp = _event_timestamp(obj)
                    if event_name == "tts_segment_start":
                        finish_stream_segment()
                        active_stream_segment = {
                            "start": timestamp or 0.0,
                            "chunks": [],
                        }
                    elif event_name in {"tts_provider_ready", "tts_first_audio"}:
                        if active_stream_segment is not None and not active_stream_segment["chunks"]:
                            active_stream_segment["start"] = timestamp or active_stream_segment["start"]
                    elif event_name == "tts_segment_end":
                        finish_stream_segment()
                    elif event_name == "tts_done":
                        if timestamp is not None:
                            pending_tts_timestamps.append(timestamp)
            except websockets.exceptions.ConnectionClosed:
                finish_stream_segment()
                return
            except Exception as exc:  # preserve partial artifacts for diagnosis
                finish_stream_segment()
                receiver_error = repr(exc)

        receiver_task = asyncio.create_task(receiver())
        clock_start = time.perf_counter()
        for offset in range(0, len(input_audio), CHUNK_SAMPLES):
            chunk = input_audio[offset:offset + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            target = offset / SAMPLE_RATE
            elapsed = time.perf_counter() - clock_start
            if target > elapsed:
                await asyncio.sleep(target - elapsed)
            await websocket.send(chunk.tobytes())

        # Do not send ``end`` immediately: fd-badcat treats it as a client
        # disconnect and cancels the active generation.  Trailing silence lets
        # VAD close the final user segment while the response is produced.
        tail_samples = int(round(max(0.0, tail_seconds) * SAMPLE_RATE))
        tail_start = len(input_audio) / SAMPLE_RATE
        for offset in range(0, tail_samples, CHUNK_SAMPLES):
            chunk = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            target = tail_start + offset / SAMPLE_RATE
            elapsed = time.perf_counter() - clock_start
            if target > elapsed:
                await asyncio.sleep(target - elapsed)
            await websocket.send(chunk.tobytes())

        if close_grace_seconds > 0:
            await asyncio.sleep(close_grace_seconds)
        await websocket.send(json.dumps({"event": "end"}))
        await websocket.close()
        await receiver_task

    output_duration = input_duration
    for segment in assistant_segments:
        output_duration = max(output_duration, segment["start"] + segment["duration"])
    output = np.zeros(max(1, int(np.ceil(output_duration * SAMPLE_RATE))), dtype=np.float32)
    serializable_segments = []
    for segment in assistant_segments:
        start = max(0.0, float(segment["start"]))
        start_sample = int(round(start * SAMPLE_RATE))
        audio = segment["audio"]
        end_sample = min(len(output), start_sample + len(audio))
        if end_sample > start_sample:
            output[start_sample:end_sample] += audio[: end_sample - start_sample]
        serializable_segments.append({
            "start": start,
            "duration": float(segment["duration"]),
        })

    sf.write(str(output_path), np.clip(output, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
    timing = {
        "schema_version": "vi-fdb-v1-fd-badcat-1",
        "task": task,
        "id": case_id,
        "condition": condition,
        "input": str(audio_path),
        "input_duration": input_duration,
        "output_duration": len(output) / SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "events": events,
        "assistant_segments": serializable_segments,
        "pending_tts_segments": len(pending_tts_timestamps),
        "receiver_error": receiver_error,
    }
    timing_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "task": task,
        "id": case_id,
        "condition": condition,
        "status": "ok" if receiver_error is None else "partial",
        "output": str(output_path),
        "timing": str(timing_path),
        "assistant_segments": len(assistant_segments),
    }


async def _main(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    records = _manifest_records(manifest_path)
    if args.limit is not None:
        records = records[: args.limit]

    conditions = ["event", "clean"] if args.condition == "both" else [args.condition]
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for record in records:
        for condition in conditions:
            if condition == "clean" and not record.get("clean_input"):
                continue
            print(f"[{condition}] {record.get('task')}/{record.get('id')}", flush=True)
            result = await _run_case(
                ws_url=args.ws_url,
                dataset_root=dataset_root,
                run_root=run_root,
                record=record,
                condition=condition,
                exp_name=args.exp_name,
                tail_seconds=args.tail_seconds,
                close_grace_seconds=args.close_grace_seconds,
            )
            summary.append(result)

    summary_path = run_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(item["status"] == "ok" for item in summary)
    partial = sum(item["status"] == "partial" for item in summary)
    skipped = sum(item["status"] == "skipped" for item in summary)
    print(f"completed={ok} partial={partial} skipped={skipped} summary={summary_path}")
    return 0 if partial == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="pilot_160 or expansion_240 directory")
    parser.add_argument("--run-root", required=True, help="directory for benchmark artifacts")
    parser.add_argument("--ws-url", default=os.getenv("FDB_WS_URL", "ws://127.0.0.1:18000/realtime"))
    parser.add_argument("--condition", choices=("event", "clean", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test limit")
    parser.add_argument("--tail-seconds", type=float, default=3.0)
    parser.add_argument("--close-grace-seconds", type=float, default=2.0)
    parser.add_argument("--exp-name", default="vi_fdb")
    parser.add_argument("--jobs", type=int, default=1, help="reserved for future parallel sessions")
    args = parser.parse_args()
    if args.jobs != 1:
        parser.error("--jobs must be 1: one websocket session preserves benchmark timing")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(parse_args())))
