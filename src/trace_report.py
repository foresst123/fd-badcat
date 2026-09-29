"""Render FD-BADCAT JSONL traces as a compact debug timeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tracing import latest_trace


_NOISY_EVENTS = {
    "assistant_delta",
    "generation_cancel_finished",
}
_ERROR_MARKERS = ("error", "failed", "fallback")


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trace_path = Path(path)
    with trace_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL lỗi tại {trace_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Trace row {line_number} phải là object JSON"
                )
            rows.append(row)
    return rows


def _value(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = data.get(name)
        if value is not None and value != "":
            return value
    return None


def _short(value: Any, limit: int = 96) -> str:
    if isinstance(value, dict) and value.get("redacted"):
        return f"<redacted:{value.get('chars', '?')} chars>"
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def diagnose_trace(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if not rows:
        return ["Trace rỗng."]
    if rows[-1].get("event") != "trace_closed":
        warnings.append(
            "Trace chưa có trace_closed: phiên có thể bị kill/restart đột ngột."
        )

    response_starts: dict[Any, dict[str, Any]] = {}
    response_audio: set[Any] = set()
    cancelled_generations: set[Any] = set()
    for row in rows:
        event = str(row.get("event", ""))
        data = row.get("data") or {}
        if event == "response_started":
            response_starts[data.get("generation")] = row
        elif event == "tts_first_audio":
            response_audio.add(data.get("generation"))
        elif event == "generation_cancel_finished":
            generation = data.get("generation")
            if generation is not None and data.get("actual_cancel", True):
                cancelled_generations.add(generation)
        if row.get("level") == "error" or any(
            marker in event for marker in _ERROR_MARKERS
        ):
            message = _value(data, "message", "reason", "fallback")
            warnings.append(
                f"{event}: {_short(message) if message is not None else 'xem data'}"
            )
        elif event == "paper_unit_replayed":
            warnings.append(
                "Đã chặn một SPEAK Unit thuộc generation cũ và replay audio "
                f"sang Unit {data.get('replay_unit')} ở "
                f"state={data.get('current_state')}."
            )
        elif event == "vad_segment_replayed":
            warnings.append(
                "Segment SPEAK thuộc generation cũ được đánh giá lại thành "
                f"Segment {data.get('replay_segment')} ở "
                f"state={data.get('current_state')}."
            )
        elif event == "vad_segment_replaced":
            warnings.append(
                f"Decision bận: bỏ Segment {data.get('dropped_segment')} và "
                f"giữ Segment mới nhất {data.get('latest_segment')}."
            )
        elif event == "vad_segment_tick":
            if data.get("current_asr_used") is not False:
                warnings.append(
                    f"Segment {data.get('segment')} không xác nhận "
                    "current_asr_used=false; kiểm tra rò transcript N vào "
                    "decision N."
                )
            if int(data.get(
                "decision_queue_depth", data.get("queue_depth", 0)
            ) or 0) > 1:
                warnings.append(
                    f"VAD Segment decision queue vượt 1: "
                    f"{data.get('decision_queue_depth', data.get('queue_depth'))}."
                )

    for generation in response_starts:
        if (
            generation not in response_audio
            and generation not in cancelled_generations
        ):
            warnings.append(
                f"Generation {generation} đã response_started nhưng không có "
                "tts_first_audio."
            )

    decisions = [
        row for row in rows if row.get("event") == "duplex_decision"
    ]
    for index, row in enumerate(decisions):
        data = row.get("data") or {}
        if (
            data.get("state") == "SPEAK"
            and data.get("flag") == "ks"
            and data.get("vad_end") is True
        ):
            later_flags = {
                (candidate.get("data") or {}).get("flag")
                for candidate in decisions[index + 1:]
            }
            if "s2l" not in later_flags:
                warnings.append(
                    "Có SPEAK+VAD-end bị giữ KS nhưng không thấy S2L sau đó. "
                    "Nếu audio là lời chen thật (không phải backchannel), kiểm "
                    "tra paper_decision_retry/ASR và prompt barge-in."
                )
                break
    return list(dict.fromkeys(warnings))


def _latency_metrics(rows: list[dict[str, Any]]) -> list[str]:
    metrics: list[str] = []
    l2s_ms = None
    s2l_ms = None
    latest_vad_end: dict[str, float] = {}
    response_ms: dict[Any, float] = {}
    for row in rows:
        event = row.get("event")
        data = row.get("data") or {}
        elapsed = float(row.get("elapsed_ms", 0.0))
        if event == "vad_done":
            latest_vad_end[str(data.get("state", "unknown"))] = elapsed
        elif event == "duplex_decision" and data.get("flag") == "l2s":
            l2s_ms = elapsed
            vad_ms = latest_vad_end.get("LISTEN")
            if vad_ms is not None:
                metrics.append(
                    f"turn {data.get('turn')} VAD-end→L2S="
                    f"{(elapsed - vad_ms) / 1000:.3f}s"
                )
        elif event == "duplex_decision" and data.get("flag") == "s2l":
            s2l_ms = elapsed
            vad_ms = latest_vad_end.get("SPEAK")
            if vad_ms is not None:
                metrics.append(
                    f"turn {data.get('turn')} VAD-end→S2L="
                    f"{(elapsed - vad_ms) / 1000:.3f}s"
                )
        elif event == "stop_audio" and s2l_ms is not None:
            metrics.append(
                f"generation {data.get('generation')} S2L→stop_audio="
                f"{(elapsed - s2l_ms) / 1000:.3f}s"
            )
        elif event == "response_started":
            response_ms[data.get("generation")] = elapsed
        elif event == "tts_first_audio":
            generation = data.get("generation")
            start = response_ms.get(generation)
            if start is not None:
                metrics.append(
                    f"generation {generation} response→audio="
                    f"{(elapsed - start) / 1000:.3f}s"
                )
            if l2s_ms is not None:
                metrics.append(
                    f"generation {generation} L2S→audio="
                    f"{(elapsed - l2s_ms) / 1000:.3f}s"
                )
    return metrics


def render_trace(path: str | Path, *, include_noisy: bool = False) -> str:
    trace_path = Path(path)
    rows = load_trace(trace_path)
    if not rows:
        return f"Trace: {trace_path}\n(empty)"
    trace_id = rows[0].get("trace_id", "unknown")
    counts = Counter(str(row.get("event", "unknown")) for row in rows)
    duration_ms = float(rows[-1].get("elapsed_ms", 0.0))
    lines = [
        f"Trace: {trace_path}",
        f"trace_id={trace_id} duration={duration_ms / 1000:.3f}s "
        f"events={len(rows)}",
        "",
        "Timeline:",
    ]
    previous_ms = 0.0
    for row in rows:
        event = str(row.get("event", "unknown"))
        if not include_noisy and event in _NOISY_EVENTS:
            continue
        elapsed_ms = float(row.get("elapsed_ms", 0.0))
        delta_ms = elapsed_ms - previous_ms
        previous_ms = elapsed_ms
        data = row.get("data") or {}
        fields: list[str] = []
        for key in (
            "state", "state_before", "state_after", "turn", "captured_turn",
            "captured_state", "decision_state", "current_state", "unit",
            "replay_unit", "segment", "replay_segment", "dropped_segment",
            "latest_segment", "generation", "captured_generation",
            "current_generation", "decision", "flag", "reason", "stage",
            "queue_depth", "queue_wait", "busy_retries", "request_id",
            "infer_time", "prefill_time", "ttft", "ttfa", "duration_ms",
            "audio_seconds", "audio_duration", "vad_end",
            "current_asr_used", "available_from_segment",
            "raw_decision", "parsed_decision", "parse_valid",
            "decision_source", "fallback_reason", "fallback_policy",
            "decision_queue_depth", "asr_frame_queue_depth",
            "tts_text_queue_depth", "actual_cancel",
        ):
            if key in data and data[key] is not None:
                fields.append(f"{key}={_short(data[key], 48)}")
        detail = _value(data, "message", "transcript", "content")
        if detail is not None:
            fields.append(f"detail={_short(detail)}")
        suffix = " " + " ".join(fields) if fields else ""
        lines.append(
            f"{elapsed_ms / 1000:8.3f}s +{delta_ms:7.1f}ms "
            f"{row.get('level', 'info'):5s} {event}{suffix}"
        )

    lines.extend(["", "Metrics:"])
    metrics = _latency_metrics(rows)
    lines.extend(f"- {metric}" for metric in metrics)
    if not metrics:
        lines.append("- Chưa đủ event để tính response/TTS latency.")
    lines.extend(["", "Diagnostics:"])
    warnings = diagnose_trace(rows)
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- Không phát hiện bất thường theo rule hiện tại.")
    lines.extend(["", "Top events:"])
    lines.append(
        "- " + ", ".join(
            f"{event}={count}" for event, count in counts.most_common(12)
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hiển thị timeline trace FD-BADCAT"
    )
    parser.add_argument("path", nargs="?", help="File trace JSONL")
    parser.add_argument(
        "--dir", default="traces", help="Thư mục để chọn trace mới nhất"
    )
    parser.add_argument(
        "--all", action="store_true", help="Hiện cả assistant_delta/noisy events"
    )
    args = parser.parse_args()
    path = Path(args.path) if args.path else latest_trace(args.dir)
    if path is None:
        raise SystemExit(f"Không tìm thấy trace trong {args.dir}")
    print(render_trace(path, include_noisy=args.all))


if __name__ == "__main__":
    main()
