from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from trace_report import diagnose_trace, load_trace, render_trace
from tracing import TRACE_SCHEMA, TraceRecorder, latest_trace


class TraceRecorderTests(unittest.TestCase):
    def test_jsonl_trace_correlates_events_and_omits_audio_and_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TraceRecorder(
                enabled=True,
                directory=directory,
                include_text=False,
                max_text_chars=128,
            )
            recorder.record("paper_unit_tick", {
                "turn": 2,
                "content": "nội dung người dùng",
                "asr_context": "ngữ cảnh nhận dạng",
                "audio": "data:audio/wav;base64,AAAA",
                "MLLM_API_KEY": "do-not-write-this",
            })
            recorder.note_audio(3200)
            recorder.close({"close_reason": "test"})

            self.assertIsNotNone(recorder.path)
            rows = load_trace(recorder.path)
            self.assertEqual(rows[0]["schema"], TRACE_SCHEMA)
            self.assertTrue(all(
                row["trace_id"] == recorder.trace_id for row in rows
            ))
            self.assertEqual(
                [row["seq"] for row in rows],
                list(range(1, len(rows) + 1)),
            )
            event = next(
                row for row in rows if row["event"] == "paper_unit_tick"
            )
            self.assertEqual(
                event["data"]["content"],
                {"redacted": True, "chars": len("nội dung người dùng")},
            )
            self.assertEqual(
                event["data"]["asr_context"]["redacted"], True
            )
            self.assertEqual(
                event["data"]["audio"], "<AUDIO_DATA_OMITTED>"
            )
            self.assertEqual(event["data"]["MLLM_API_KEY"], "<REDACTED>")
            closed = rows[-1]
            self.assertEqual(closed["event"], "trace_closed")
            self.assertEqual(closed["data"]["outbound_audio_chunks"], 1)
            self.assertEqual(closed["data"]["outbound_audio_bytes"], 3200)
            self.assertEqual(latest_trace(directory), recorder.path)

    def test_unwritable_trace_target_fails_open(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-a-directory"
            target.write_text("occupied", encoding="utf-8")
            recorder = TraceRecorder(
                enabled=True,
                directory=target,
                include_text=True,
                max_text_chars=100,
            )
            self.assertFalse(recorder.enabled)
            self.assertIsNone(recorder.path)
            self.assertIn("FileExistsError", recorder.error)
            recorder.record("must_not_raise")
            recorder.close()

    def test_disabled_recorder_does_not_create_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TraceRecorder(
                enabled=False,
                directory=directory,
                include_text=True,
                max_text_chars=100,
            )
            recorder.record("ignored", {"content": "x"})
            recorder.close()
            self.assertIsNone(recorder.path)
            self.assertEqual(list(Path(directory).iterdir()), [])


class TraceReportTests(unittest.TestCase):
    def test_report_computes_tts_latency_and_detects_missing_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TraceRecorder(
                enabled=True,
                directory=directory,
                include_text=True,
                max_text_chars=256,
            )
            recorder.record("duplex_decision", {
                "state": "LISTEN", "flag": "l2s", "turn": 0,
            })
            recorder.record("response_started", {
                "generation": 1, "turn": 0,
            })
            recorder.record("tts_first_audio", {
                "generation": 1, "ttfa": 0.42,
            })
            recorder.close({"close_reason": "test"})
            report = render_trace(recorder.path)
            self.assertIn("L2S→audio", report)
            self.assertIn("response→audio", report)
            self.assertEqual(diagnose_trace(load_trace(recorder.path)), [])

            second = TraceRecorder(
                enabled=True,
                directory=directory,
                include_text=True,
                max_text_chars=256,
            )
            second.record("response_started", {"generation": 9})
            second.close()
            warnings = diagnose_trace(load_trace(second.path))
            self.assertTrue(any(
                "không có tts_first_audio" in warning for warning in warnings
            ))

    def test_cancelled_generation_without_audio_is_not_reported_as_tts_failure(self):
        rows = [{
            "event": "response_started",
            "data": {"generation": 3},
        }, {
            "event": "generation_cancel_finished",
            "data": {"generation": 3, "actual_cancel": True},
        }, {
            "event": "trace_closed",
            "data": {},
        }]

        warnings = diagnose_trace(rows)

        self.assertFalse(any(
            "Generation 3" in warning and "tts_first_audio" in warning
            for warning in warnings
        ))

    def test_report_surfaces_replayed_stale_speak_unit(self):
        rows = [{
            "event": "paper_unit_replayed",
            "data": {
                "unit": 4,
                "replay_unit": 5,
                "current_state": "LISTEN",
            },
        }, {
            "event": "trace_closed",
            "data": {},
        }]

        warnings = diagnose_trace(rows)

        self.assertTrue(any(
            "generation cũ" in warning and "Unit 5" in warning
            for warning in warnings
        ))


    def test_report_validates_segment_context_and_latest_wins_queue(self):
        rows = [{
            "event": "vad_segment_tick",
            "data": {
                "segment": 3,
                "current_asr_used": False,
                "queue_depth": 1,
            },
        }, {
            "event": "vad_segment_replaced",
            "data": {"dropped_segment": 2, "latest_segment": 3},
        }, {
            "event": "trace_closed",
            "data": {},
        }]

        warnings = diagnose_trace(rows)

        self.assertFalse(any("rò transcript" in item for item in warnings))
        self.assertTrue(any(
            "bỏ Segment 2" in item and "Segment mới nhất 3" in item
            for item in warnings
        ))

    def test_report_warns_if_current_asr_leaks_into_same_decision(self):
        rows = [{
            "event": "vad_segment_tick",
            "data": {
                "segment": 7,
                "current_asr_used": True,
                "queue_depth": 2,
            },
        }, {
            "event": "trace_closed",
            "data": {},
        }]

        warnings = diagnose_trace(rows)

        self.assertTrue(any("rò transcript N" in item for item in warnings))
        self.assertTrue(any("queue vượt 1" in item for item in warnings))


if __name__ == "__main__":
    unittest.main()
