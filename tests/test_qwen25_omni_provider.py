from __future__ import annotations

import logging
import queue
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from model_providers.factory import create_mllm_provider
from model_providers.qwen25_omni import Qwen25OmniProvider


class _Inputs(dict):
    def to(self, value):
        return self


class _Tokenizer:
    pass


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.conversation = None
        self.processor_kwargs = None
        self.decoded_ids = None

    def apply_chat_template(self, conversation, **kwargs):
        self.conversation = conversation
        logging.warning(
            "System prompt modified, audio output may not work as expected. "
            "Audio output mode only works with the default prompt."
        )
        return "prompt"

    def __call__(self, **kwargs):
        self.processor_kwargs = kwargs
        return _Inputs(input_ids=np.array([[10, 11]]))

    def batch_decode(self, ids, **kwargs):
        self.decoded_ids = ids
        return [" switch "]


class _Model:
    device = "cuda:0"
    dtype = "float16"

    def __init__(self):
        self.calls = []
        self.stream_chunks = ["Xin ", "chào"]

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        streamer = kwargs.get("streamer")
        if streamer is not None:
            for chunk in self.stream_chunks:
                streamer.push(chunk)
            streamer.end()
        return np.array([[10, 11, 12]])


def _process_mm_info(conversation, *, use_audio_in_video):
    audio = conversation[-1]["content"][0]
    assert audio["type"] == "audio"
    assert audio["audio"].startswith("data:audio/wav;base64,")
    return [audio["audio"]], None, None


class _StoppingCriteria:
    pass


class _StoppingCriteriaList(list):
    pass


class _TextIteratorStreamer:
    _END = object()

    def __init__(self, tokenizer, **kwargs):
        self.items = queue.Queue()

    def push(self, text):
        self.items.put(text)

    def end(self):
        self.items.put(self._END)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.items.get(timeout=2)
        if value is self._END:
            raise StopIteration
        return value


class Qwen25OmniProviderTests(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.processor = _Processor()
        self.provider = Qwen25OmniProvider(
            self.model,
            self.processor,
            process_mm_info=_process_mm_info,
        )
        self.messages = [
            {"role": "system", "content": "Return continue or switch."},
            {
                "role": "user",
                "content": [{
                    "type": "audio",
                    "audio": np.zeros(1_600, dtype=np.float32),
                }],
            },
        ]

    def test_decision_converts_audio_and_decodes_only_new_tokens(self):
        with self.assertNoLogs(level="WARNING"):
            self.assertEqual(self.provider.decide(self.messages), "switch")
        self.assertEqual(self.processor.decoded_ids.tolist(), [[12]])
        call = self.model.calls[-1]
        self.assertEqual(call["max_new_tokens"], 4)
        self.assertFalse(call["do_sample"])
        self.assertFalse(call["use_audio_in_video"])
        self.assertNotIn("return_audio", call)

    def test_factory_builds_qwen_provider_with_explicit_processor(self):
        provider = create_mllm_provider(
            model=self.model,
            processor=self.processor,
            env={"MLLM_PROVIDER": "qwen25_omni_local"},
        )
        self.assertIsInstance(provider, Qwen25OmniProvider)
        self.assertIs(provider.processor, self.processor)
        self.assertFalse(provider.capabilities.native_prefill)
        self.assertFalse(provider.capabilities.native_duplex)

    def test_stream_generate_yields_text_and_supports_close(self):
        fake_transformers = ModuleType("transformers")
        fake_transformers.StoppingCriteria = _StoppingCriteria
        fake_transformers.StoppingCriteriaList = _StoppingCriteriaList
        fake_transformers.TextIteratorStreamer = _TextIteratorStreamer
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            stream = self.provider.stream_generate(self.messages)
            self.assertEqual("".join(stream), "Xin chào")
        call = self.model.calls[-1]
        self.assertEqual(call["max_new_tokens"], 512)
        self.assertIn("stopping_criteria", call)
        self.assertNotIn("return_audio", call)


    def test_stream_stops_before_hallucinated_human_turn(self):
        self.model.stream_chunks = [
            "Chào bạn! ",
            "Tôi có thể giúp gì?\nHu",
            "man: Tôi muốn biết cách nấu cơm.\n\n\n",
        ]
        fake_transformers = ModuleType("transformers")
        fake_transformers.StoppingCriteria = _StoppingCriteria
        fake_transformers.StoppingCriteriaList = _StoppingCriteriaList
        fake_transformers.TextIteratorStreamer = _TextIteratorStreamer
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            stream = self.provider.stream_generate(self.messages)
            self.assertEqual(
                "".join(stream),
                "Chào bạn! Tôi có thể giúp gì?",
            )
        self.assertTrue(stream.cancelled.is_set())

    def test_stream_discards_trailing_blank_lines(self):
        self.model.stream_chunks = ["Xin chào!", "\n\n\n"]
        fake_transformers = ModuleType("transformers")
        fake_transformers.StoppingCriteria = _StoppingCriteria
        fake_transformers.StoppingCriteriaList = _StoppingCriteriaList
        fake_transformers.TextIteratorStreamer = _TextIteratorStreamer
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            self.assertEqual(
                "".join(self.provider.stream_generate(self.messages)),
                "Xin chào!",
            )


if __name__ == "__main__":
    unittest.main()
