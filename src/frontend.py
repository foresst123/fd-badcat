# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from datetime import datetime
import asyncio
import json
import io
import time

import numpy as np
import soundfile as sf
import websockets
from tqdm import tqdm
import yaml
WS_URL = None
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 256  # 16 ms per frame
LOG_MAX_SIZE = 5 * 1024 * 1024  # 5MB

class RotatingLogger:
    def __init__(self, base_path: Path, max_size: int = LOG_MAX_SIZE):
        self.base_path = base_path
        self.max_size = max_size
        self.file_index = 1
        self.current_file = self._get_log_path()
        self.current_file.write_text("")  

    def _get_log_path(self) -> Path:
        return self.base_path.with_name(f"{self.base_path.name}_{self.file_index}.txt")

    def _rotate_if_needed(self):
        if self.current_file.exists() and self.current_file.stat().st_size >= self.max_size:
            self.file_index += 1
            self.current_file = self._get_log_path()
            self.current_file.write_text("")

    def log(self, msg: str):
        ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{ts} {msg}"
        self._rotate_if_needed()
        with self.current_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)

class SpeakerSimulator:
    def __init__(self, total_duration: float, sr: int = SAMPLE_RATE, output_path: Path | None = None, *, log=print):
        self.log = log
        self.sr = sr
        self.chunk_time = CHUNK_SAMPLES / sr
        self.total_samples = int(total_duration * sr)
        self.output_path = output_path
        self.audio_buffer = np.zeros(self.total_samples, dtype=np.float32)
        self.start_wall = None
        self.playback_cursor = 0
        self.interrupted = False
        self.log(f"default buffer, with {total_duration:.2f}s（{self.total_samples} samples）")

    def _ensure_capacity(self, required_samples: int):
        if required_samples <= len(self.audio_buffer):
            return
        self.audio_buffer = np.pad(
            self.audio_buffer,
            (0, required_samples - len(self.audio_buffer)),
        )

    def write_pcm_chunk(self, pcm_bytes: bytes, receive_time: float):
        if len(pcm_bytes) % 2:
            raise ValueError("PCM16 chunk có số byte lẻ")
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
        samples /= 32768.0
        start_sample = max(int(receive_time * self.sr), self.playback_cursor)
        end_sample = start_sample + len(samples)
        self._ensure_capacity(end_sample)
        self.audio_buffer[start_sample:end_sample] = samples
        self.playback_cursor = end_sample

    def reset_for_new_audio(self, wav_bytes: bytes, start_time: float):
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        assert sr == self.sr, f"sample wrong: {sr} != {self.sr}"

        start_sample = int(start_time * self.sr)
        self._ensure_capacity(start_sample + len(data))
        write_samples = len(data)

        if write_samples > 0:
            self.audio_buffer[start_sample:start_sample + write_samples] = data[:write_samples]
            self.log(f"at {start_time:.2f}s insert TTS（{write_samples / self.sr:.2f}s）")
        else:
            self.log(f"discord start={start_time:.3f}s")

    def handle_interrupt(self):
        self.interrupted = True
        self.log("interrupt happen")

    def save_output(self):
        if self.output_path is None:
            self.log("no path")
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.output_path, self.audio_buffer, self.sr)
        self.log(f"save output {self.output_path}, with {len(self.audio_buffer) / self.sr:.2f}s")


async def mic_sender(ws, wav_path: Path, *, log=print):
    data, sr = sf.read(str(wav_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"sample is {sr} != {SAMPLE_RATE}")

    frame_time = CHUNK_SAMPLES / sr
    total_frames = int(np.ceil(len(data) / CHUNK_SAMPLES))
    log(f"start push frame {wav_path.name}, with {total_frames} frames, {len(data)/sr:.2f}s")
    t0 = time.perf_counter()

    for i in range(0, len(data), CHUNK_SAMPLES):
        chunk = data[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        await ws.send(chunk.tobytes())
        await asyncio.sleep(frame_time)

    log(f"client ok and audio length: {time.perf_counter() - t0:.2f}s")


async def simulate_full_frontend(wav_path: Path, output_path: Path, *, log=print, lang, exp):
    data, sr = sf.read(str(wav_path), dtype="float32")
    total_duration = len(data) / sr

    async with websockets.connect(WS_URL, max_size=None) as ws:
        log(f"connect server: {WS_URL}")
        await ws.send(json.dumps({
            "event": "config",
            "data": {"lang": lang,
                     "exp": exp }
        }))
        speaker = SpeakerSimulator(total_duration, sr=sr, output_path=output_path, log=log)
        speaker.start_wall = time.perf_counter()
        last_tts_timestamp = None
        pcm_stream_active = False
        response_done = asyncio.Event()

        async def receiver():
            nonlocal last_tts_timestamp, pcm_stream_active
            while True:
                try:
                    msg = await ws.recv()
                except websockets.exceptions.ConnectionClosed:
                    log("WebSocket disconnect")
                    break

                if isinstance(msg, bytes):
                    if pcm_stream_active:
                        elapsed = time.perf_counter() - speaker.start_wall
                        speaker.write_pcm_chunk(msg, elapsed)
                        continue
                    if last_tts_timestamp is None:
                        log("receive tts bug no timestamp, wright file wrong")
                        continue
                    speaker.reset_for_new_audio(msg, last_tts_timestamp)
                    last_tts_timestamp = None
                    continue

                try:
                    obj = json.loads(msg)
                except Exception:
                    log(f"text message: {msg}")
                    continue

                event = obj.get("event")
                if event == "tts_done":
                    last_tts_timestamp = obj.get("data", {}).get("timestamp")
                    log(f"tts_done: {obj}")
                    log(f"tts_done, timestamp={last_tts_timestamp}s")
                elif event == "tts_stream_start":
                    pcm_stream_active = True
                    log(f"tts_stream_start: {obj}")
                elif event == "tts_stream_end":
                    pcm_stream_active = False
                    response_done.set()
                    log(f"tts_stream_end: {obj}")
                elif event == "stop_audio":
                    pcm_stream_active = False
                    speaker.handle_interrupt()
                elif event == "generation_error":
                    response_done.set()
                    log(f"generation_error: {obj}")
                else:
                    log(f"info: {obj}")

                await asyncio.sleep(0)

        send_task = asyncio.create_task(mic_sender(ws, wav_path, log=log))
        recv_task = asyncio.create_task(receiver())

        await send_task
        await asyncio.wait_for(response_done.wait(), timeout=180)
        await ws.send(json.dumps({"event": "end"}))
        await ws.close()
        await recv_task

        speaker.save_output()
        log(f"save TTS output: {output_path}")


async def main():
    global WS_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml", help="YAML file config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    client_cfg = cfg.get("client", {})

    modes = client_cfg.get("lang", {})
    exp = client_cfg.get("exp", {})
    port = client_cfg.get("port", {})

    WS_URL = f"ws://127.0.0.1:{port}/realtime"

    if modes == "zhen":
        langs = ["zh", "en"]
    elif modes == "tc":
        langs = ["test", "clean"]
    else:
        langs = [modes]

    for mode in langs:
        #base_dir = Path(f"exp/{exp}/dev_{cur_lang}")
        base_dir = Path(f"exp/{exp}/{mode}")
        base_dir.mkdir(parents=True, exist_ok=True)

        output_dir = Path(f"exp/{exp}/HD-Track2/{mode}")
        output_dir.mkdir(parents=True, exist_ok=True)

        log_base = Path(f"exp/{exp}/{exp}_lg_{mode}")
        log_base.parent.mkdir(parents=True, exist_ok=True)

        logger = RotatingLogger(log_base)
        log = logger.log
        log(f"========== client start ({mode}) ==========")

        read_all_subdirs = True
        max_files = None
        # target_subdir = ["Follow-up Questions", "Negation or Dissatisfaction", "Repetition Requests", "Silence or Termination", "Topic Switching"]
        # target_subdir = ["Negation or Dissatisfaction"]
        # target_subdir = ["User Real-time Backchannels"]

        wav_files = sorted([
            f for f in base_dir.glob("*.wav")
            if not f.name.endswith("_output.wav")
        ])
        log(f"load {len(wav_files)} wavs file ({mode})")
        for wav_path in tqdm(wav_files, desc=f"{mode} set", ncols=80):
            output_path = output_dir / f"{wav_path.stem}_output.wav"

            log(f"processing file : {wav_path.name}")
            if output_path.exists():
                log(f"ignore already done file: {wav_path.name}")
                continue
            try:
                await simulate_full_frontend(wav_path, output_path, log=log, lang=mode, exp=exp)
                log("----------------------------------------------------------------------")
            except Exception as e:
                log(f"wrong with {wav_path.name} and {e}")
        log(f"========== all files done ({mode}) ==========")

if __name__ == "__main__":
    asyncio.run(main())