#!/usr/bin/env python3
"""live_translate.py

Live Chinese -> English speech translation, fully local.

Pipeline:
  mic audio (sounddevice) -> simple energy-based VAD chunking
    -> Qwen3-ASR (MLX, Metal GPU) for Chinese transcription
    -> Argos Translate (offline NMT) for English translation
    -> printed live to the terminal

See README.md for setup and usage.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
BLOCK_MS = 30  # audio callback block size in ms
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Chinese -> English speech translation (local, Metal GPU).")
    p.add_argument("--list-devices", action="store_true", help="List audio input devices and exit.")
    p.add_argument("--device", type=int, default=None, help="Input device index (see --list-devices).")
    p.add_argument("--silence-ms", type=int, default=700, help="Silence gap (ms) that ends a speech segment.")
    p.add_argument("--min-chunk-s", type=float, default=1.0, help="Minimum segment length (s) to bother transcribing.")
    p.add_argument("--max-chunk-s", type=float, default=20.0, help="Force-flush a segment after this many seconds of continuous speech.")
    p.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-ASR-1.7B-bf16",
        help="Qwen3-ASR model id (HuggingFace Hub) or local path.",
    )
    p.add_argument("--energy-threshold", type=float, default=0.012, help="RMS energy threshold for 'speech' (tune for your mic/environment).")
    p.add_argument("--no-translate", action="store_true", help="Print Chinese transcript only, skip English translation.")
    return p.parse_args()


def list_devices() -> None:
    import sounddevice as sd

    print("Available audio input devices:\n")
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            default_marker = " (default)" if idx == sd.default.device[0] else ""
            print(f"  [{idx}] {dev['name']}  — {dev['max_input_channels']}ch, {dev['default_samplerate']:.0f}Hz{default_marker}")
    print("\nTip: to translate system/call audio instead of your mic, install BlackHole")
    print("(brew install blackhole-2ch) and pick its device index here. See README.md.")


# ---------------------------------------------------------------------------
# Simple energy-based VAD + segmenter
# ---------------------------------------------------------------------------

@dataclass
class Segmenter:
    """Buffers incoming audio blocks and yields a full utterance once silence follows speech."""

    sample_rate: int
    silence_blocks: int
    min_chunk_samples: int
    max_chunk_samples: int
    energy_threshold: float

    def __post_init__(self) -> None:
        self._buffer: list[np.ndarray] = []
        self._silence_run = 0
        self._in_speech = False

    def push(self, block: np.ndarray) -> np.ndarray | None:
        """Feed one audio block. Returns a completed segment (np.ndarray) or None."""
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        is_speech_block = rms >= self.energy_threshold

        if is_speech_block:
            self._buffer.append(block)
            self._silence_run = 0
            self._in_speech = True
        elif self._in_speech:
            # still append a little trailing silence so words aren't clipped
            self._buffer.append(block)
            self._silence_run += 1

        total_samples = sum(len(b) for b in self._buffer)

        should_flush = False
        if self._in_speech and self._silence_run >= self.silence_blocks:
            should_flush = True
        elif total_samples >= self.max_chunk_samples:
            should_flush = True

        if should_flush and self._buffer:
            segment = np.concatenate(self._buffer)
            self._buffer = []
            self._silence_run = 0
            self._in_speech = False
            if len(segment) >= self.min_chunk_samples:
                return segment
            return None

        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    import sounddevice as sd

    print("Loading Qwen3-ASR model (first run downloads it from HuggingFace Hub, ~GB)...")
    from qwen3_asr_mlx import Qwen3ASR

    asr = Qwen3ASR.from_pretrained(args.model)
    print("Model loaded. Warming up...")
    asr.warm_up()
    print("Ready.\n")

    translator = None
    if not args.no_translate:
        try:
            import argostranslate.translate as at_translate

            installed = at_translate.get_installed_languages()
            zh = next((l for l in installed if l.code == "zh"), None)
            en = next((l for l in installed if l.code == "en"), None)
            if zh is None or en is None:
                print(
                    "WARNING: Argos Translate zh->en model not installed. "
                    "Run setup.sh first, or use --no-translate. Continuing without translation.",
                    file=sys.stderr,
                )
            else:
                translator = zh.get_translation(en)
        except ImportError:
            print(
                "WARNING: argostranslate not installed. Run setup.sh first, or use --no-translate.",
                file=sys.stderr,
            )

    silence_blocks = max(1, int(args.silence_ms / BLOCK_MS))
    seg = Segmenter(
        sample_rate=SAMPLE_RATE,
        silence_blocks=silence_blocks,
        min_chunk_samples=int(args.min_chunk_s * SAMPLE_RATE),
        max_chunk_samples=int(args.max_chunk_s * SAMPLE_RATE),
        energy_threshold=args.energy_threshold,
    )

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def audio_callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    stream_kwargs = dict(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SAMPLES,
        callback=audio_callback,
    )
    if args.device is not None:
        stream_kwargs["device"] = args.device

    print("Listening... speak Chinese near the mic (Ctrl+C to stop).\n")

    stop_event = threading.Event()

    def worker() -> None:
        while not stop_event.is_set():
            try:
                block = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            segment = seg.push(block)
            if segment is None:
                continue

            ts = time.strftime("%H:%M:%S")
            try:
                result = asr.transcribe(segment, language="zh")
            except Exception as exc:  # noqa: BLE001
                print(f"[{ts}] ASR error: {exc}", file=sys.stderr)
                continue

            zh_text = result.text.strip()
            if not zh_text:
                continue

            print(f"[{ts}] 中文: {zh_text}")

            if translator is not None:
                try:
                    en_text = translator.translate(zh_text)
                    print(f"[{ts}] EN:   {en_text}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[{ts}] translation error: {exc}", file=sys.stderr)

            print()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    try:
        with sd.InputStream(**stream_kwargs):
            while True:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        worker_thread.join(timeout=2)


if __name__ == "__main__":
    main()
