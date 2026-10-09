# live-zh-translate

Live Chinese → English speech translation, running **entirely locally** on Apple Silicon.

- **ASR**: [Qwen3-ASR](https://github.com/moona3k/mlx-qwen3-asr) via MLX — runs on the Metal GPU (fast on M1/M2/M3/M4, not CPU-bound like plain `openai-whisper`).
- **Translation**: [Argos Translate](https://github.com/argosopentech/argos-translate) — fully offline neural MT, no API keys, no internet needed after first model download.
- **Audio capture**: microphone input via `sounddevice`, auto-chunked on silence (simple VAD).

Speak/play Chinese audio near your Mac's microphone → see rolling Chinese transcript + English translation printed live in the terminal.

## Requirements

- macOS, Apple Silicon (M1/M2/M3/M4) — needs Metal GPU for MLX to be fast. Will still run on Intel Mac / other OS but ASR will fall back to CPU and be much slower.
- Python 3.10+
- Microphone access permission for Terminal (see Setup below)

## Setup

```bash
git clone https://github.com/mekwins/live-zh-translate.git
cd live-zh-translate
./setup.sh
```

`setup.sh` creates a venv, installs dependencies, and downloads the Chinese→English Argos Translate model (~50-100MB, one-time).

### Grant microphone access

The first time you run it, macOS will prompt for microphone permission for your terminal app (Terminal.app / iTerm / etc). Approve it.

If it silently fails with no prompt: go to **System Settings → Privacy & Security → Microphone** and manually enable your terminal app, then restart the terminal.

## Usage

```bash
source .venv/bin/activate
python live_translate.py
```

Default: listens on your system's default microphone input, auto-detects speech segments (pause-based chunking), transcribes each segment with Qwen3-ASR, translates to English, and prints both side by side.

Press `Ctrl+C` to stop.

### Options

```bash
python live_translate.py --list-devices          # show available audio input devices
python live_translate.py --device 2              # use a specific input device index
python live_translate.py --silence-ms 700         # pause length (ms) that ends a speech segment (default 700)
python live_translate.py --min-chunk-s 1.0         # ignore segments shorter than this (default 1.0s)
python live_translate.py --model mlx-community/Qwen3-ASR-1.7B-bf16   # ASR model (default)
python live_translate.py --no-translate            # Chinese transcript only, skip translation
```

## Translating audio PLAYING ON your Mac (not just the mic)

By default this captures your physical microphone — fine for in-person conversations or a phone/speaker next to your Mac. To translate audio from a video call, a video file, or streaming audio playing *on* your Mac (i.e. system audio, not mic):

1. Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) (free virtual audio driver): `brew install blackhole-2ch`
2. In **System Settings → Sound → Output**, create a Multi-Output Device combining BlackHole + your speakers (so you can still hear it), or just route the app's output to BlackHole.
3. Run `python live_translate.py --list-devices` and pick the BlackHole input device with `--device N`.

## How it works

1. `sounddevice` streams mic audio in small blocks at 16kHz mono.
2. A simple RMS-energy voice activity detector buffers audio while speech is detected, and flushes a "segment" once a configurable silence gap is seen (default 700ms) — this is what gives you live-feeling, sentence-by-sentence output instead of waiting for you to stop talking entirely.
3. Each segment goes to Qwen3-ASR (`language='zh'`) running on MLX/Metal for transcription.
4. The Chinese text is translated to English locally via Argos Translate (no network call).
5. Both are printed with a timestamp.

## Known limitations

- Segment-based (not word-by-word streaming) — there's a small delay (silence-gap length + inference time) before each line appears, not true simultaneous interpretation.
- Argos Translate's zh→en quality is decent for plain speech but not as good as a large LLM. If you have a local or gateway LLM endpoint available, you can swap `translate.py` to call that instead (see comments in the file) for better quality.
- Background noise can trigger false speech segments; adjust `--silence-ms` and mic placement if needed.
