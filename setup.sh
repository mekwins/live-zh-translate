#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "== live-zh-translate setup =="

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 13 ]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "No suitable Python found (need 3.10-3.13; qwen3-asr-mlx doesn't support 3.14 yet)."
  echo "Install one with: brew install python@3.12"
  exit 1
fi

echo "-- Using $PYTHON ($($PYTHON --version)) --"
echo "-- Creating virtual environment (.venv) --"
"$PYTHON" -m venv .venv
source .venv/bin/activate

echo "-- Upgrading pip --"
pip install --upgrade pip --quiet

echo "-- Installing dependencies (this downloads MLX, numpy, sounddevice, argostranslate) --"
pip install -r requirements.txt

echo "-- Downloading Chinese -> English translation model (Argos Translate, one-time, ~50-100MB) --"
python3 - <<'PY'
import argostranslate.package
import argostranslate.translate

argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()

pkg = next(
    (p for p in available if p.from_code == "zh" and p.to_code == "en"),
    None,
)
if pkg is None:
    raise SystemExit("Could not find a zh->en Argos Translate package in the index.")

installed = argostranslate.package.get_installed_packages()
already = any(p.from_code == "zh" and p.to_code == "en" for p in installed)
if already:
    print("zh->en model already installed, skipping download.")
else:
    path = pkg.download()
    argostranslate.package.install_from_path(path)
    print("zh->en model installed.")
PY

echo ""
echo "== Setup complete =="
echo ""
echo "Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. python live_translate.py --list-devices     # find your mic/BlackHole device index"
echo "  3. python live_translate.py                    # start live translation (default mic)"
echo ""
echo "First run will also download the Qwen3-ASR model (~a few hundred MB) from HuggingFace Hub"
echo "and may prompt macOS for microphone permission -- approve it."
