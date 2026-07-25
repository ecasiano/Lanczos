#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Lanczos ED macOS App Builder ==="
echo ""

echo "[1/4] Installing build dependencies..."
pip install pyinstaller Pillow --break-system-packages 2>/dev/null || \
pip install pyinstaller Pillow

echo ""
echo "[2/4] Generating app icon..."
python build_icon.py

echo ""
echo "[3/4] Building app bundle with PyInstaller..."
pyinstaller --noconfirm LanczosED.spec

echo ""
echo "[4/4] Done!"
echo ""
echo "App location: dist/Lanczos ED.app"
echo ""
echo "To install, drag 'Lanczos ED.app' from dist/ to /Applications."
echo "Or run directly:"
echo "  open \"dist/Lanczos ED.app\""
