#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# MLX Vision Server — Gemma 4 E4B (4B dense) on Apple M4
# ──────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-$DIR/models/gemma4-e4b-4bit}"

export MODEL_PATH="$MODEL"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8001}"

echo "╔══════════════════════════════════════════════╗"
echo "║  MLX Vision Server — Gemma 4 E4B (4-bit)    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Model:  $MODEL_PATH"
echo "  URL:    http://${HOST}:${PORT}"
echo "  API:    http://${HOST}:${PORT}/v1/chat/completions"
echo ""

exec python3 "$DIR/server.py" \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH"
