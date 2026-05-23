#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  M1 Max — Qwen3.5 4B · Vision · APC Cache · Creative Writing
#  Single command:  ./setup_m1max.sh
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
info() { echo -e "${CYAN}[*]${NC} $*"; }

# ── Config (override via env) ────────────────────────────────────────
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen35-4b-server}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/mlx-vlm/apc}"
CACHE_GB="${CACHE_GB:-25}"
APC_BLOCKS="${APC_BLOCKS:-4096}"
CTX="${CTX:-32768}"

TARGET="mlx-community/Qwen3.5-4B-4bit"
TARGET_LOCAL="$MODEL_DIR/Qwen3.5-4B-4bit"

# ── Banner ───────────────────────────────────────────────────────────
cat << BANNER

  ╔══════════════════════════════════════════════════════════════╗
  ║   M1 Max · Qwen3.5 4B · Vision · APC · Creative Writing     ║
  ╚══════════════════════════════════════════════════════════════╝

BANNER

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}')
info "Chip: $CHIP  |  Memory: ${MEM} GB  |  Context: ${CTX} tokens"
info "Model: Qwen3.5 4B 4-bit (~2.8 GB)"
echo ""

# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Install mlx-vlm from git main
# ═══════════════════════════════════════════════════════════════════════
log "Step 1/4: Checking mlx-vlm..."

if python3 -c "from mlx_vlm.server import app" 2>/dev/null; then
    VER=$(python3 -c "import mlx_vlm; print(mlx_vlm.__version__)" 2>/dev/null)
    info "  Already installed: v$VER"
else
    log "  Installing mlx-vlm (this may take 2-3 minutes)..."
    pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@main" 2>&1 | \
        grep -E "Successfully|ERROR|Already" || true
    info "  Installed."
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Download model
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 2/4: Model..."

mkdir -p "$MODEL_DIR"

if [ -f "$TARGET_LOCAL/model.safetensors" ]; then
    SIZE=$(du -sh "$TARGET_LOCAL" 2>/dev/null | cut -f1)
    info "  Already downloaded: $SIZE"
else
    log "  Downloading from HuggingFace (~2.8 GB)..."
    python3 -c "
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download
snapshot_download('$TARGET', local_dir='$TARGET_LOCAL', local_dir_use_symlinks=False)
" 2>&1 | tail -2
    info "  Downloaded: $(du -sh "$TARGET_LOCAL" 2>/dev/null | cut -f1)"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Setup APC cache
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 3/4: APC cache..."

mkdir -p "$CACHE_DIR"
info "  L1 RAM cache  |  $APC_BLOCKS blocks"
info "  L2 disk cache |  $CACHE_DIR  (max ${CACHE_GB} GB)"

export APC_ENABLED=1
export APC_NUM_BLOCKS="$APC_BLOCKS"
export APC_DISK_PATH="$CACHE_DIR"
export APC_DISK_MAX_GB="$CACHE_GB"

# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Start server
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 4/4: Starting server..."

# ── Endpoint display ──────────────────────────────────────────────────
cat << ENDPOINT

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   ${BOLD}🎯  Endpointhttp://${HOST}:${PORT}${NC}                                  │
  │                                                                  │
  │   ${BOLD}📡  APIhttp://${HOST}:${PORT}/v1/chat/completions${NC}                     │
  │   ${BOLD}💚  Healthhttp://${HOST}:${PORT}/health${NC}                                  │
  │   ${BOLD}📊  Cache Statshttp://${HOST}:${PORT}/v1/cache/stats${NC}                          │
  │                                                                  │
  │   ${BOLD}Configuration:${NC}                                                │
  │   • Model:      Qwen3.5 4B 4-bit                                 │
  │   • Vision:     ✅ native early-fusion                            │
  │   • Cache:      ✅ APC L1+L2 persistent                           │
  │   • Context:    ${CTX} tokens                                        │
  │   • Thinking:   OFF by default (creative mode)                   │
  │   • Defaults:   temp=0.9  top_p=0.95  presence_penalty=1.5       │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  ── Quick test ──────────────────────────────────────────────────

  curl http://${HOST}:${PORT}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
      "model": "${TARGET_LOCAL}",
      "messages": [{"role":"user","content":"Write a haiku about silence."}],
      "max_tokens": 64,
      "temperature": 0.9,
      "top_p": 0.95,
      "presence_penalty": 1.5,
      "chat_template_kwargs": {"enable_thinking": false}
    }'

  ────────────────────────────────────────────────────────────────

ENDPOINT

exec python3 -m mlx_vlm.server \
  --model "$TARGET_LOCAL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-kv-size "$CTX" \
  "$@"
