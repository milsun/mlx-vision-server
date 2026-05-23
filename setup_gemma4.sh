#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Gemma 4 E2B 2B · Vision · APC Cache · Max Speed
#  Single command:  ./setup_gemma4.sh
#  Best for single-user production with persistent system prompts.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
info() { echo -e "${CYAN}[*]${NC} $*"; }

# ── Config (override via env) ────────────────────────────────────────
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma4-e2b}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/mlx-vlm/apc-gemma}"
CACHE_GB="${CACHE_GB:-25}"
APC_BLOCKS="${APC_BLOCKS:-4096}"
CTX="${CTX:-32768}"

TARGET="mlx-community/gemma-4-e2b-it-4bit"
TARGET_LOCAL="$MODEL_DIR/gemma-4-e2b-it-4bit"

# ── Banner ───────────────────────────────────────────────────────────
cat << BANNER

  ╔══════════════════════════════════════════════════════════════╗
  ║   Gemma 4 E2B 2B · Vision · APC Cache · Max Speed          ║
  ╚══════════════════════════════════════════════════════════════╝

BANNER

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}')
info "Chip: $CHIP  |  Memory: ${MEM} GB  |  Context: ${CTX} tokens"
info "Model: Gemma 4 E2B 2B 4-bit (~3.3 GB)"
info "Speed: ~35 t/s (M4)  |  ~100 t/s (M1 Max)  |  cold 7s / hot 6s"
echo ""

# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Install mlx-vlm from git main
# ═══════════════════════════════════════════════════════════════════════
log "Step 1/4: Installing mlx-vlm..."

# Check if mlx-vlm is already installed and working
if python3 -c "from mlx_vlm.server import app" 2>/dev/null; then
    VER=$(python3 -c "import mlx_vlm; print(mlx_vlm.__version__)" 2>/dev/null)
    info "  Already installed: $VER"
    # Try upgrading in background — don't block startup
    pip install --upgrade "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@main" 2>/dev/null &
else
    log "  Installing mlx-vlm (this may take a few minutes)..."
    pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@main" 2>&1 | \
        grep -E "Successfully|ERROR|Collecting" || true
    info "  Installed."
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Download model
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 2/4: Model..."
mkdir -p "$MODEL_DIR"

if [ -f "$TARGET_LOCAL/model.safetensors" ]; then
    info "  Already downloaded: $(du -sh "$TARGET_LOCAL" | cut -f1)"
else
    log "  Downloading from HuggingFace (~3.3 GB)..."
    pip install -q huggingface_hub 2>/dev/null || true
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$TARGET', local_dir='$TARGET_LOCAL', local_dir_use_symlinks=False)
" 2>&1 | tail -2
    info "  Downloaded: $(du -sh "$TARGET_LOCAL" | cut -f1)"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Setup APC cache
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 3/4: APC cache..."
mkdir -p "$CACHE_DIR"
info "  L1 RAM  |  $APC_BLOCKS blocks"
info "  L2 disk |  $CACHE_DIR  (max ${CACHE_GB} GB, persists across restarts)"

export APC_ENABLED=1
export APC_NUM_BLOCKS="$APC_BLOCKS"
export APC_DISK_PATH="$CACHE_DIR"
export APC_DISK_MAX_GB="$CACHE_GB"

# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Start server
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 4/4: Starting server..."
cat << ENDPOINT

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   🎯  Endpoint    http://${HOST}:${PORT}                                    │
  │   📡  API         http://${HOST}:${PORT}/v1/chat/completions               │
  │   💚  Health      http://${HOST}:${PORT}/health                            │
  │   📊  Cache Stats http://${HOST}:${PORT}/v1/cache/stats                    │
  │                                                                  │
  │   Model:      Gemma 4 E2B 2B 4-bit (~3.3 GB)                    │
  │   Vision:     ✅ native (no mmproj needed)                       │
  │   Cache:      ✅ APC L1+L2 (persistent, 1.14x speedup)          │
  │   Context:    ${CTX} tokens                                          │
  │   Thinking:   OFF by default                                     │
  │                                                                  │
  │   Per-request (big prompt, M4):                                  │
  │     Cold: ~7s    Hot: ~6s (cached)                              │
  │   Per-request (M1 Max est.):                                     │
  │     Cold: ~2s    Hot: ~1.5s (cached)                            │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

ENDPOINT

exec python3 -m mlx_vlm.server \
  --model "$TARGET_LOCAL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-kv-size "$CTX" \
  "$@"
