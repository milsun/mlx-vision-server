#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Gemma 4 E2B 2B · Vision · APC Cache · Max Speed
#  Single command:  ./setup_gemma4.sh
#  Requires: Python 3.10+
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
info() { echo -e "${CYAN}[*]${NC} $*"; }
err()  { echo -e "${RED}[!]${NC} $*"; exit 1; }

# ── Config ───────────────────────────────────────────────────────────
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

# ── Step 0: Check prerequisites ──────────────────────────────────────
echo ""
log "Step 0/4: Checking prerequisites..."

PYTHON=""
for py in python3 python3.12 python3.11 python3.10; do
    if command -v $py &>/dev/null && $py -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
        PYTHON="$py"
        break
    fi
done
[ -z "$PYTHON" ] && err "Python 3.10+ required. Install: brew install python@3.12"
PYVER=$($PYTHON --version 2>&1)
info "  $PYVER ($(which $PYTHON))"

if ! $PYTHON -m pip --version &>/dev/null; then
    err "pip not available. Run: $PYTHON -m ensurepip --upgrade"
fi
info "  pip: $($PYTHON -m pip --version | cut -d' ' -f1-2)"

# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Install dependencies
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 1/4: Installing dependencies..."

$PYTHON -m pip install --quiet huggingface_hub 2>/dev/null || true

if $PYTHON -c "from mlx_vlm.server import app" 2>/dev/null; then
    VER=$($PYTHON -c "import mlx_vlm; print(mlx_vlm.__version__)" 2>/dev/null)
    info "  mlx-vlm v$VER (already installed)"
else
    log "  Installing mlx-vlm (pip install from git, ~2-3 min)..."
    $PYTHON -m pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@main" 2>&1 | \
        grep -E "Successfully|ERROR" || true
    $PYTHON -c "from mlx_vlm.server import app" 2>/dev/null || \
        err "mlx-vlm failed to install. Check pip and internet connection."
    info "  Installed: v$($PYTHON -c 'import mlx_vlm; print(mlx_vlm.__version__)')"
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
    log "  Downloading from HuggingFace (~3.3 GB, one-time)..."
    $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('$TARGET', local_dir='$TARGET_LOCAL', local_dir_use_symlinks=False)
print('Download complete.')
" || err "Model download failed. Check internet connection and try again."
    info "  Downloaded: $(du -sh "$TARGET_LOCAL" | cut -f1)"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Setup APC cache
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 3/4: APC cache..."
mkdir -p "$CACHE_DIR"
info "  L1 RAM  |  $APC_BLOCKS blocks"
info "  L2 disk |  $CACHE_DIR  (max ${CACHE_GB} GB, survives restarts)"

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
  │   Cache:      ✅ APC L1+L2 persistent                            │
  │   Context:    ${CTX} tokens                                          │
  │   Thinking:   OFF by default (toggle via API)                    │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

ENDPOINT

exec $PYTHON -m mlx_vlm.server \
  --model "$TARGET_LOCAL" \
  --host "$HOST" \
  --port "$PORT" \
  --max-kv-size "$CTX" \
  "$@"
