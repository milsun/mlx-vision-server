#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Gemma 4 E2B 2B — Vision + Prompt Cache + Max Speed (llama.cpp)
#  Fastest vision model. Cold start ~2s, cached ~0.2s.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${GREEN}[+]${NC} $*"; }
info() { echo -e "${CYAN}[*]${NC} $*"; }

# ── Config ───────────────────────────────────────────────────────────
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma4-e2b}"
CTX="${CTX:-16384}"
THREADS="${THREADS:-3}"
BATCH="${BATCH:-4096}"

MODEL_REPO="unsloth/gemma-4-E2B-it-GGUF"
MODEL_FILE="gemma-4-E2B-it-UD-Q4_K_XL.gguf"
MMPROJ_REPO="unsloth/gemma-4-E2B-it-GGUF"
MMPROJ_FILE="mmproj-F16.gguf"

MODEL="$MODEL_DIR/$MODEL_FILE"
MMPROJ="$MODEL_DIR/$MMPROJ_FILE"

# ── Banner ───────────────────────────────────────────────────────────
cat << BANNER

  ╔══════════════════════════════════════════════════════════════╗
  ║   Gemma 4 E2B 2B · Vision · Prompt Cache · Max Speed        ║
  ╚══════════════════════════════════════════════════════════════╝

BANNER

CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}')
info "Chip: $CHIP  |  Memory: ${MEM} GB  |  Context: ${CTX} tokens"
info "Model: Gemma 4 E2B 2B (3.0 GB GGUF + 0.9 GB mmproj)"
info "Speed: ~37 tok/s (M4)  |  ~100+ tok/s (M1 Max)"
echo ""

# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Download models
# ═══════════════════════════════════════════════════════════════════════
log "Step 1/3: Downloading models..."
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL" ]; then
    info "  GGUF: $(du -sh "$MODEL" | cut -f1) ✓"
else
    log "  Downloading GGUF (~3 GB)..."
    curl -L "https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE" -o "$MODEL" 2>&1 | tail -1
fi

if [ -f "$MMPROJ" ]; then
    info "  mmproj: $(du -sh "$MMPROJ" | cut -f1) ✓"
else
    log "  Downloading mmproj (~0.9 GB)..."
    curl -L "https://huggingface.co/$MMPROJ_REPO/resolve/main/$MMPROJ_FILE" -o "$MMPROJ" 2>&1 | tail -1
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Check/build llama.cpp
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 2/3: Checking llama.cpp..."

LLAMA_DIR="$HOME/llama.cpp"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"

if [ ! -x "$LLAMA_SERVER" ]; then
    log "  Building llama.cpp..."
    if [ ! -d "$LLAMA_DIR" ]; then
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR" 2>&1 | tail -1
    fi
    cd "$LLAMA_DIR"
    cmake -B build \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DGGML_NATIVE=ON \
        -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
    cmake --build build --config Release -j8 2>&1 | tail -1
    info "  Built: $LLAMA_SERVER"
else
    info "  Found: $LLAMA_SERVER"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Start server
# ═══════════════════════════════════════════════════════════════════════
echo ""
log "Step 3/3: Starting server..."

cat << ENDPOINT

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   🎯  Endpoint    http://${HOST}:${PORT}                                    │
  │                                                                  │
  │   📡  API         http://${HOST}:${PORT}/v1/chat/completions               │
  │   💚  Health      http://${HOST}:${PORT}/health                            │
  │   📊  Models      http://${HOST}:${PORT}/v1/models                         │
  │                                                                  │
  │   Configuration:                                                 │
  │   • Model:      Gemma 4 E2B 2B (GGUF Q4_K_XL)                   │
  │   • Vision:     ✅ (mmproj loaded)                               │
  │   • Cache:      ✅ kv_unified (auto prompt cache)               │
  │   • Context:    ${CTX} tokens                                        │
  │   • Speed:      ~0.2s hot / ~2s cold (M4)                       │
  │   • Thinking:   OFF                                              │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  ── Quick test ──────────────────────────────────────────────────

  curl http://${HOST}:${PORT}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
      "messages": [{"role":"user","content":[
        {"type":"text","text":"Describe this image."},
        {"type":"image_url","image_url":{"url":"https://example.com/photo.jpg"}}
      ]}],
      "max_tokens": 128,
      "temperature": 0
    }'

  ────────────────────────────────────────────────────────────────

ENDPOINT

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  -mm "$MMPROJ" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl 99 \
  -t "$THREADS" \
  -c "$CTX" \
  -b "$BATCH" \
  -fa 1 \
  -ctk q8_0 \
  -ctv q8_0 \
  --reasoning off \
  "$@"
