# Speculative Decoding for Qwen3.5 on Mac M4 — Complete Analysis

**Date:** 2026-05-22 | **Hardware:** Apple M4 (10-core, 32GB RAM, Metal GPU)

---

## Executive Summary

| Approach | Framework | Speedup | Stability | Verdict |
|----------|-----------|---------|-----------|---------|
| **MTP** (llama.cpp GGUF) | llama.cpp | **SLOWER** (25 vs 42 tok/s) | ✅ stable | ❌ Counterproductive for 2B |
| **DFlash** (mlx-vlm built-in) | mlx-vlm 0.5.0 | **CRASHED** | ❌ unstable | ❌ Version mismatch |
| **DFlash** (Aryagm/dflash-mlx) | dflash-mlx v0.1.0 | **SLOWER** (17 vs 30 tok/s) | ❌ crashed | ❌ Qwen3.5 path not optimized |
| **DFlash** (bstnxbt/dflash-mlx) | dflash-mlx v0.1.7 | **NO GAIN** (30 vs 30 tok/s) | ✅ stable | ⚠️ No benefit with 4-bit |

---

## 1. MTP (Multi-Token Prediction) via llama.cpp GGUF

### Setup
```bash
llama-server -m Qwen3.5-2B-MTP-IQ4_XS.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ngl 99
```

### Results
| Prompt | MTP | Regular | Delta |
|--------|-----|---------|-------|
| Short (32t) | 33 t/s | 43 t/s | **-23%** |
| Medium (128t) | 28 t/s | 42 t/s | **-33%** |
| Long (256t) | 25 t/s | 41 t/s | **-39%** |

Draft acceptance rate: 49-60% (drops with longer outputs)

### Why it fails
- MTP draft model is the same size as the target for 2B models
- Draft computation overhead > benefit for small models
- GPU memory bandwidth contention between target + draft
- **Designed for large models (26B+)** where draft is tiny relative to target

### Files downloaded
- `unsloth/Qwen3.5-2B-MTP-GGUF` → `Qwen3.5-2B-MTP-IQ4_XS.gguf` (1.2 GB)

---

## 2. DFlash via mlx-vlm built-in

### Setup
```bash
python3 -m mlx_vlm.server \
  --model Qwen3.5-4B-4bit \
  --draft-model z-lab/Qwen3.5-4B-DFlash
```

### Results
❌ **Crashed:** `TypeError: concatenate(): incompatible function arguments`
- Version mismatch between mlx-vlm 0.5.0 and DFlash drafter
- mlx-vlm's DFlash integration is not production-ready for Qwen3.5

---

## 3. DFlash via Aryagm/dflash-mlx (original fork)

### Setup
```bash
pip install dflash-mlx  # from Aryagm repo
dflash-mlx-openai-server --target-model Qwen3.5-4B-4bit --draft-model z-lab/Qwen3.5-4B-DFlash
```

### Results
| Prompt | DFlash | Regular MLX | Delta |
|--------|--------|-------------|-------|
| Short (32t) | 6 t/s | 30 t/s | **-80%** |
| Medium (128t) | 17 t/s | 30 t/s | **-43%** |

❌ Crashed on long generation. README states: "Qwen3.5 support is functional but incomplete. It is not as fast as the Qwen3 path today."

---

## 4. DFlash via bstnxbt/dflash-mlx (optimized fork, v0.1.7)

**707 stars, actively maintained. Best DFlash implementation for MLX.**

### Setup
```bash
pip install dflash-mlx  # from PyPI (bstnxbt)
dflash serve \
  --model mlx-community/Qwen3.5-4B-4bit \
  --port 8000 \
  --chat-template-args '{"enable_thinking":false}'
```

### Features
- Tape-replay rollback for GatedDeltaNet state
- `verify_qmm` custom Metal kernel for M=16 quantized matmul
- Prefix cache L1+L2 (RAM + SSD)
- Adaptive verify policy (auto-tunes block size)
- OpenAI-compatible `/v1/chat/completions` + streaming
- Live `/metrics` endpoint

### Our Results (M4, 4-bit quant)
| Tokens | DFlash | Regular MLX | Speedup |
|--------|--------|-------------|---------|
| 32 | 5.2 t/s | 30 t/s | 0.17x |
| 128 | 23.2 t/s | 30 t/s | 0.77x |
| 256 | 31.5 t/s | 30 t/s | 1.05x |
| 512 | 30.4 t/s | 30 t/s | 1.01x |
| 2048 | 29.3 t/s | 30 t/s | 0.98x |

**Acceptance rate:** 65.8% (adaptive blocks from 2-16)

### Published Benchmarks (M5 Max, BF16) — for comparison
| Tokens | Baseline | DFlash | Speedup |
|--------|----------|--------|---------|
| 1024 | 53.8 t/s | 182.9 t/s | **3.40x** |
| 2048 | 53.9 t/s | 188.7 t/s | **3.49x** |
| 4096 | 53.5 t/s | 195.8 t/s | **3.66x** |
| 8192 | 53.3 t/s | 160.5 t/s | **3.02x** |

---

## Why DFlash Doesn't Help on Our Setup

1. **4-bit quantization bottleneck**: Qwen3.5 4B at 4-bit is already bandwidth-bound (~30 tok/s). DFlash saves compute, but our bottleneck is memory bandwidth, not compute.
2. **M4 vs M5 Max**: Published benchmarks use M5 Max with higher bandwidth and BF16 models. M4 has lower GPU bandwidth.
3. **Model is too small**: At 4B params, the draft overhead (~1B DFlash model) is significant relative to target computation. DFlash shines on 9B+ models.

## When DFlash WILL Help

- **BF16/fp16 models**: When target is compute-bound (BF16 4B ~53 tok/s), DFlash gives 3-4x
- **Larger models**: Qwen3.5 9B or 27B where draft overhead is proportionally smaller
- **M4 Pro/Max**: Higher memory bandwidth would shift the bottleneck from bandwidth to compute
- **Long generations**: >1024 tokens where the per-cycle savings accumulate (published benchmarks show this)

## Recommendation for Your Setup

For **max speed on M4 with small models**, skip speculative decoding. The 4-bit quantized models are already bandwidth-optimized. Focus on:
- Direct MLX inference with 4-bit quants (already at max speed)
- Use bstnxbt/dflash-mlx only when running BF16 or 9B+ models
- Cloud vision APIs (Gemini Flash Lite) for zero-local-latency vision

## Files Downloaded

| File | Size | Purpose |
|------|------|---------|
| `Qwen3.5-2B-MTP-IQ4_XS.gguf` | 1.2 GB | MTP speculative (llama.cpp) |
| `Qwen3.5-4B-DFlash/` | 1.0 GB | DFlash drafter (z-lab) |
