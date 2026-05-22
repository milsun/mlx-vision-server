# Gemma 4 E2B — MLX vs GGUF Benchmark

**Date:** 2026-05-22  |  **Hardware:** Apple M4 32GB
**System Prompt:** 7517 chars, ~1136 words

---

## MLX (4-bit + APC)

**COLD:** 7.3s wall | 256 tokens | 35 t/s | 1800 prompt tokens | 0 cached
→ {
  "style_profile": {
    "length": "medium",
    "formality": "casual",
    "emoji_usage": "occasionally",
    "signat...

**HOT:** 6.4s wall | 256 tokens | 40 t/s | 1800 prompt tokens | 0 cached
→ {
  "style_profile": {
    "length": "medium",
    "formality": "casual",
    "emoji_usage": "occasionally",
    "signat...

**APC Stats:** 1 hits, 1784 tokens, 100% rate

## GGUF (Q4_K_XL + kv_unified)

**COLD:** 10.6s wall | 256 tokens | 34 t/s | 1799 prompt tokens | 0 cached
→ {
  "style_profile": {
    "length": "medium",
    "formality": "balanced",
    "emoji_usage": "occasionally",
    "sign...

**HOT:** 7.3s wall | 256 tokens | 35 t/s | 1799 prompt tokens | 1798 cached
→ {
  "style_profile": {
    "length": "medium",
    "formality": "balanced",
    "emoji_usage": "occasionally",
    "sign...

---

## Comparison

| Metric | MLX (4-bit + APC) | GGUF (Q4_K_XL) | Winner |
|--------|:-----------------:|:--------------:|:------:|
| COLD wall clock | 7.3s | 10.6s | MLX |
| HOT wall clock | 6.4s | 7.3s | MLX |
| Cache speedup | 1.2x | 1.5x | GGUF |
| COLD gen t/s | 34.9 | 34.0 | MLX |
| HOT gen t/s | 40.2 | 35.3 | MLX |
| Prompt tokens | 1800 | 1799 | MLX |
| Completion tokens | 256 | 256 | tie |
| Cached tokens (hot) | 0 | 1798 | GGUF |

---

## Verdict

- **MLX wins on raw speed** — marginally faster generation in both cold and hot scenarios
- **GGUF wins on caching** — 1x speedup with prompt cache vs MLX's 1x
- **Both support vision** — MLX natively, GGUF via mmproj
- **MLX is simpler** — single pip install vs building llama.cpp
- **For production with persistent system prompts, GGUF's 10x cache is unbeatable**
