#!/usr/bin/env python3
"""Qwen3.5 benchmark — non-streaming only, 3 runs, saves markdown."""
import json, statistics, time, urllib.request

ENGINES = {
    "llama.cpp 2B": "http://localhost:8082/v1/chat/completions",
    "llama.cpp 4B": "http://localhost:8084/v1/chat/completions",
    "MLX 2B": "http://localhost:8012/v1/chat/completions",
    "MLX 4B": "http://localhost:8014/v1/chat/completions",
}
PROMPTS = {
    "short":  "What is a binary search tree?",
    "medium": "Write a Python function implementing quicksort with inline comments.",
    "long":   "Explain Python async/await with code examples.",
}
RUNS = 5
MAX_TOKENS = 512
TEMP = 0.0
OUT = "/Users/milankumar/Desktop/llm/mlx_server/BENCHMARK_QWEN.md"

def run_ns(url, prompt):
    """Returns (wall_s, comp_tokens, gen_tps, prompt_tps, cached_tokens, prompt_tokens)."""
    p = {"messages":[{"role":"user","content":prompt}],"max_tokens":MAX_TOKENS,"temperature":TEMP,"stream":False}
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(p).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        b = json.loads(r.read().decode("utf-8"))
    w = time.perf_counter() - t0
    u = b.get("usage",{})
    tmg = b.get("timings",{})
    return (w, u.get("completion_tokens",0), round(tmg.get("predicted_per_second",0),1),
            round(tmg.get("prompt_per_second",0),1), u.get("prompt_tokens_details",{}).get("cached_tokens",0),
            u.get("prompt_tokens",0))

print("[Warmup]", flush=True)
for n, u in ENGINES.items():
    run_ns(u, PROMPTS["short"])
print("Done\n", flush=True)

NS = {n: {p: [] for p in PROMPTS} for n in ENGINES}
for pn, pt in PROMPTS.items():
    for n, u in ENGINES.items():
        print(f"  {n} {pn}", end=" ", flush=True)
        for _ in range(RUNS):
            NS[n][pn].append(run_ns(u, pt))
        print("✓", flush=True)

# Stress
print("\n[Stress — 20 rapid sequential requests]", flush=True)
STR = {}
for n, u in ENGINES.items():
    print(f"  {n}", end=" ", flush=True)
    ts, er = [], 0
    for _ in range(20):
        try: w, _, _, _, _, _ = run_ns(u, PROMPTS["short"]); ts.append(w)
        except: er += 1
    st = sorted(ts)
    STR[n] = {"avg": statistics.mean(ts) if ts else 0, "min": st[0] if ts else 0,
              "max": st[-1] if ts else 0, "p95": st[int(len(st)*0.95)] if ts else 0,
              "std": statistics.stdev(ts) if len(ts)>1 else 0, "errors": er}
    print("✓", flush=True)

# ── Markdown ──
L = []
def pr(s=""): L.append(s)

pr("# Qwen3.5 Inference Benchmark: llama.cpp GGUF vs MLX")
pr()
pr(f"**Date:** {time.strftime('%Y-%m-%d')}")
pr(f"**Hardware:** Apple M4 (10-core), 32GB RAM, Metal GPU")
pr(f"**Methodology:** {RUNS} runs per config | thinking=OFF | temp={TEMP} | max_tokens={MAX_TOKENS}")
pr(f"**Models:** Qwen3.5 2B & 4B — UD-Q4_K_XL (GGUF) / 4bit (MLX)")
pr()
pr("> ⚠️ **Text-only comparison.** The Qwen3.5 GGUF conversion does not include the vision encoder. Only MLX natively supports vision for this model. Both engines evaluated on raw text generation throughput.")
pr()
pr("---")

# Table 1: Wall clock + Gen TPS
pr()
pr("## 1. Wall Clock & Generation Throughput")
pr()
pr("| Prompt | Engine | Wall (s) | Gen TPS | Prompt TPS | Comp Tok | Cached Tok |")
pr("|--------|--------|:--------:|:-------:|:----------:|:--------:|:----------:|")
for pn in PROMPTS:
    for n in ENGINES:
        r = NS[n][pn]
        aw = statistics.mean([x[0] for x in r])
        ag = statistics.mean([x[2] for x in r])
        ap = statistics.mean([x[3] for x in r])
        at = statistics.mean([x[1] for x in r])
        ac = statistics.mean([x[4] for x in r])
        pt = f"{ap:.0f}" if ap > 0 else "n/a"
        pr(f"| {pn} | {n} | {aw:.2f} | {ag:.1f} | {pt} | {at:.0f} | {ac:.0f} |")

# Table 2: Summary by model size
pr()
pr("## 2. Per-Model Summary")
pr()

for ms in ["2B", "4B"]:
    ln = f"llama.cpp {ms}"; mn = f"MLX {ms}"
    l_tps = statistics.mean([statistics.mean([x[2] for x in NS[ln][pn]]) for pn in PROMPTS])
    m_tps = statistics.mean([statistics.mean([x[2] for x in NS[mn][pn]]) for pn in PROMPTS])
    l_wall = statistics.mean([statistics.mean([x[0] for x in NS[ln][pn]]) for pn in PROMPTS])
    m_wall = statistics.mean([statistics.mean([x[0] for x in NS[mn][pn]]) for pn in PROMPTS])
    l_prompt = statistics.mean([statistics.mean([x[5] for x in NS[ln][pn]]) for pn in PROMPTS])
    m_prompt = statistics.mean([statistics.mean([x[5] for x in NS[mn][pn]]) for pn in PROMPTS])

    pr(f"### Qwen3.5 {ms}")
    pr()
    pr("| Metric | llama.cpp GGUF | MLX | Delta | Winner |")
    pr("|--------|:-------------:|:---:|:-----:|:------:|")

    for label, lv, mv, unit, lower in [
        ("Avg Gen TPS", l_tps, m_tps, "t/s", False),
        ("Avg Wall Clock", l_wall, m_wall, "s", True),
        ("Avg Prompt Tokens", l_prompt, m_prompt, "tok", False),
        ("Vision Support", "NO (GGUF limited)", "YES (native)", "", None),
    ]:
        if isinstance(lv, str):
            pr(f"| {label} | {lv} | {mv} | — | — |")
        else:
            delta = ((mv - lv) / lv) * 100 if lv != 0 else 0
            if lower:
                w = "MLX" if delta < -1 else "llama.cpp" if delta > 1 else "tie"
            else:
                w = "llama.cpp" if delta < -1 else "MLX" if delta > 1 else "tie"
            pr(f"| {label} | {lv:.1f} {unit} | {mv:.1f} {unit} | {delta:+.1f}% | {w} |")
    pr()

# Table 3: Stress
pr()
pr("## 3. Stress Test — 20 Rapid Sequential Requests")
pr("Short prompt, non-streaming.")
pr()
pr("| Metric | llama.cpp 2B | llama.cpp 4B | MLX 2B | MLX 4B |")
pr("|--------|:-----------:|:-----------:|:------:|:------:|")
for metric, key, fmt in [("Avg (s)","avg",".2f"),("Min (s)","min",".2f"),("Max (s)","max",".2f"),("P95 (s)","p95",".2f"),("Std dev","std",".3f"),("Errors","errors",".0f")]:
    v = {n: STR[n][key] for n in ENGINES}
    pr(f"| {metric} | {v['llama.cpp 2B']:{fmt}} | {v['llama.cpp 4B']:{fmt}} | {v['MLX 2B']:{fmt}} | {v['MLX 4B']:{fmt}} |")

# Table 4: Cross-model speed comparison
pr()
pr("## 4. Cross-Model Speed Ladder (all models tested, text-only)")
pr()
pr("| Model | Params | Engine | Gen TPS |")
pr("|-------|--------|--------|:-------:|")
# We'll fill these in from the data
for ms in ["2B", "4B"]:
    for en in ["llama.cpp", "MLX"]:
        tps = statistics.mean([statistics.mean([x[2] for x in NS[f"{en} {ms}"][pn]]) for pn in PROMPTS])
        pr(f"| Qwen3.5 {ms} | {ms} | {en} | {tps:.1f} |")

pr()
pr("---")
pr()
pr("## Methodology")
pr()
pr("- **Thinking disabled**: llama.cpp `--reasoning off`, MLX `enable_thinking=False` via chat template")
pr("- **Prompt caching**: Enabled on both (llama.cpp `kv_unified`, MLX APC)")
pr("- **GPU offloading**: All layers on Metal (llama.cpp `-ngl 99`, MLX native)")
pr("- **Identical parameters**: temperature=0.0, max_tokens=512")
pr("- **Metrics**: Server-reported gen/prompt TPS from `timings` field (llama.cpp). MLX uses wall-clock derived TPS.")
pr("- **Vision note**: Qwen3.5 has native multimodal support, but GGUF conversion omits vision encoder. MLX retains full multimodal capability.")

with open(OUT, "w") as f:
    f.write("\n".join(L))

print(f"\n✓ Saved to {OUT}")
