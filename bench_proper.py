#!/usr/bin/env python3
"""Proper benchmark: TTFT, TPS, stress test — llama.cpp vs MLX for Qwen3.5 4B."""

import argparse
import json
import statistics
import sys
import time
import urllib.request

LLAMA_URL = "http://localhost:8080/v1/chat/completions"
MLX_URL = "http://localhost:8010/v1/chat/completions"

PROMPTS = {
    "short": "Explain what a linked list is.",
    "medium": "Write a Python function that reverses a string without using built-in methods. Include comments explaining each step.",
    "long": "Write a detailed explanation of how garbage collection works in Python. Cover reference counting, cyclic garbage collector, and generational collection. Include code examples.",
}

def run_stream(api_url: str, prompt: str, label: str):
    """Streaming request — measures TTFT + per-chunk timing."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": True,
    }
    t_request = time.perf_counter()
    req = urllib.request.Request(
        api_url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=120)
    t_first = None
    total_text = ""
    chunk_count = 0
    token_count = 0
    t_last = t_request

    while True:
        line = resp.readline().decode("utf-8")
        if not line:
            break
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            chunk_count += 1
            now = time.perf_counter()
            if t_first is None:
                t_first = now
                ttft = t_first - t_request
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    total_text += content
                    token_count += 1
            t_last = now

    t_end = time.perf_counter()
    total_time = t_end - t_request
    ttft = (t_first - t_request) if t_first else total_time
    gen_time = t_last - t_first if t_first else total_time
    tps = token_count / gen_time if gen_time > 0 and token_count > 0 else 0

    return {
        "ttft_s": round(ttft, 3),
        "total_time_s": round(total_time, 3),
        "gen_time_s": round(gen_time, 3),
        "tokens": token_count,
        "chunks": chunk_count,
        "tps": round(tps, 1),
        "output_len": len(total_text),
    }


def run_nonstream(api_url: str, prompt: str):
    """Non-streaming — system-wall-clock + server timings."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        api_url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read().decode("utf-8"))
    wall = time.perf_counter() - t0
    timings = body.get("timings", {})
    usage = body.get("usage", {})
    return {
        "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "server_prompt_tps": round(timings.get("prompt_per_second", 0), 1),
        "server_gen_tps": round(timings.get("predicted_per_second", 0), 1),
        "server_prompt_ms": timings.get("prompt_ms", 0),
        "server_gen_ms": timings.get("predicted_ms", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    engines = {
        "llama.cpp": LLAMA_URL,
        "MLX": MLX_URL,
    }

    print(f"\n{'#'*80}")
    print(f"  QWEN3.5 4B BENCHMARK — {args.runs} runs per prompt × 2 engines")
    print(f"  Mode: text-only (GGUF has no vision encoder)")
    print(f"{'#'*80}")

    # ── Phase 1: TTFT via streaming ─────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  PHASE 1: Streaming — measuring TTFT + TPS")
    print(f"{'─'*80}")

    streaming_results = {e: {} for e in engines}

    for pname, prompt in PROMPTS.items():
        print(f"\n  [Prompt: {pname}]")
        for ename, url in engines.items():
            results = []
            print(f"    {ename}: ", end="", flush=True)
            for i in range(args.runs):
                r = run_stream(url, prompt, ename)
                results.append(r)
                print(f"r{i+1}:{r['ttft_s']:.2f}s/{r['tps']:.0f}tps ", end="", flush=True)
            print()

            avg_ttft = statistics.mean([r["ttft_s"] for r in results])
            avg_tps = statistics.mean([r["tps"] for r in results])
            std_ttft = statistics.stdev([r["ttft_s"] for r in results]) if len(results) > 1 else 0
            streaming_results[ename][pname] = {
                "avg_ttft": avg_ttft, "std_ttft": std_ttft,
                "avg_tps": avg_tps, "min_tps": min(r["tps"] for r in results),
                "max_tps": max(r["tps"] for r in results),
            }

    # ── Phase 2: Non-streaming with server timings ──────────────
    print(f"\n{'─'*80}")
    print(f"  PHASE 2: Non-streaming — wall clock + server-reported timings")
    print(f"{'─'*80}")

    nonstream_results = {e: {} for e in engines}

    for pname, prompt in PROMPTS.items():
        print(f"\n  [Prompt: {pname}]")
        for ename, url in engines.items():
            results = []
            print(f"    {ename}: ", end="", flush=True)
            for i in range(args.runs):
                r = run_nonstream(url, prompt)
                results.append(r)
                if r["server_gen_tps"]:
                    print(f"r{i+1}:{r['wall_s']:.1f}s/{r['server_gen_tps']:.0f}tps ", end="", flush=True)
                else:
                    print(f"r{i+1}:{r['wall_s']:.1f}s/{r['completion_tokens']/r['wall_s']:.0f}tps ", end="", flush=True)
            print()

            avg_wall = statistics.mean([r["wall_s"] for r in results])
            avg_tps = statistics.mean([
                r["server_gen_tps"] if r["server_gen_tps"] > 0
                else r["completion_tokens"] / r["wall_s"]
                for r in results
            ])
            avg_prompt_tps = statistics.mean([r["server_prompt_tps"] for r in results if r["server_prompt_tps"]] or [0])

            nonstream_results[ename][pname] = {
                "avg_wall": avg_wall,
                "avg_tps": avg_tps,
                "avg_prompt_tps": avg_prompt_tps,
            }

    # ── Phase 3: Stress test (quick repeated calls) ─────────────
    print(f"\n{'─'*80}")
    print(f"  PHASE 3: Stress test — 20 rapid sequential requests (short prompt)")
    print(f"{'─'*80}")

    stress_results = {}
    for ename, url in engines.items():
        times = []
        errors = 0
        print(f"    {ename}: ", end="", flush=True)
        for i in range(20):
            try:
                r = run_nonstream(url, PROMPTS["short"])
                times.append(r["wall_s"])
                if (i + 1) % 5 == 0:
                    print(f"[{i+1}] ", end="", flush=True)
            except Exception as e:
                errors += 1
        print()
        stress_results[ename] = {
            "avg": statistics.mean(times) if times else 0,
            "min": min(times) if times else 0,
            "max": max(times) if times else 0,
            "std": statistics.stdev(times) if len(times) > 1 else 0,
            "p95": sorted(times)[int(len(times) * 0.95)] if times else 0,
            "errors": errors,
        }

    # ── TABLES ──────────────────────────────────────────────────

    def winner(a, b, lower_better=True):
        if abs(a - b) < 0.01:
            return "tie"
        if lower_better:
            return "MLX" if b < a else "llama.cpp"
        return "llama.cpp" if a > b else "MLX"

    # Table 1: TTFT
    print(f"\n{'='*80}")
    print(f"  TABLE 1: Time to First Token (TTFT) — lower is better")
    print(f"{'='*80}")
    print(f"  {'Prompt':<10} {'llama TTFT':>12} {'MLX TTFT':>12} {'Δ':>8} {'Winner':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8} {'-'*12}")
    for pname in PROMPTS:
        l = streaming_results["llama.cpp"][pname]
        m = streaming_results["MLX"][pname]
        delta = ((m["avg_ttft"] - l["avg_ttft"]) / l["avg_ttft"]) * 100
        w = winner(l["avg_ttft"], m["avg_ttft"])
        print(f"  {pname:<10} {l['avg_ttft']:>8.2f}s ±{l['std_ttft']:.2f}  {m['avg_ttft']:>8.2f}s ±{m['std_ttft']:.2f}  {delta:>+6.1f}%  {w:>12}")

    # Table 2: TPS
    print(f"\n{'='*80}")
    print(f"  TABLE 2: Tokens Per Second (streaming) — higher is better")
    print(f"{'='*80}")
    print(f"  {'Prompt':<10} {'llama TPS':>12} {'MLX TPS':>12} {'Δ':>8} {'Winner':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8} {'-'*12}")
    for pname in PROMPTS:
        l = streaming_results["llama.cpp"][pname]
        m = streaming_results["MLX"][pname]
        delta = ((m["avg_tps"] - l["avg_tps"]) / l["avg_tps"]) * 100
        w = winner(l["avg_tps"], m["avg_tps"], lower_better=False)
        print(f"  {pname:<10} {l['avg_tps']:>8.1f} [{l['min_tps']:.0f}-{l['max_tps']:.0f}]  {m['avg_tps']:>8.1f} [{m['min_tps']:.0f}-{m['max_tps']:.0f}]  {delta:>+6.1f}%  {w:>12}")

    # Table 3: Non-streaming
    print(f"\n{'='*80}")
    print(f"  TABLE 3: Non-streaming (wall clock + server timings)")
    print(f"{'='*80}")
    print(f"  {'Prompt':<10} {'llama wall':>10} {'llama TPS':>10} {'llama prom':>10}  {'MLX wall':>10} {'MLX TPS':>10} {'Δ TPS':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*10} {'-'*10} {'-'*8}")
    for pname in PROMPTS:
        l = nonstream_results["llama.cpp"][pname]
        m = nonstream_results["MLX"][pname]
        delta = ((m["avg_tps"] - l["avg_tps"]) / l["avg_tps"]) * 100
        prom_l = f"{l['avg_prompt_tps']:.0f}t/s" if l["avg_prompt_tps"] > 0 else "n/a"
        print(f"  {pname:<10} {l['avg_wall']:>7.2f}s  {l['avg_tps']:>7.1f}t/s {prom_l:>10}  {m['avg_wall']:>7.2f}s  {m['avg_tps']:>7.1f}t/s {delta:>+6.1f}%")

    # Table 4: Stress test
    print(f"\n{'='*80}")
    print(f"  TABLE 4: Stress test — 20 rapid sequential requests")
    print(f"{'='*80}")
    print(f"  {'Metric':<12} {'llama.cpp':>15} {'MLX':>15} {'Winner':>12}")
    print(f"  {'-'*12} {'-'*15} {'-'*15} {'-'*12}")
    for metric, key, fmt, lower in [
        ("Avg time", "avg", ".2fs", True),
        ("Min time", "min", ".2fs", True),
        ("Max time", "max", ".2fs", True),
        ("Std dev", "std", ".3fs", True),
        ("P95 time", "p95", ".2fs", True),
        ("Errors", "errors", ".0f", True),
    ]:
        lv = stress_results["llama.cpp"][key]
        mv = stress_results["MLX"][key]
        w = winner(lv, mv, lower)
        print(f"  {metric:<12} {lv:>{len(fmt)-1}}{fmt}  {mv:>{len(fmt)-1}}{fmt}  {w:>12}")

    # Table 5: Summary
    print(f"\n{'='*80}")
    print(f"  TABLE 5: Overall Summary — Qwen3.5 4B (text-only, both engines)")
    print(f"{'='*80}")

    all_l_ttft = [r["avg_ttft"] for r in streaming_results["llama.cpp"].values()]
    all_m_ttft = [r["avg_ttft"] for r in streaming_results["MLX"].values()]
    all_l_tps = [r["avg_tps"] for r in streaming_results["llama.cpp"].values()]
    all_m_tps = [r["avg_tps"] for r in streaming_results["MLX"].values()]

    print(f"  {'Metric':<20} {'llama.cpp':>12} {'MLX':>12} {'Δ':>8} {'Winner':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*8} {'-'*12}")

    rows = [
        ("Avg TTFT", statistics.mean(all_l_ttft), statistics.mean(all_m_ttft), "s", True),
        ("Avg TPS (stream)", statistics.mean(all_l_tps), statistics.mean(all_m_tps), "t/s", False),
        ("Stress avg wall", stress_results["llama.cpp"]["avg"], stress_results["MLX"]["avg"], "s", True),
        ("Stress P95", stress_results["llama.cpp"]["p95"], stress_results["MLX"]["p95"], "s", True),
    ]
    for label, lv, mv, unit, lower in rows:
        delta = ((mv - lv) / lv) * 100
        w = winner(lv, mv, lower)
        print(f"  {label:<20} {lv:>8.2f}{unit}  {mv:>8.2f}{unit}  {delta:>+6.1f}%  {w:>12}")

    # ── Vision note ────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  Note: llama.cpp GGUF lacks Qwen3.5 vision encoder.")
    print(f"  Only MLX supports vision inference for this model.")
    print(f"  MLX vision TPS: ~27.3 tok/s (measured separately)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
