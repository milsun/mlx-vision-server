#!/usr/bin/env python3
"""Vision benchmark: llama.cpp GGUF vs MLX — multi-run with averages."""

import argparse
import base64
import json
import os
import statistics
import sys
import time
import urllib.request

LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080/v1/chat/completions")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8001/v1/chat/completions")
DEFAULT_PROMPT = "Describe this image in detail. What shapes and colors do you see?"
MAX_TOKENS = 512
TEMPERATURE = 0.0


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        mime = f"image/{ext}" if ext in ("jpg","jpeg","png","webp","gif") else "image/png"
        return f"data:{mime};base64,{b64}"


def run_one(api_url: str, image_b64: str, prompt: str, max_tokens: int):
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_b64}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        api_url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall = time.perf_counter() - t0

    usage = body.get("usage", {})
    timings = body.get("timings", {})
    msg = body["choices"][0]["message"] if body.get("choices") else {}
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""

    return {
        "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "server_prompt_ms": timings.get("prompt_ms", 0),
        "server_gen_ms": timings.get("predicted_ms", 0),
        "server_prompt_tps": round(timings.get("prompt_per_second", 0), 1),
        "server_gen_tps": round(timings.get("predicted_per_second", 0), 1),
        "reasoning_len": len(reasoning),
        "output_len": len(content),
    }


def generate_test_image(path: str):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 200], fill="red", outline="black", width=3)
    draw.ellipse([300, 50, 500, 250], fill="blue", outline="black", width=3)
    draw.polygon([(600, 250), (700, 50), (800, 250)], fill="green", outline="black")
    draw.rectangle([50, 300, 350, 450], fill="yellow", outline="black", width=3)
    draw.polygon([(450, 300), (650, 250), (750, 400), (550, 500), (400, 450)],
                 fill="magenta", outline="black", width=3)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((500, 520), "Benchmark Image", fill="black", font=font)
    img.save(path, "PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-url", default=LLAMA_URL)
    parser.add_argument("--mlx-url", default=MLX_URL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("-n", type=int, default=5, help="Number of runs per engine")
    args = parser.parse_args()

    image_path = "/tmp/bench_test_image.png"
    if not os.path.exists(image_path):
        generate_test_image(image_path)
    image_b64 = encode_image(image_path)

    print(f"\n{'#'*72}")
    print(f"  VISION BENCHMARK — {args.n} runs per engine")
    print(f"  Model: Gemma 4 E4B | max_tokens={args.max_tokens} | temp={TEMPERATURE}")
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"{'#'*72}")

    engines = {
        "llama.cpp": {"url": args.llama_url, "results": [], "name": "llama.cpp GGUF"},
        "MLX":       {"url": args.mlx_url, "results": [], "name": "MLX"},
    }

    # ── Warm-up (1 run each) ─────────────────────────────────────
    print("\n  [Warm-up]")
    for label in ["llama.cpp", "MLX"]:
        r = run_one(engines[label]["url"], image_b64, args.prompt, args.max_tokens)
        print(f"    {label}: {r['wall_s']:.2f}s  "
              f"(prompt={r['prompt_tokens']}t, gen={r['completion_tokens']}t"
              f"{', think=' + str(r['reasoning_len']) + 'ch' if r['reasoning_len'] else ''})")

    # ── Benchmark runs (alternating) ─────────────────────────────
    for run_i in range(args.n):
        for label in ["llama.cpp", "MLX"]:
            print(f"  [Run {run_i+1}/{args.n} {label}]", end="", flush=True)
            r = run_one(engines[label]["url"], image_b64, args.prompt, args.max_tokens)
            engines[label]["results"].append(r)
            print(f"  wall={r['wall_s']:.2f}s  "
                  f"prompt={r['prompt_tokens']}t  gen={r['completion_tokens']}t  "
                  f"{'think=' + str(r['reasoning_len']) + 'ch  ' if r['reasoning_len'] else ''}"
                  f"output={r['output_len']}ch")

    # ── Aggregate ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  RESULTS: {args.n} runs × Gemma 4 E4B (reasoning=OFF)")
    print(f"{'='*80}")

    def fmt(vals, fmt_spec=".2f", unit=""):
        if not vals:
            return "n/a"
        return f"{statistics.mean(vals):{fmt_spec}}{unit}"

    def rng(vals, fmt_spec=".2f"):
        if not vals or len(vals) < 2:
            return ""
        return f"  [{min(vals):{fmt_spec}}..{max(vals):{fmt_spec}}]"

    meta_rows = [
        ("", "llama.cpp GGUF", "MLX", "Delta", "Winner"),
        ("——",  "——", "——", "——", "——"),
    ]

    lr = engines["llama.cpp"]["results"]
    mr = engines["MLX"]["results"]

    wall_l = [r["wall_s"] for r in lr]
    wall_m = [r["wall_s"] for r in mr]
    gen_l = [r["completion_tokens"] for r in lr]
    gen_m = [r["completion_tokens"] for r in mr]
    out_l = [r["output_len"] for r in lr]
    out_m = [r["output_len"] for r in mr]
    sps_l = [r["server_gen_tps"] for r in lr if r["server_gen_tps"]]
    ppt_l = [r["server_prompt_tps"] for r in lr if r["server_prompt_tps"]]
    # effective tok/s (wall clock based, includes prompt processing)
    eff_l = [r["completion_tokens"] / r["wall_s"] for r in lr]
    eff_m = [r["completion_tokens"] / r["wall_s"] for r in mr]

    avg_wl = statistics.mean(wall_l)
    avg_wm = statistics.mean(wall_m)
    delta_pct = ((avg_wm - avg_wl) / avg_wl) * 100

    data_rows = [
        ("Wall clock",  wall_l, wall_m, "s", True, ".2f"),
        ("Completion tokens", gen_l, gen_m, "", False, ".0f"),
        ("Output chars", out_l, out_m, "", False, ".0f"),
        ("Effective tok/s", eff_l, eff_m, "", False, ".1f"),
    ]

    if sps_l:
        data_rows.append(("Gen speed (server)", sps_l, [], "t/s", False, ".1f"))
    if ppt_l:
        data_rows.append(("Prompt speed (server)", ppt_l, [], "t/s", False, ".1f"))

    # Print header
    print(f"  {'Metric':<26} {'llama.cpp':>14} {'MLX':>14} {'Δ%':>8} {'Winner':>10}")
    print(f"  {'-'*26} {'-'*14} {'-'*14} {'-'*8} {'-'*10}")

    for label, lvals, mvals, unit, lower_better, fmt_s in data_rows:
        if not lvals:
            continue
        avg_l = statistics.mean(lvals)
        avg_m = statistics.mean(mvals) if mvals else None

        if avg_m is not None:
            delta = ((avg_m - avg_l) / avg_l) * 100
            if lower_better:
                winner = "MLX" if delta < 0 else "llama.cpp" if delta > 0 else "tie"
            else:
                winner = "llama.cpp" if delta < 0 else "MLX" if delta > 0 else "tie"
        else:
            delta = None
            winner = "n/a"

        val_l = f"{avg_l:{fmt_s}}{unit}"
        val_m = f"{avg_m:{fmt_s}}{unit}" if avg_m is not None else "n/a"
        d = f"{delta:+.1f}%" if delta is not None else "—"

        print(f"  {label:<26} {val_l:>14} {val_m:>14} {d:>8} {winner:>10}")

    # ── Per-run detail ───────────────────────────────────────────
    print(f"\n{'─'*80}")
    header_run = f"{'Run#':>5} {'llama wall':>11} {'llama gen':>10} {'llama out':>10}  {'MLX wall':>10} {'MLX gen':>9} {'MLX out':>9}"
    print(f"  {header_run}")
    print(f"  {'-'*5} {'-'*11} {'-'*10} {'-'*10}  {'-'*10} {'-'*9} {'-'*9}")
    for i in range(args.n):
        print(f"  {i+1:>5} {lr[i]['wall_s']:>8.2f}s {lr[i]['completion_tokens']:>7}t {lr[i]['output_len']:>7}ch  "
              f"{mr[i]['wall_s']:>7.2f}s {mr[i]['completion_tokens']:>6}t {mr[i]['output_len']:>6}ch")

    # ── Speed delta summary ──────────────────────────────────────
    faster = "llama.cpp" if avg_wl < avg_wm else "MLX"
    pct = abs(delta_pct)
    print(f"\n  {'='*60}")
    if abs(delta_pct) < 3:
        print(f"  VERDICT: Statistical TIE — both deliver ~{statistics.mean(eff_l + eff_m):.1f} tok/s")
    elif faster == "llama.cpp":
        print(f"  WINNER: llama.cpp is {pct:.1f}% faster ({avg_wl:.2f}s vs {avg_wm:.2f}s)")
    else:
        print(f"  WINNER: MLX is {pct:.1f}% faster ({avg_wm:.2f}s vs {avg_wl:.2f}s)")
    print(f"  {'='*60}")
    print()


if __name__ == "__main__":
    main()
