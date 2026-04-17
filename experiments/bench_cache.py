"""Benchmark: cold vs warm cache for a realistic trip-planning query.

Runs one query twice through run_agent in the same process:
  Run 1 — cache cleared first (all API calls hit the network)
  Run 2 — same query, cache warm (search_places/get_weather served from cache)

Reports per-tool wall-clock times and overall speedup.
Requires TRACE=1 so the cache wrapper records per-call timings.

Usage:
    TRACE=1 python experiments/bench_cache.py
"""

import os
import sys
import time
from pathlib import Path

# Make the project root importable regardless of CWD
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force TRACE on so the cache wrapper records events before modules are imported
os.environ["TRACE"] = "1"

import tools                           # noqa: E402  (sets up TRACE_EVENTS, _clear_cache)
from agent import run_agent            # noqa: E402
from user_memory import save_preferences  # noqa: E402

QUERY = "Plan a 2-day trip to Kyoto focused on temples and local food"


def _summarize(events: list[dict]) -> dict:
    """Group events by (fn, kwargs) and collapse to totals per tool."""
    per_tool: dict[str, list[dict]] = {}
    for ev in events:
        per_tool.setdefault(ev["fn"], []).append(ev)
    return per_tool


def _run_once(label: str) -> tuple[float, list[dict]]:
    """Run the agent once and capture timings + trace events."""
    # Reset the TRACE_EVENTS list for this run
    tools.TRACE_EVENTS.clear()
    print(f"\n===== {label} =====", flush=True)
    t0 = time.perf_counter()
    try:
        result = run_agent(QUERY, conversation_history=[])
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[ERROR] run_agent raised: {e!r} after {dt:.2f}s", flush=True)
        raise
    dt = time.perf_counter() - t0
    has_itin = bool(result.get("itinerary"))
    print(f"[{label}] total wall-clock: {dt:.2f}s  itinerary={'yes' if has_itin else 'no'}",
          flush=True)
    events = list(tools.TRACE_EVENTS)
    # Per-call breakdown
    for ev in events:
        kw = ev["kwargs"]
        tag = "HIT" if ev["hit"] else "MISS"
        print(f"  {tag:4} {ev['fn']}({kw})  dt={ev['duration']:.4f}s", flush=True)
    return dt, events


def main() -> int:
    # Preflight: keys must be present
    try:
        from config import GEMINI_API_KEY, TAVILY_API_KEY
    except Exception as e:
        print(f"[FATAL] could not import config: {e}", file=sys.stderr)
        return 2
    if not GEMINI_API_KEY or not TAVILY_API_KEY:
        print("[FATAL] GEMINI_API_KEY or TAVILY_API_KEY missing from .env", file=sys.stderr)
        return 2

    print(f"Query: {QUERY}")

    # Pre-save prefs so the agent goes straight into planning (no preference
    # gathering round — we want to measure tool caching, not pref dialogs).
    save_preferences(pace="moderate", interests=["history", "food"])

    # Run 1 — cold cache
    tools._clear_cache()
    cold_total, cold_events = _run_once("Run 1 — cold cache")

    # Run 2 — warm cache (do NOT clear)
    warm_total, warm_events = _run_once("Run 2 — warm cache")

    # Aggregate totals per tool
    cold_by = _summarize(cold_events)
    warm_by = _summarize(warm_events)

    all_fns = sorted(set(cold_by) | set(warm_by))

    print("\n=== Summary (markdown) ===")
    print("| Tool | Run 1 (cold) | Run 2 (warm) | Speedup |")
    print("|------|--------------|--------------|---------|")
    for fn in all_fns:
        c = sum(ev["duration"] for ev in cold_by.get(fn, []))
        w = sum(ev["duration"] for ev in warm_by.get(fn, []))
        c_n = len(cold_by.get(fn, []))
        w_n = len(warm_by.get(fn, []))
        speedup = f"{(c / w):.1f}x" if w > 0 else "—"
        print(f"| {fn} ({c_n} cold / {w_n} warm calls) | {c:.4f}s | {w:.4f}s | {speedup} |")

    tot_speedup = f"{(cold_total / warm_total):.2f}x" if warm_total > 0 else "—"
    print(f"| **TOTAL agent wall-clock** | **{cold_total:.2f}s** | **{warm_total:.2f}s** | **{tot_speedup}** |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
