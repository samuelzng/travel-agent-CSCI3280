"""Tool registry — exports all tool functions for the agent."""

import hashlib
import json
import logging
import os
import time
import unicodedata

from tools.places import search_places
from tools.weather import get_weather
from user_memory import save_preferences, save_memory

logger = logging.getLogger(__name__)

_TRACE = os.getenv("TRACE") == "1"
# Optional list that the bench/trace harness can populate to capture per-call timings.
# Each entry: {"fn": str, "kwargs": dict, "hit": bool, "duration": float}
TRACE_EVENTS: list[dict] = []

# ── Simple in-memory TTL cache ────────────────────────────────────────────────
# Weather and places data don't change within a session.  Cache avoids
# duplicate API calls when the LLM re-requests the same data.

_cache: dict[str, tuple[float, dict]] = {}   # key → (expires_at, result)

# Per-tool TTL: weather changes fast (thunderstorms), places are stable
_TOOL_TTL = {
    "get_weather": 1800,     # 30 min — Open-Meteo updates hourly
    "search_places": 21600,  # 6 hours — POI data rarely changes
}
_DEFAULT_TTL = 3600


def _normalize_val(v: str) -> str:
    """Normalize string values for cache key: lowercase, strip, NFKC unicode."""
    return unicodedata.normalize("NFKC", v.strip().lower())


def _cache_key(fn_name: str, kwargs: dict) -> str:
    """Deterministic cache key from function name + normalized args."""
    normalized = {
        k: _normalize_val(v) if isinstance(v, str) else v
        for k, v in kwargs.items()
    }
    raw = json.dumps({"fn": fn_name, **normalized}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _cached(fn, fn_name: str):
    """Wrap a tool function with TTL cache."""
    ttl = _TOOL_TTL.get(fn_name, _DEFAULT_TTL)

    def wrapper(**kwargs):
        key = _cache_key(fn_name, kwargs)
        now = time.time()
        t0 = time.perf_counter()
        if key in _cache:
            expires, result = _cache[key]
            if now < expires:
                dt = time.perf_counter() - t0
                logger.info("Cache hit: %s(%s) in %.4fs", fn_name, kwargs, dt)
                if _TRACE:
                    TRACE_EVENTS.append({"fn": fn_name, "kwargs": dict(kwargs),
                                         "hit": True, "duration": dt})
                return result
        result = fn(**kwargs)
        dt = time.perf_counter() - t0
        if _TRACE:
            TRACE_EVENTS.append({"fn": fn_name, "kwargs": dict(kwargs),
                                 "hit": False, "duration": dt})
            logger.info("Cache miss: %s(%s) in %.4fs", fn_name, kwargs, dt)
        # Only cache successful results (no error key)
        if isinstance(result, dict) and "error" not in result:
            _cache[key] = (now + ttl, result)
        return result
    return wrapper


TOOL_REGISTRY = {
    "search_places": _cached(search_places, "search_places"),
    "get_weather": _cached(get_weather, "get_weather"),
    "save_user_preferences": save_preferences,
    "save_memory": save_memory,
}


def _clear_cache() -> None:
    """Empty the in-memory TTL cache (used by bench/tests)."""
    _cache.clear()


__all__ = ["search_places", "get_weather", "TOOL_REGISTRY", "TRACE_EVENTS", "_clear_cache"]
