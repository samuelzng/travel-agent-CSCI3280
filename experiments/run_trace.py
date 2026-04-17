"""Run the agent with TRACE=1 logging enabled and capture output.

Usage:
    TRACE=1 python experiments/run_trace.py text <output_file> "<query>"
    TRACE=1 python experiments/run_trace.py image <output_file> <image_path> "<query>"
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["TRACE"] = "1"

import tools                     # noqa: E402
from agent import run_agent      # noqa: E402
from user_memory import save_preferences  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2

    mode = sys.argv[1]

    # Pre-save preferences so the agent goes straight to tool calls
    save_preferences(pace="moderate", interests=["history", "food"])
    tools._clear_cache()

    if mode == "text":
        query = sys.argv[3]
        print(f"[harness] mode=text query={query!r}")
        result = run_agent(query, conversation_history=[])
    elif mode == "image":
        image_path = Path(sys.argv[3])
        query = sys.argv[4]
        if not image_path.exists():
            print(f"[FATAL] image not found: {image_path}", file=sys.stderr)
            return 2
        img_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower().lstrip(".")
        mime = f"image/{ {'jpg':'jpeg'}.get(suffix, suffix)}"
        print(f"[harness] mode=image path={image_path} mime={mime} query={query!r}")
        result = run_agent(query, conversation_history=[],
                            image_bytes=img_bytes, image_mime=mime)
    else:
        print(f"unknown mode: {mode}", file=sys.stderr)
        return 2

    has_itin = bool(result.get("itinerary"))
    text_preview = (result.get("text") or "")[:200].replace("\n", " ")
    print(f"[harness] itinerary={'yes' if has_itin else 'no'} text[:200]={text_preview!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
