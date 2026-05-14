"""Session pause/resume helper.

I can't introspect my own token count from inside a Claude session. So we
implement an honest fallback: I judge when context is getting heavy, write
the current state to disk, and schedule a wakeup chain to resume after a few
hours when the rolling-window quota has reset.

The state file (outputs/session_state.json) lets a resumed session pick up
where it left off without needing the chat history to reconstruct context.
Always check this file first on resume.

Usage from a Python REPL or another script (not from inside the model directly):
    python src/session_pause.py log "v8 stream at 22/872, val_mmse 0.054"
    python src/session_pause.py status
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(ROOT, "outputs", "session_state.json")

# heuristic budgets — Claude Pro rolling-window quota guidance.
# Adjust as you learn your actual limit; default conservative.
DEFAULT_BUDGET = int(os.environ.get("CLAUDE_TOKEN_BUDGET", "150000"))
PAUSE_THRESHOLD_FRAC = 0.98     # trigger pause at 98% of budget

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))
except Exception:
    def count_tokens(text: str) -> int:
        # ~4 chars per token is the widely-used rough heuristic
        return max(1, len(text or "") // 4)


def _load() -> dict:
    if os.path.exists(STATE_PATH):
        s = json.load(open(STATE_PATH))
        s.setdefault("token_total", 0)
        s.setdefault("token_budget", DEFAULT_BUDGET)
        return s
    return {"events": [], "active_run": None,
            "token_total": 0, "token_budget": DEFAULT_BUDGET}


def _save(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as fh:
        json.dump(state, fh, indent=2)


def log(message: str, kind: str = "note") -> None:
    """Append a timestamped event to the session log."""
    state = _load()
    state["events"].append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "message": message,
    })
    _save(state)


def set_active_run(out_dir: str, model: str, command: str) -> None:
    """Record which run is currently in flight, so a resumed session knows."""
    state = _load()
    state["active_run"] = {
        "out_dir": out_dir, "model": model, "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save(state)


def add_tokens(text: str, kind: str = "turn") -> dict:
    """Count tokens in `text` and add to the running total. Returns
    {tokens_added, total, budget, frac, should_pause}."""
    n = count_tokens(text)
    state = _load()
    state["token_total"] += n
    state["events"].append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "tokens", "message": f"+{n} ({kind}); total={state['token_total']}",
    })
    _save(state)
    frac = state["token_total"] / max(state["token_budget"], 1)
    return {"tokens_added": n, "total": state["token_total"],
            "budget": state["token_budget"], "frac": frac,
            "should_pause": frac >= PAUSE_THRESHOLD_FRAC}


def reset_tokens() -> None:
    """Call after a quota-window reset (e.g. resuming after a 5-hour pause)."""
    state = _load()
    state["token_total"] = 0
    state["events"].append({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "reset", "message": "token counter reset",
    })
    _save(state)


def status() -> None:
    state = _load()
    print("active run:", json.dumps(state.get("active_run"), indent=2) if state.get("active_run") else "none")
    frac = state["token_total"] / max(state["token_budget"], 1)
    pct = frac * 100
    flag = "PAUSE NOW" if frac >= PAUSE_THRESHOLD_FRAC else "ok"
    print(f"\nTOKENS: {state['token_total']:,} / {state['token_budget']:,} ({pct:5.1f}%)  [{flag}]")
    print(f"\n{len(state['events'])} events (last 10):")
    for e in state["events"][-10:]:
        print(f"  {e['ts']}  [{e['kind']:6s}]  {e['message']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("log"); pl.add_argument("message"); pl.add_argument("--kind", default="note")
    pa = sub.add_parser("active"); pa.add_argument("out_dir"); pa.add_argument("model"); pa.add_argument("command")
    sub.add_parser("status")
    pt = sub.add_parser("tokens"); pt.add_argument("text"); pt.add_argument("--kind", default="turn")
    sub.add_parser("reset-tokens")
    args = p.parse_args()
    if args.cmd == "log": log(args.message, args.kind)
    elif args.cmd == "active": set_active_run(args.out_dir, args.model, args.command)
    elif args.cmd == "status": status()
    elif args.cmd == "tokens":
        r = add_tokens(args.text, args.kind)
        print(json.dumps(r, indent=2))
    elif args.cmd == "reset-tokens": reset_tokens()


if __name__ == "__main__":
    main()
