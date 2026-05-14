# Session pause / resume protocol

## What I can and can't do

**I cannot read my own session-wide token quota** from inside the model — there
is no API exposing it. **But I CAN estimate token usage** of every text I send
or receive using `tiktoken`, accumulate it in a JSON file on disk, and trigger
the pause when the running total crosses 98 % of a configurable budget.

Set the budget once (default 150 000, override with `CLAUDE_TOKEN_BUDGET` env
var). After every substantial turn I run:

```bash
python src/session_pause.py tokens "<the text I just generated>"
```

The script:

1. Counts tokens via `tiktoken` (`cl100k_base` encoding — close enough to Claude's tokeniser for budgeting).
2. Adds to the running total in `outputs/session_state.json`.
3. Returns `should_pause: true` if total/budget ≥ 0.98.

When that flag flips true, I follow the pause protocol below.

I also still watch the heuristic signals (very long conversation, large
attachments, harness context-pressure warnings, trimmed responses) as a
backstop in case the count drifts.

## Pause protocol (4–5 hour wait for the rolling-window quota to reset)

1. **Snapshot state to disk** so a resumed session doesn't need the chat history:
   ```bash
   python src/session_pause.py log "<short summary of where we are>"
   ```
   Always log: which run is in flight, which `out_dir`, last metric values,
   and what the next concrete action is.

2. **Schedule a wakeup chain.** `ScheduleWakeup` is capped at 1 hour, so a
   4–5 hour pause needs 4–5 chained wakeups. Each chained wakeup's prompt is
   minimal: "if quota still tight, schedule the next; otherwise resume the
   current run from the session log." Keeps each turn cheap so we don't burn
   more tokens during the pause itself.

3. **The first wakeup's prompt** should be something like:
   > Read `outputs/session_state.json`. If the active run is still alive
   > (`ps aux | grep python`), do nothing and reschedule a 1-hour wakeup
   > with the same prompt. If 4 wakeups have fired since the pause began,
   > resume normally — read the session log, check `outputs/<active>/log`
   > for the latest progress, and continue.

4. **Background training is unaffected.** The Python process is independent
   of the Claude session; it keeps streaming files even if Claude is paused.
   The pause only pauses my *response generation*, not the model training.

## Resume protocol

When a wakeup fires after the pause:

1. `python src/session_pause.py status` — read the last 20 events
2. Check the active run's log + history.json for the latest progress
3. If the run finished, run the standard `_evaluate_model.py` and commit
4. Update the session log with `python src/session_pause.py log "resumed"`
5. Continue normal work

## Why a Python helper instead of just Markdown notes

So the state is **machine-readable** across pauses — a resumed Claude session
can open `outputs/session_state.json` and reconstruct what's going on without
parsing the chat history (which the new session may not even have).
