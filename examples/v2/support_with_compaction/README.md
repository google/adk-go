# support_with_compaction

A long-running customer-support session that auto-summarizes older
turns so the model context window doesn't blow up.

This is the realistic pattern adk-python's `App.events_compaction_config`
addresses: on a typical support thread the customer asks 8-12 follow-up
questions, by which point the raw event history is too large to send
to the LLM verbatim. The runner periodically replaces older turns with
a single "compacted" event whose `Actions.Compaction` carries a
synthesized summary; the contents-builder transparently folds the
summary in place of the subsumed events.

This demo uses an `EventsCompactionConfig` with:

- `CompactionInterval=2` — compact after every 2 new user invocations.
- `OverlapSize=1` — keep the previous turn for context continuity.
- A fake `EventsSummarizer` that produces a deterministic summary so
  you can read what the model would normally compose.

After 6 user turns you see 3 compaction events appended to the
session, each wrapping a window of older raw events.
