# support_with_compaction

Real Gemini-backed customer-support agent whose long session auto-
summarizes older turns via `app.EventsCompactionConfig` and a
Gemini-driven `app.LlmEventSummarizer`.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/support_with_compaction/
```

This example uses an interactive console loop directly (rather than the
standard launcher) so it can wire `runner.Config.App` and the
compaction config in one place. After every 3 user turns the runner
appends a compaction event whose `Actions.Compaction.CompactedContent`
holds a Gemini-generated summary; the contents-builder folds the
summary in place of the subsumed raw events on the next LLM call,
keeping the prompt size flat as the conversation grows.

To go from this pattern to a launcher-driven agent, lift the
`runner.Config.App` field onto `cmd/launcher.Config` and wire the same
`adkapp.App` value through the standard `launcher.Execute` path.
