# Bolt Architectural Performance Ledger

## 2026-11-09 - Zero-allocation fast path for RequestConfirmationRequestProcessor

**Learning:**
`RequestConfirmationRequestProcessor` runs as part of `DefaultRequestProcessors` on every LLM agent turn. On standard conversation turns without human-in-the-loop tool confirmations, it unconditionally allocated a `toolsmap` map (`make(map[string]tool.Tool)`), invoked closure-based iterators across all session events (`ctx.Session().Events().All()`), dynamically grew event slices via `append`, and called `utils.FunctionResponses` allocating slice pointers. This added 912 B/op and 12 allocations per turn even when no tool confirmations were present.

**Action:**
Fast-path empty sessions (`sess.Events().Len() == 0`), locate the latest user event using reverse index traversal (`sessEvents.At(k)`), and inspect `Content.Parts` directly for `toolconfirmation.FunctionCallName`. If no confirmation response is present, return immediately (0 B/op, 0 allocs/op, ~41x faster). Defer `toolsmap` allocation until confirmed tool calls are actually present.
