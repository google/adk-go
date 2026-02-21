# Potential Issues in Live Mode Error Handling

This document outlines potential issues identified during the analysis of [internal/llminternal/base_flow_live.go](internal/llminternal/base_flow_live.go) and [model/gemini/gemini_live.go](model/gemini/gemini_live.go) regarding error propagation and connection stability.

### 1. Unsafe Concurrent `yield` in `RunLive` (FIXED)

- **Issue:** Go's `iter.Seq2` (and iterators in general) are not designed to have the `yield` function called from a different goroutine than the one performing the iteration.
- **Fix:** Refactored `RunLive` to use a centralized `results` channel. Background goroutines (`runSender`, `runReceiver`) now send events and errors to this channel, and the main `RunLive` loop consumes the channel to safely call `yield` from the correct goroutine.

### 2. Infinite Busy Loop in `receiveFromLiveModel` (FIXED)

- **Issue:** Using `break` inside a `select` only exited the `select`, not the surrounding `for` loop, causing an infinite busy loop (100% CPU) when channels were closed.
- **Fix:** Consolidated the receiver logic into `runReceiver` and ensured the loop returns immediately when the `resps` or `errs` channels are closed or on context cancellation.

### 3. Lack of Synchronization between Sender and Receiver (FIXED)

- **Issue:** Independent sender and receiver loops could lead to "zombie" goroutines if one failed without signaling the other.
- **Fix:** Implemented a unified lifecycle management using a shared cancelable context (`context.WithCancel(ctx)`) and a `sync.WaitGroup`. A failure in either the sender or receiver triggers a cancellation that stops all background tasks for that connection.

### 4. Retry Loop Ambiguity in `RunLive`

- **Issue:** The outer reconnection loop indiscriminately attempts to reconnect whenever an error occurs. It does not distinguish between a normal session closure (EOF) and a fatal error (e.g., authentication failure).
- **Status:** Partially improved by the refactoring, but explicit logic to distinguish retryable vs. fatal errors is still needed.

### 5. Swallowed Errors in `gemini_live.go` `process` Goroutine (FIXED)

- **Issue:** The `process` function had no independent error channel, meaning mapping errors were lost.
- **Fix:** (Fixed in previous session) Updated `process` to return an error channel and merged errors in `Receive`.

### 6. Resource Leak on Early Return (FIXED)

- **Issue:** If `RunLive` returned early, background goroutines could remain active if the context wasn't cancelled.
- **Fix:** Ensured `cancel()` is called and the connection is closed in all exit paths of the `RunLive` reconnection loop.

### 7. Resumption Handle Reset Logic

- **Issue:** If a connection fails repeatedly due to an invalid `LiveSessionResumptionHandle`, there's no logic to "clear" the handle and fallback to a fresh session.
- **Status:** Open.
