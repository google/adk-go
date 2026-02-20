# Potential Issues in Live Mode Error Handling

This document outlines potential issues identified during the analysis of [internal/llminternal/base_flow_live.go](internal/llminternal/base_flow_live.go) and [model/gemini/gemini_live.go](model/gemini/gemini_live.go) regarding error propagation and connection stability.

### 1. Unsafe Concurrent `yield` in `RunLive`

In [internal/llminternal/base_flow_live.go:104](internal/llminternal/base_flow_live.go#L104), the `RunLive` function starts a background goroutine to handle sending requests to the Gemini model. This goroutine calls `yield` directly if `conn.Send` fails at [line 119](internal/llminternal/base_flow_live.go#L119), [line 133](internal/llminternal/base_flow_live.go#L133), [line 162](internal/llminternal/base_flow_live.go#L162), and [line 170](internal/llminternal/base_flow_live.go#L170).

- **Issue:** Go's `iter.Seq2` (and iterators in general) are not designed to have the `yield` function called from a different goroutine than the one performing the iteration. This is a significant thread-safety violation that can lead to race conditions, panics, or the error being silently ignored if the main iteration loop has already moved on or returned.

### 2. Infinite Busy Loop in `receiveFromLiveModel`

In [internal/llminternal/base_flow_live.go:268](internal/llminternal/base_flow_live.go#L268), the `receiveFromLiveModel` function contains a `select` statement within a `for` loop to process responses and errors from the connection.

- **Issue:** When either the `resps` or `errs` channel is closed (`!ok`), the code calls `break` at [line 272](internal/llminternal/base_flow_live.go#L272) and [line 336](internal/llminternal/base_flow_live.go#L336). In Go, a `break` inside a `select` only exits the `select` block, not the surrounding `for` loop. Consequently, if a channel is closed (which happens when the connection drops), the function will enter an infinite busy loop, repeatedly hitting the closed channel case and consuming 100% CPU until the context is cancelled. This effectively hangs the agent instead of propagating the closure.

### 3. Lack of Synchronization between Sender and Receiver

The sender goroutine at [base_flow_live.go:104](internal/llminternal/base_flow_live.go#L104) and the receiver loop in `RunLive` operate mostly independently.

- **Issue:** If the receiver loop (which yields model events) encounter a fatal error and exits, the sender goroutine remains active until the context is cancelled or a new item is pushed to the queue. Conversely, if the sender fails, it "dies" (or attempts an unsafe yield) without explicitly signaling the receiver to stop. This can lead to "zombie" goroutines attempting to use a connection that is either closed or in an error state.

### 4. Retry Loop Ambiguity in `RunLive`

`RunLive` uses an outer `for` loop at [internal/llminternal/base_flow_live.go:59](internal/llminternal/base_flow_live.go#L59) to handle reconnection logic, potentially using `LiveSessionResumptionHandle`.

- **Issue:** The loop indiscriminately attempts to reconnect whenever `receiveFromLiveModel` exits. It does not distinguish between a normal session closure (EOF) and a fatal error that should not be retried (e.g., authentication failure or invalid configuration). This could lead to unnecessary reconnection attempts or masked errors.

### 5. Swallowed Errors in `gemini_live.go` `process` Goroutine

In [model/gemini/gemini_live.go:114](model/gemini/gemini_live.go#L114), the `process` function runs in a goroutine to transform `genai.LiveServerMessage` into `model.LLMResponse`.

- **Issue:** While it correctly handles the closing of the input channel, it has no independent error channel. If an error were to occur during the processing/mapping logic (e.g., unexpected message types or internal state inconsistencies), there is no way for this specific goroutine to propagate that error back to the `Receive` caller.

### 6. Resource Leak on Early Return

In [internal/llminternal/base_flow_live.go](internal/llminternal/base_flow_live.go), if `RunLive` returns early during an error (e.g., in the `preprocess` step or if `yield` returns `false`), the connection is closed via `defer conn.Close()`.

- **Issue:** However, if the sender goroutine is still waiting on a channel or blocked on a send, and the context isn't cancelled, it may stay alive longer than the `RunLive` call itself, potentially leading to leaks if many connections are cycled.

### 7. Resumption Handle Reset Logic

If a connection fails repeatedly due to an expired or invalid `LiveSessionResumptionHandle`, the current logic in `RunLive` at [base_flow_live.go:61](internal/llminternal/base_flow_live.go#L61) doesn't seem to have a clear way to "clear" the handle and fall back to a fresh connection. It will simply keep attempting to reconnect using the same (potentially bad) handle provided by the context.
