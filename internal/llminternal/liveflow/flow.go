// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package liveflow

import (
	"context"
	"fmt"
	"iter"
	"sync"
	"time"

	semconv "go.opentelemetry.io/otel/semconv/v1.36.0"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/telemetry"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// LiveFlow is the core engine for bidirectional live streaming.
type LiveFlow struct {
	Model                model.LLM
	Tools                []tool.Tool
	BeforeToolCallbacks  []llminternal.BeforeToolCallback
	AfterToolCallbacks   []llminternal.AfterToolCallback
	OnToolErrorCallbacks []llminternal.OnToolErrorCallback
	CoalesceWindow       time.Duration
}

// modelName returns the model's name for span labels, falling back to
// "unknown" when the model is unset (e.g., bare-LiveFlow tests) or returns
// an empty string. Avoids producing span names with trailing whitespace
// like "generate_content " that would break filtering by name.
func (lf *LiveFlow) modelName() string {
	if lf.Model == nil {
		return "unknown"
	}
	if name := lf.Model.Name(); name != "" {
		return name
	}
	return "unknown"
}

type eventOrError struct {
	event *session.Event
	err   error
}

// sendEvent sends a message to eventCh, aborting if ctx is cancelled.
// This prevents goroutines from blocking on a full channel after shutdown.
func sendEvent(ctx context.Context, eventCh chan<- eventOrError, msg eventOrError) {
	select {
	case eventCh <- msg:
	case <-ctx.Done():
	}
}

// RunLive runs the live flow, returning an iterator of events.
// It accepts a ConnectFn instead of a raw connection so it can reconnect
// on GoAway signals using session resumption handles.
func (lf *LiveFlow) RunLive(
	ctx agent.InvocationContext,
	connectFn ConnectFn,
	queue *agent.LiveRequestQueue,
) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// live_session wraps the entire iterator lifetime, including all
		// reconnect attempts. Reconnects appear as sibling generate_content
		// children — the diagnostic shape needed for "how often did this
		// session reconnect?".
		spanCtx, span := telemetry.StartLiveSessionSpan(ctx, telemetry.StartLiveSessionSpanParams{
			AgentName:    ctx.Agent().Name(),
			SessionID:    ctx.Session().ID(),
			InvocationID: ctx.InvocationID(),
		})
		ctx = ctx.WithContext(spanCtx)
		yield, endSpan := telemetry.WrapYield(span, yield, func(s trace.Span, ev *session.Event, err error) {
			telemetry.TraceLiveSessionResult(s, telemetry.TraceLiveSessionResultParams{
				ResponseEvent: ev,
				Error:         err,
			})
		})
		defer endSpan()

		toolsFuncMap := buildToolsFuncMap(lf.Tools)
		ts := &liveTimingState{}
		attempt := 0

		// Outer loop owns the reconnect lifecycle; it terminates only via an
		// explicit return (clean shutdown, no GoAway, fatal error, or context
		// cancel). Each outer iteration owns a fresh inner connect-retry budget,
		// so a successful connect cannot accidentally weaken the retry budget
		// for the next reconnect cycle.
		for {
			// Capture the resumption handle once per outer iteration so the
			// receiver-loop's mutations don't desync our gating decisions
			// (e.g., whether to send history) from what connectFn observed.
			handle := ctx.LiveSessionResumptionHandle()

			conn, err := connectWithRetry(ctx, handle, connectFn)
			if err != nil {
				yield(nil, err)
				return
			}

			shouldReconnect, terminated := lf.runSession(ctx, conn, queue, toolsFuncMap, ts, handle, attempt, yield)
			if terminated || !shouldReconnect {
				return
			}
			attempt++

			// Brief backoff before reconnecting; bail promptly on cancel so
			// callers see context.Canceled instead of waiting out the timer.
			// Yield the cancel error so consumers don't observe a silent
			// iterator close that's indistinguishable from a clean exit.
			if !sleepOrCancel(ctx, reconnectGoAwaySleep) {
				yield(nil, ctx.Err())
				return
			}
		}
	}
}

// runSession runs a single live session over conn. It returns whether the
// caller should attempt to reconnect (GoAway received) and whether the
// iterator has been terminated by the consumer (yield returned false) or
// by a fatal error during history replay.
//
// handle is the resumption handle that was passed to connectFn for this
// session. When non-empty, the server is replaying conversation history
// from the resumed session — re-sending it from our side would duplicate
// turns. When empty (initial connect, or after a non-resumable update
// cleared a stale handle), we send the local history so the model has
// the full context.
func (lf *LiveFlow) runSession(
	ctx agent.InvocationContext,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	toolsFuncMap map[string]toolinternal.FunctionTool,
	ts *liveTimingState,
	handle string,
	attempt int,
	yield func(*session.Event, error) bool,
) (shouldReconnect, terminated bool) {
	cancelCtx, cancel := context.WithCancel(ctx)

	// generate_content covers one WebSocket attempt. On GoAway-driven
	// reconnects, the outer RunLive loop calls runSession again, producing
	// sibling generate_content spans under the same live_session.
	spanCtx, span := telemetry.StartGenerateContentSpan(cancelCtx, telemetry.StartGenerateContentSpanParams{
		ModelName:    lf.modelName(),
		InvocationID: ctx.InvocationID(),
	})
	span.SetAttributes(
		semconv.GenAIConversationID(ctx.Session().ID()),
		telemetry.GCPVertexAgentReconnectAttemptKey.Int(attempt),
	)
	var sessionErr error
	defer func() {
		telemetry.TraceGenerateContentResult(span, telemetry.TraceGenerateContentResultParams{
			Error: sessionErr,
		})
		span.End()
	}()
	// cancelCtx now carries the generate_content span. Downstream goroutines
	// (startSessionLoops, receiverLoop, executeToolGroups) inherit the span
	// via cancelCtx — invCtx (`ctx`) is intentionally NOT rebound, so that
	// applySessionResumptionUpdate's mutations to the resumption handle
	// propagate back to RunLive's outer loop.
	cancelCtx = spanCtx

	if handle == "" {
		if err := lf.sendHistory(cancelCtx, ctx, conn, ts); err != nil {
			cancel()
			_ = conn.Close()
			sessionErr = fmt.Errorf("history handoff failed: %w", err)
			yield(nil, sessionErr)
			return false, true
		}
	}

	eventCh, wg := lf.startSessionLoops(cancelCtx, ctx, conn, queue, toolsFuncMap, ts)

	for msg := range eventCh {
		applySessionResumptionUpdate(ctx, msg)
		if isGoAway(msg) {
			shouldReconnect = true
			break
		}
		// The server may close the transport without a preceding GoAway
		// frame (e.g., a normal close at code 1000 or an abnormal one at
		// 1006, or a plain io.EOF). When we hold a resumption handle, the
		// outer reconnect cycle is the right place to retry — surface the
		// error instead and the iterator would terminate prematurely.
		//
		// Skip the reconnect path if the queue has already been drained
		// (client-initiated shutdown): the sender exits on queue.Done(),
		// which closes the connection and makes the receiver observe an
		// EOF that is part of the natural unwind, not a server-initiated
		// disconnect that warrants a reconnect.
		if msg.err != nil && isConnectionClosed(msg.err) && ctx.LiveSessionResumptionHandle() != "" && !queueDone(queue) {
			shouldReconnect = true
			break
		}
		if !yield(msg.event, msg.err) {
			cancel()
			_ = conn.Close()
			wg.Wait()
			return false, true
		}
	}

	cancel()
	_ = conn.Close()
	wg.Wait()
	return shouldReconnect, false
}

// startSessionLoops launches the sender and receiver goroutines for a session
// and returns the event channel they write to (closed once both exit) plus
// the WaitGroup that tracks them.
func (lf *LiveFlow) startSessionLoops(
	cancelCtx context.Context,
	invCtx agent.InvocationContext,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	toolsFuncMap map[string]toolinternal.FunctionTool,
	ts *liveTimingState,
) (chan eventOrError, *sync.WaitGroup) {
	eventCh := make(chan eventOrError, 64)
	wg := &sync.WaitGroup{}
	cs := &coalesceState{
		buffer:         make(map[string]*genai.FunctionCall),
		dedup:          make(map[string]string),
		inFlightCancel: make(map[string]context.CancelFunc),
		flushCh:        make(chan struct{}, 1),
		nonThoughtIDs:  make(map[string]struct{}),
	}
	turnResetCh := make(chan struct{}, 1)

	wg.Add(1)
	go func() {
		defer wg.Done()
		lf.senderLoop(cancelCtx, conn, queue, ts, eventCh, turnResetCh)
		_ = conn.Close()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		lf.receiverLoop(cancelCtx, invCtx, conn, queue, cs, toolsFuncMap, ts, eventCh, wg, turnResetCh)
	}()

	go func() {
		wg.Wait()
		close(eventCh)
	}()

	return eventCh, wg
}

func (lf *LiveFlow) sendHistory(
	cancelCtx context.Context,
	ctx agent.InvocationContext,
	conn model.LiveConnection,
	ts *liveTimingState,
) error {
	events := ctx.Session().Events()

	// Collect non-nil content events.
	var turns []*genai.Content
	for i := range events.Len() {
		ev := events.At(i)
		if ev.Content != nil {
			turns = append(turns, ev.Content)
		}
	}
	if len(turns) == 0 {
		return nil
	}

	// Send all history turns with TurnComplete=false so the model absorbs
	// them as context without responding to each one individually.
	// Only the last turn is sent with TurnComplete=true to signal
	// that history replay is complete.
	falseVal := false
	for i, content := range turns {
		isLast := i == len(turns)-1
		req := &model.LiveRequest{Content: content}
		if !isLast {
			req.TurnComplete = &falseVal
		}
		// Last turn uses default (TurnComplete=nil → true).
		if err := trackedSend(cancelCtx, conn, req, ts); err != nil {
			return err
		}
	}
	return nil
}

// ErrIter returns an iterator that yields a single error.
func ErrIter(err error) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(nil, err)
	}
}
