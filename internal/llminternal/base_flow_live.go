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

package llminternal

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"strings"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// LiveFlow is the core engine for bidirectional live streaming.
type LiveFlow struct {
	Model                model.LLM
	Tools                []tool.Tool
	BeforeToolCallbacks  []BeforeToolCallback
	AfterToolCallbacks   []AfterToolCallback
	OnToolErrorCallbacks []OnToolErrorCallback
	CoalesceWindow       time.Duration
}

type eventOrError struct {
	event *session.Event
	err   error
}

type coalesceState struct {
	mu             sync.Mutex
	buffer         map[string]*genai.FunctionCall
	dedup          map[string]string
	timer          *time.Timer
	inFlightCancel map[string]context.CancelFunc
	flushCh        chan struct{} // signaled by timer; consumed by receiver loop
	// nonThoughtIDs holds the set of buffered FunctionCall IDs that came
	// from non-thought parts. Maintained in lockstep with `buffer`:
	//   - bufferToolCalls inserts on buffer-add (skipping thought parts)
	//   - handleToolCancellation removes when a cancelled call is dropped
	//   - collectBufferedCalls reads len() and clears on drain
	// flushToolCalls uses len(nonThoughtIDs) to decide whether to reset
	// the turn-cycle guard at the FunctionResponse boundary. Tracking IDs
	// (rather than a bool) keeps the gate accurate when a real call is
	// cancelled before flush and only thought calls remain.
	nonThoughtIDs map[string]struct{}
}

type recvResult struct {
	resp *model.LLMResponse
	err  error
}

// turnCycleGuard tracks turn-cycle boundaries within receiverLoop to
// suppress duplicate model content caused by orphaned tool results.
// NOT goroutine-safe — only accessed from the receiverLoop goroutine.
type turnCycleGuard struct {
	contentDelivered bool // model content emitted since last reset
	suppressActive   bool // suppress subsequent model content
}

func (g *turnCycleGuard) onModelContent() bool {
	if g.suppressActive {
		return true // suppress
	}
	g.contentDelivered = true
	return false
}

func (g *turnCycleGuard) onTurnComplete() {
	if g.contentDelivered {
		g.suppressActive = true
	}
}

// markContentDelivered records that model content was emitted in the current
// turn. Used by the Interrupted bypass path so a subsequent onTurnComplete
// arms suppression for any post-Interrupted duplicates that may follow.
func (g *turnCycleGuard) markContentDelivered() {
	g.contentDelivered = true
}

func (g *turnCycleGuard) reset() {
	g.contentDelivered = false
	g.suppressActive = false
}

type liveTimingState struct {
	mu            sync.Mutex
	lastEventTime time.Time
	lastSendTime  time.Time
}

func (s *liveTimingState) recordSend(t time.Time) {
	s.mu.Lock()
	s.lastSendTime = t
	s.mu.Unlock()
}

// trackedSend wraps conn.Send, stamping SentAt on the request and
// recording the send time in shared timing state.
func trackedSend(ctx context.Context, conn model.LiveConnection, req *model.LiveRequest, ts *liveTimingState) error {
	now := time.Now()
	req.SentAt = now
	ts.recordSend(now)
	return conn.Send(ctx, req)
}

// buildBaseDiagnostics creates a LiveDiagnostics with timing fields that
// apply to all event types (function-call, function-response, and content).
func buildBaseDiagnostics(
	queue *agent.LiveRequestQueue,
	ts *liveTimingState,
	receivedAt time.Time,
	eventTime time.Time,
) *session.LiveDiagnostics {
	diag := &session.LiveDiagnostics{
		ModelSpeaking: queue.ModelSpeaking(),
		QueueDepth:    queue.Len(),
	}

	ts.mu.Lock()
	if !ts.lastEventTime.IsZero() {
		diag.TimeSinceLastEvent = eventTime.Sub(ts.lastEventTime)
	}
	ts.lastEventTime = eventTime
	if !ts.lastSendTime.IsZero() && !receivedAt.IsZero() {
		diag.TimeSinceLastSend = receivedAt.Sub(ts.lastSendTime)
	}
	ts.mu.Unlock()

	return diag
}

// sendEvent sends a message to eventCh, aborting if ctx is cancelled.
// This prevents goroutines from blocking on a full channel after shutdown.
func sendEvent(ctx context.Context, eventCh chan<- eventOrError, msg eventOrError) {
	select {
	case eventCh <- msg:
	case <-ctx.Done():
	}
}

// ConnectFn creates a new live connection, optionally resuming a previous
// session via the provided handle (empty string = new session).
type ConnectFn func(handle string) (model.LiveConnection, error)

// Reconnect retry knobs. Declared as vars (rather than consts) so tests can
// override them to keep wall time low and make context-cancellation timing
// deterministic. Production behavior is unchanged.
var (
	maxReconnectRetries  = 3
	reconnectBaseBackoff = time.Second
	reconnectGoAwaySleep = 500 * time.Millisecond
)

// RunLive runs the live flow, returning an iterator of events.
// It accepts a ConnectFn instead of a raw connection so it can reconnect
// on GoAway signals using session resumption handles.
func (lf *LiveFlow) RunLive(
	ctx agent.InvocationContext,
	connectFn ConnectFn,
	queue *agent.LiveRequestQueue,
) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		toolsFuncMap := buildToolsFuncMap(lf.Tools)
		ts := &liveTimingState{}

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

			shouldReconnect, terminated := lf.runSession(ctx, conn, queue, toolsFuncMap, ts, handle, yield)
			if terminated || !shouldReconnect {
				return
			}

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

func buildToolsFuncMap(tools []tool.Tool) map[string]toolinternal.FunctionTool {
	m := make(map[string]toolinternal.FunctionTool)
	for _, t := range tools {
		if ft, ok := t.(toolinternal.FunctionTool); ok {
			m[t.Name()] = ft
		}
	}
	return m
}

// connectWithRetry attempts to connect up to maxReconnectRetries+1 times,
// applying linear backoff between attempts. Returns context.Canceled (or
// the deadline error) immediately if the parent context is cancelled at
// any point — including during backoff — so callers do not wait out timers
// after a cancel.
func connectWithRetry(ctx context.Context, handle string, connectFn ConnectFn) (model.LiveConnection, error) {
	var lastErr error
	for attempt := 0; attempt <= maxReconnectRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		conn, err := connectFn(handle)
		if err == nil {
			return conn, nil
		}
		lastErr = err
		if attempt < maxReconnectRetries {
			if !sleepOrCancel(ctx, time.Duration(attempt+1)*reconnectBaseBackoff) {
				return nil, ctx.Err()
			}
		}
	}
	return nil, fmt.Errorf("failed to connect live after %d retries: %w", maxReconnectRetries, lastErr)
}

// sleepOrCancel waits for d to elapse or for ctx to be cancelled, returning
// true if the duration elapsed and false if the context was cancelled.
func sleepOrCancel(ctx context.Context, d time.Duration) bool {
	select {
	case <-time.After(d):
		return true
	case <-ctx.Done():
		return false
	}
}

// queueDone reports whether the LiveRequestQueue has been closed by the
// caller (queue.Done() channel is signalled). Used to skip the close-
// reconnect path during a client-initiated shutdown — without this gate,
// closing the queue triggers conn.Close, which the receiver observes as
// an EOF and would otherwise re-enter the reconnect cycle.
func queueDone(queue *agent.LiveRequestQueue) bool {
	select {
	case <-queue.Done():
		return true
	default:
		return false
	}
}

// isConnectionClosed reports whether err looks like a closed transport:
// io.EOF, an unexpectedly-closed read, or a WebSocket close frame with
// code 1000 (normal) / 1006 (abnormal). When such a close arrives without
// a preceding GoAway frame, callers can decide — based on whether a
// resumption handle is available — whether to reconnect via the outer
// reconnect cycle or surface the error.
//
// Permissive on purpose: when in doubt, prefer treating it as closed so
// the outer cycle's retry budget remains the gate against runaway
// reconnects.
func isConnectionClosed(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	// gorilla/websocket's *CloseError exposes Code() int.
	var ce interface{ Code() int }
	if errors.As(err, &ce) {
		switch ce.Code() {
		case 1000, 1006:
			return true
		}
	}
	// Substring fallback for transports that render close frames as
	// plain strings without a typed wrapper. "close 1000" / "close 1006"
	// are supersets of "websocket: close 100X" so we don't need both.
	msg := err.Error()
	switch {
	case strings.Contains(msg, "close 1000"),
		strings.Contains(msg, "close 1006"):
		return true
	}
	return false
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
	yield func(*session.Event, error) bool,
) (shouldReconnect, terminated bool) {
	cancelCtx, cancel := context.WithCancel(ctx)

	if handle == "" {
		if err := lf.sendHistory(cancelCtx, ctx, conn, ts); err != nil {
			cancel()
			_ = conn.Close()
			yield(nil, fmt.Errorf("history handoff failed: %w", err))
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

// applySessionResumptionUpdate mutates ctx's resumption handle in response
// to a server-supplied SessionResumptionUpdate event, if the message carries
// one. Non-resumable updates clear any stale handle so the next reconnect
// uses an empty handle.
func applySessionResumptionUpdate(ctx agent.InvocationContext, msg eventOrError) {
	if msg.event == nil || msg.event.SessionResumptionUpdate == nil {
		return
	}
	upd := msg.event.SessionResumptionUpdate
	if !upd.Resumable {
		ctx.SetLiveSessionResumptionHandle("")
		return
	}
	if upd.NewHandle != "" {
		ctx.SetLiveSessionResumptionHandle(upd.NewHandle)
	}
}

func isGoAway(msg eventOrError) bool {
	return msg.event != nil && msg.event.GoAway != nil
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

// sendAndSignal sends a request on the connection and signals turnResetCh.
// Returns a non-nil error if the send fails.
func sendAndSignal(
	ctx context.Context,
	conn model.LiveConnection,
	req *model.LiveRequest,
	ts *liveTimingState,
	eventCh chan<- eventOrError,
	turnResetCh chan<- struct{},
) error {
	if err := trackedSend(ctx, conn, req, ts); err != nil {
		sendEvent(ctx, eventCh, eventOrError{err: err})
		return err
	}
	// Signal receiverLoop that user activity occurred.
	// Non-blocking: if signal already pending, skip — receiver will see it.
	select {
	case turnResetCh <- struct{}{}:
	default:
	}
	return nil
}

// senderLoop forwards queue messages to the live connection.
// On queue.Done(), it drains any remaining buffered messages before returning.
func (lf *LiveFlow) senderLoop(
	ctx context.Context,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	ts *liveTimingState,
	eventCh chan<- eventOrError,
	turnResetCh chan<- struct{},
) {
	for {
		select {
		case req := <-queue.Events():
			if sendAndSignal(ctx, conn, req, ts, eventCh, turnResetCh) != nil {
				return
			}
		case <-queue.Done():
			lf.drainQueue(ctx, conn, queue, ts, eventCh, turnResetCh)
			return
		case <-ctx.Done():
			return
		}
	}
}

// drainQueue sends any remaining buffered messages before senderLoop exits.
func (lf *LiveFlow) drainQueue(
	ctx context.Context,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	ts *liveTimingState,
	eventCh chan<- eventOrError,
	turnResetCh chan<- struct{},
) {
	for {
		select {
		case req := <-queue.Events():
			if sendAndSignal(ctx, conn, req, ts, eventCh, turnResetCh) != nil {
				return
			}
		default:
			return
		}
	}
}

// startReceiveWorker launches a goroutine that continuously reads from
// conn.Receive and forwards results to the returned channel.
// Tool cancellation messages are handled immediately in the worker goroutine
// so they take effect even when the receiver loop is blocked executing tools.
// The goroutine is tracked by wg so RunLive can join it before closing eventCh.
func startReceiveWorker(ctx context.Context, conn model.LiveConnection, cs *coalesceState, wg *sync.WaitGroup) <-chan recvResult {
	ch := make(chan recvResult, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			resp, err := conn.Receive(ctx)
			// Handle cancellations immediately so they can interrupt
			// in-flight tool calls without waiting for the receiver loop.
			if err == nil {
				if ids, ok := resp.CustomMetadata["tool_cancellation_ids"].([]string); ok {
					handleToolCancellation(cs, ids)
					continue
				}
			}
			select {
			case ch <- recvResult{resp, err}:
			case <-ctx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()
	return ch
}

// receiverLoop processes messages from the live connection.
// It uses a background goroutine for non-blocking receives and selects
// on the receive channel, flush signal, and context cancellation.
// All sends to eventCh happen only in this goroutine, preventing
// data races with the channel close.
func (lf *LiveFlow) receiverLoop(
	ctx context.Context,
	invCtx agent.InvocationContext,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	cs *coalesceState,
	toolsFuncMap map[string]toolinternal.FunctionTool,
	ts *liveTimingState,
	eventCh chan<- eventOrError,
	wg *sync.WaitGroup,
	turnResetCh <-chan struct{},
) {
	defer stopCoalesceTimer(cs)
	recvCh := startReceiveWorker(ctx, conn, cs, wg)
	guard := &turnCycleGuard{}

	for {
		// Prioritize turnResetCh: drain any pending reset signal before
		// processing a receive or flush. This ensures the guard is in the
		// correct state even when both channels are ready simultaneously.
		select {
		case <-turnResetCh:
			guard.reset()
		default:
		}

		select {
		case r := <-recvCh:
			// Double-check for a reset that arrived between the priority
			// drain and this select (narrow but possible window).
			select {
			case <-turnResetCh:
				guard.reset()
			default:
			}
			if done := lf.handleRecv(ctx, invCtx, conn, queue, cs, toolsFuncMap, ts, r, eventCh, guard); done {
				return
			}
		case <-cs.flushCh:
			lf.flushToolCalls(ctx, invCtx, conn, cs, toolsFuncMap, ts, queue, eventCh, guard)
		case <-turnResetCh:
			guard.reset()
		case <-ctx.Done():
			return
		}
	}
}

func stopCoalesceTimer(cs *coalesceState) {
	cs.mu.Lock()
	if cs.timer != nil {
		cs.timer.Stop()
		cs.timer = nil
	}
	cs.mu.Unlock()
}

// handleRecv processes a single receive result. Returns true when the loop should exit.
func (lf *LiveFlow) handleRecv(
	ctx context.Context,
	invCtx agent.InvocationContext,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	cs *coalesceState,
	toolsFuncMap map[string]toolinternal.FunctionTool,
	ts *liveTimingState,
	r recvResult,
	eventCh chan<- eventOrError,
	guard *turnCycleGuard,
) bool {
	if r.err != nil {
		if ctx.Err() != nil {
			return true
		}
		// An EOF observed during a client-initiated shutdown (queue closed,
		// sender exited and called conn.Close) is part of the natural
		// unwind, not a server-initiated disconnect. Stay silent there to
		// preserve the existing clean-shutdown contract; forward all other
		// errors — including EOF mid-session — so runSession can decide
		// whether to reconnect via the saved handle.
		if isEOF(r.err) && queueDone(queue) {
			return true
		}
		sendEvent(ctx, eventCh, eventOrError{err: r.err})
		return true
	}
	lf.processMessage(ctx, invCtx, conn, queue, cs, toolsFuncMap, ts, r.resp, eventCh, guard)
	return false
}

func (lf *LiveFlow) processMessage(
	ctx context.Context,
	invCtx agent.InvocationContext,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	cs *coalesceState,
	toolsFuncMap map[string]toolinternal.FunctionTool,
	ts *liveTimingState,
	resp *model.LLMResponse,
	eventCh chan<- eventOrError,
	guard *turnCycleGuard,
) {
	if ids, ok := resp.CustomMetadata["tool_cancellation_ids"].([]string); ok {
		handleToolCancellation(cs, ids)
		return
	}

	if hasFunctionCallParts(resp) {
		// FC-arrival no longer resets the guard; the FunctionResponse
		// boundary owns the reset (see flushToolCalls), gated on the
		// presence of at least one non-thought call. Thought-only
		// emissions therefore never end the turn cycle on their own.
		lf.bufferToolCalls(cs, resp)
		return
	}

	lf.flushCoalesceBuffer(ctx, invCtx, conn, cs, toolsFuncMap, ts, queue, eventCh, guard)

	isAudio := false
	if resp.CustomMetadata != nil {
		if v, ok := resp.CustomMetadata["is_audio"].(bool); ok {
			isAudio = v
		}
	}

	// Audio speaking state (unchanged).
	if isAudio {
		queue.SetModelSpeaking(true)
	}
	if resp.TurnComplete || resp.Interrupted {
		queue.SetModelSpeaking(false) // ALWAYS fires, even when suppressing
	}

	// Guard logic: suppress duplicate model content across turn boundaries.
	resp = applyTurnCycleGuard(resp, guard, isAudio)
	if resp == nil {
		return // fully suppressed
	}

	ev := session.NewEvent(invCtx.InvocationID())
	ev.Author = invCtx.Agent().Name()
	// Only relabel as "user" when the content is purely transcription-derived
	// (no ModelTurn). A single server message can carry both modelTurn and
	// inputTranscription; blindly overriding Author would misattribute model
	// content as user-produced.
	if resp.InputTranscription != nil && (resp.Content == nil || resp.Content.Role == "user") {
		ev.Author = "user"
	}
	ev.Branch = invCtx.Branch()
	ev.LLMResponse = *resp

	diag := buildBaseDiagnostics(queue, ts, resp.ReceivedAt, ev.Timestamp)
	diag.ReceiveLatency = resp.ReceiveLatency
	populateProtocolState(resp, diag)
	ev.LiveDiagnostics = diag

	sendEvent(ctx, eventCh, eventOrError{event: ev})
}

// applyTurnCycleGuard evaluates the guard for the given response and returns
// the response to forward downstream, or nil when the message should be
// fully suppressed.
//
// Suppression matrix (when the guard is armed and the message is model-
// content-bearing — i.e. Content!=nil with role=="model" and !isAudio):
//
//	Content+OT only       → nil (fully suppressed)
//	Content+OT+IT         → Content/OT stripped, IT preserved
//	Content+IT (no OT)    → Content stripped, IT preserved
//
// OT-stripping nuance: OutputTranscription is stripped only when the
// response is model-content-bearing AND the guard suppresses. An OT-only
// message (Content==nil) is a documented bypass case — isModelContent is
// false, the guard is not consulted, and OT is forwarded untouched. Audio
// model content also bypasses the guard via the !isAudio carve-out, so
// audio frames are never stripped here.
//
// InputTranscription (user input) is never stripped: it is not a model
// duplicate. The Interrupted branch bypasses the suppress check for the
// current message (it carries the truncated model output, which the
// caller must observe) and arms the guard so post-Interrupted duplicates
// that arrive before user activity are suppressed.
func applyTurnCycleGuard(resp *model.LLMResponse, guard *turnCycleGuard, isAudio bool) *model.LLMResponse {
	isModelContent := resp.Content != nil && resp.Content.Role == "model" && !isAudio

	// Interrupted: the truncated model output IS this message — deliver it
	// (bypass suppression). Arm the guard so any subsequent duplicates that
	// arrive before user activity are suppressed.
	if resp.Interrupted {
		if isModelContent {
			guard.markContentDelivered()
		}
		guard.onTurnComplete()
		return resp
	}

	suppressModelContent := false
	if isModelContent {
		suppressModelContent = guard.onModelContent()
	}

	// Arm only on a non-partial TurnComplete. Partial events fire before
	// the aggregate that carries the real content; arming on them would
	// strip the aggregate's content.
	if resp.TurnComplete && !resp.Partial {
		guard.onTurnComplete()
	}

	if !suppressModelContent {
		return resp
	}

	// Suppression: strip model output (Content + OutputTranscription).
	// InputTranscription (user input) is preserved since it is not a
	// model duplicate. With no IT remaining, the message is fully
	// suppressed (return nil).
	stripped := *resp
	stripped.Content = nil
	stripped.OutputTranscription = nil
	if stripped.InputTranscription == nil {
		return nil
	}
	return &stripped
}

// populateProtocolState extracts protocol state from CustomMetadata into LiveDiagnostics.
func populateProtocolState(resp *model.LLMResponse, diag *session.LiveDiagnostics) {
	if reason, ok := resp.CustomMetadata["turn_complete_reason"].(string); ok {
		diag.TurnCompleteReason = reason
	}
	if ms, ok := resp.CustomMetadata["go_away_time_left_ms"].(float64); ok {
		diag.GoAwayTimeLeft = time.Duration(ms) * time.Millisecond
	}
	if vad, ok := resp.CustomMetadata["vad_signal_type"].(string); ok {
		diag.VADSignalType = vad
	}
	if handle, ok := resp.CustomMetadata["session_resumption_handle"].(string); ok {
		diag.SessionResumptionHandle = handle
	}
	if resumable, ok := resp.CustomMetadata["session_resumption_resumable"].(bool); ok {
		diag.SessionResumable = resumable
	}
	if wi, ok := resp.CustomMetadata["waiting_for_input"].(bool); ok {
		diag.WaitingForInput = wi
	}
}

func hasFunctionCallParts(resp *model.LLMResponse) bool {
	if resp.Content == nil {
		return false
	}
	for _, p := range resp.Content.Parts {
		if p.FunctionCall != nil {
			return true
		}
	}
	return false
}

// bufferToolCalls adds function calls to the coalesce buffer and resets the
// flush timer. When the timer fires, it signals flushCh which is consumed
// by the receiver loop (same goroutine that owns eventCh).
func (lf *LiveFlow) bufferToolCalls(
	cs *coalesceState,
	resp *model.LLMResponse,
) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	for _, part := range resp.Content.Parts {
		if part.FunctionCall == nil {
			continue
		}
		fc := part.FunctionCall
		if !part.Thought {
			cs.nonThoughtIDs[fc.ID] = struct{}{}
		}
		hash := hashCall(fc.Name, fc.Args)
		if _, ok := cs.dedup[hash]; !ok {
			cs.dedup[hash] = fc.ID
		}
		cs.buffer[fc.ID] = fc
	}

	if cs.timer != nil {
		cs.timer.Stop()
	}
	window := lf.CoalesceWindow
	if window == 0 {
		window = 150 * time.Millisecond
	}
	cs.timer = time.AfterFunc(window, func() {
		select {
		case cs.flushCh <- struct{}{}:
		default:
		}
	})
}

func hashCall(name string, args map[string]any) string {
	data, err := json.Marshal(args)
	if err != nil {
		data = []byte(fmt.Sprintf("%v", args))
	}
	h := sha256.Sum256(append([]byte(name+":"), data...))
	return fmt.Sprintf("%x", h[:8])
}

func (lf *LiveFlow) flushCoalesceBuffer(
	ctx context.Context,
	invCtx agent.InvocationContext,
	conn model.LiveConnection,
	cs *coalesceState,
	toolsFuncMap map[string]toolinternal.FunctionTool,
	ts *liveTimingState,
	queue *agent.LiveRequestQueue,
	eventCh chan<- eventOrError,
	guard *turnCycleGuard,
) {
	cs.mu.Lock()
	if cs.timer != nil {
		cs.timer.Stop()
		cs.timer = nil
	}
	// Drain any pending flush signal from the timer.
	select {
	case <-cs.flushCh:
	default:
	}
	empty := len(cs.buffer) == 0
	cs.mu.Unlock()

	if empty {
		return
	}
	lf.flushToolCalls(ctx, invCtx, conn, cs, toolsFuncMap, ts, queue, eventCh, guard)
}

func (lf *LiveFlow) flushToolCalls(
	ctx context.Context,
	invCtx agent.InvocationContext,
	conn model.LiveConnection,
	cs *coalesceState,
	toolsFuncMap map[string]toolinternal.FunctionTool,
	ts *liveTimingState,
	queue *agent.LiveRequestQueue,
	eventCh chan<- eventOrError,
	guard *turnCycleGuard,
) {
	calls, hasNonThought := lf.collectBufferedCalls(cs)
	if len(calls) == 0 {
		return
	}

	groups := deduplicateCalls(calls)
	lf.emitFunctionCallEvent(ctx, invCtx, calls, queue, ts, eventCh)

	toolStart := time.Now()
	responses := lf.executeToolGroups(ctx, invCtx, cs, groups, toolsFuncMap)
	toolExecTime := time.Since(toolStart)

	if len(responses) == 0 {
		return
	}

	req := &model.LiveRequest{ToolResponse: responses}
	if err := trackedSend(ctx, conn, req, ts); err != nil {
		eventCh <- eventOrError{err: fmt.Errorf("failed to send tool response: %w", err)}
		return
	}
	lf.emitFunctionResponseEvent(ctx, invCtx, responses, queue, ts, toolExecTime, eventCh)

	// Reset the turn-cycle guard only after the FunctionResponse was
	// successfully emitted, and only if the batch contained at least one
	// non-thought call. Thought-only batches must preserve guard state so
	// post-thought duplicates remain suppressed. A failed trackedSend
	// returns above without resetting — the cycle didn't close, so the
	// guard must stay armed.
	if hasNonThought {
		guard.reset()
	}
}

// collectBufferedCalls drains the buffered FunctionCalls together with a
// snapshot of whether any non-thought call was present in the batch.
// Returning the flag alongside the calls keeps the drain atomic — the
// post-flushToolCalls reset gating in flushToolCalls observes the same
// batch boundary as the buffer it just emptied. Cancellations between
// buffer-add and drain are already reflected in nonThoughtIDs (see
// handleToolCancellation), so a real call cancelled before flush does
// not falsely arm the reset.
func (lf *LiveFlow) collectBufferedCalls(cs *coalesceState) (map[string]*genai.FunctionCall, bool) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	if cs.timer != nil {
		cs.timer.Stop()
		cs.timer = nil
	}
	if len(cs.buffer) == 0 {
		// Clear the set even on empty drains so a stale entry can't
		// bleed into the next batch (defence in depth — handlers
		// already keep the set in sync with the buffer).
		cs.nonThoughtIDs = make(map[string]struct{})
		return nil, false
	}

	calls := make(map[string]*genai.FunctionCall, len(cs.buffer))
	maps.Copy(calls, cs.buffer)
	cs.buffer = make(map[string]*genai.FunctionCall)
	cs.dedup = make(map[string]string)
	hasNonThought := len(cs.nonThoughtIDs) > 0
	cs.nonThoughtIDs = make(map[string]struct{})
	return calls, hasNonThought
}

type callGroup struct {
	fc  *genai.FunctionCall
	ids []string
}

func deduplicateCalls(calls map[string]*genai.FunctionCall) map[string]*callGroup {
	groups := make(map[string]*callGroup)
	for id, fc := range calls {
		hash := hashCall(fc.Name, fc.Args)
		if g, ok := groups[hash]; ok {
			g.ids = append(g.ids, id)
		} else {
			groups[hash] = &callGroup{fc: fc, ids: []string{id}}
		}
	}
	return groups
}

func (lf *LiveFlow) emitFunctionCallEvent(
	ctx context.Context,
	invCtx agent.InvocationContext,
	calls map[string]*genai.FunctionCall,
	queue *agent.LiveRequestQueue,
	ts *liveTimingState,
	eventCh chan<- eventOrError,
) {
	fcParts := make([]*genai.Part, 0, len(calls))
	for _, fc := range calls {
		fcParts = append(fcParts, &genai.Part{FunctionCall: fc})
	}
	ev := session.NewEvent(invCtx.InvocationID())
	ev.Author = invCtx.Agent().Name()
	ev.Branch = invCtx.Branch()
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{Role: "model", Parts: fcParts},
	}
	// ReceivedAt is zero for FC events (they aren't from conn.Receive),
	// so TimeSinceLastSend will be zero (guarded by !receivedAt.IsZero()).
	ev.LiveDiagnostics = buildBaseDiagnostics(queue, ts, time.Time{}, ev.Timestamp)
	sendEvent(ctx, eventCh, eventOrError{event: ev})
}

func (lf *LiveFlow) executeToolGroups(
	ctx context.Context,
	invCtx agent.InvocationContext,
	cs *coalesceState,
	groups map[string]*callGroup,
	toolsFuncMap map[string]toolinternal.FunctionTool,
) []*genai.FunctionResponse {
	var responses []*genai.FunctionResponse
	for _, g := range groups {
		fc := g.fc
		primaryID := g.ids[0]

		toolCtx, toolCancel := context.WithCancel(ctx)
		cs.mu.Lock()
		cs.inFlightCancel[primaryID] = toolCancel
		cs.mu.Unlock()

		result := lf.callToolLive(toolCtx, invCtx, fc, toolsFuncMap)

		// Check for external cancellation BEFORE calling toolCancel(),
		// otherwise toolCtx.Err() is always non-nil after our own cleanup cancel.
		wasCancelled := toolCtx.Err() != nil
		toolCancel()
		cs.mu.Lock()
		delete(cs.inFlightCancel, primaryID)
		cs.mu.Unlock()

		if wasCancelled {
			continue
		}

		for _, id := range g.ids {
			responses = append(responses, &genai.FunctionResponse{
				ID: id, Name: fc.Name, Response: result,
			})
		}
	}
	return responses
}

func (lf *LiveFlow) emitFunctionResponseEvent(
	ctx context.Context,
	invCtx agent.InvocationContext,
	responses []*genai.FunctionResponse,
	queue *agent.LiveRequestQueue,
	ts *liveTimingState,
	toolExecTime time.Duration,
	eventCh chan<- eventOrError,
) {
	frParts := make([]*genai.Part, 0, len(responses))
	for _, fr := range responses {
		frParts = append(frParts, &genai.Part{FunctionResponse: fr})
	}
	ev := session.NewEvent(invCtx.InvocationID())
	ev.Author = invCtx.Agent().Name()
	ev.Branch = invCtx.Branch()
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{Role: "user", Parts: frParts},
	}
	diag := buildBaseDiagnostics(queue, ts, time.Time{}, ev.Timestamp)
	diag.ToolExecutionTime = toolExecTime
	ev.LiveDiagnostics = diag
	sendEvent(ctx, eventCh, eventOrError{event: ev})
}

func (lf *LiveFlow) callToolLive(
	ctx context.Context,
	invCtx agent.InvocationContext,
	fc *genai.FunctionCall,
	toolsFuncMap map[string]toolinternal.FunctionTool,
) map[string]any {
	funcTool, ok := toolsFuncMap[fc.Name]
	if !ok {
		return map[string]any{"error": fmt.Sprintf("tool %q not found", fc.Name)}
	}

	toolCtx := toolinternal.NewToolContext(invCtx, fc.ID, &session.EventActions{StateDelta: make(map[string]any)}, nil)
	wrappedCtx := &cancelableToolContext{Context: toolCtx, cancelCtx: ctx}

	response, err := lf.runToolWithCallbacks(wrappedCtx, invCtx, funcTool, fc.Args)
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	return response
}

func (lf *LiveFlow) runToolWithCallbacks(
	toolCtx tool.Context,
	invCtx agent.InvocationContext,
	funcTool toolinternal.FunctionTool,
	args map[string]any,
) (map[string]any, error) {
	pluginMgr := pluginManagerFromContext(invCtx)

	response, err := lf.runBeforeToolPhase(toolCtx, pluginMgr, funcTool, args)
	if response != nil || err != nil {
		return lf.runAfterToolPhase(toolCtx, pluginMgr, funcTool, args, response, err)
	}

	response, err = funcTool.Run(toolCtx, args)
	if err != nil {
		response, err = lf.runOnErrorPhase(toolCtx, pluginMgr, funcTool, args, err)
	}

	return lf.runAfterToolPhase(toolCtx, pluginMgr, funcTool, args, response, err)
}

func (lf *LiveFlow) runBeforeToolPhase(
	toolCtx tool.Context,
	pluginMgr pluginManager,
	funcTool toolinternal.FunctionTool,
	args map[string]any,
) (map[string]any, error) {
	if pluginMgr != nil {
		if r, e := pluginMgr.RunBeforeToolCallback(toolCtx, funcTool, args); r != nil || e != nil {
			return r, e
		}
	}
	return lf.invokeBeforeToolCallbacks(toolCtx, funcTool, args)
}

func (lf *LiveFlow) runOnErrorPhase(
	toolCtx tool.Context,
	pluginMgr pluginManager,
	funcTool toolinternal.FunctionTool,
	args map[string]any,
	origErr error,
) (map[string]any, error) {
	if pluginMgr != nil {
		if r, e := pluginMgr.RunOnToolErrorCallback(toolCtx, funcTool, args, origErr); r != nil || e != nil {
			return r, e
		}
	}
	return lf.invokeOnToolErrorCallbacks(toolCtx, funcTool, args, origErr)
}

func (lf *LiveFlow) runAfterToolPhase(
	toolCtx tool.Context,
	pluginMgr pluginManager,
	funcTool toolinternal.FunctionTool,
	args, response map[string]any,
	err error,
) (map[string]any, error) {
	if pluginMgr != nil {
		if r, e := pluginMgr.RunAfterToolCallback(toolCtx, funcTool, args, response, err); r != nil || e != nil {
			return r, e
		}
	}
	return lf.invokeAfterToolCallbacks(toolCtx, funcTool, args, response, err)
}

// cancelableToolContext wraps a tool.Context with a separate cancellation context.
type cancelableToolContext struct {
	tool.Context
	cancelCtx context.Context
}

func (c *cancelableToolContext) Done() <-chan struct{} {
	return c.cancelCtx.Done()
}

func (c *cancelableToolContext) Err() error {
	return c.cancelCtx.Err()
}

func handleToolCancellation(cs *coalesceState, cancelledIDs []string) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	for _, id := range cancelledIDs {
		delete(cs.buffer, id)
		// Keep the non-thought set in sync with the buffer. If a real
		// (non-thought) call is cancelled before flush, the post-flush
		// reset must NOT fire when only thought calls remain.
		delete(cs.nonThoughtIDs, id)
		if cancelFn, ok := cs.inFlightCancel[id]; ok {
			cancelFn()
			delete(cs.inFlightCancel, id)
		}
	}
}

func (lf *LiveFlow) invokeBeforeToolCallbacks(toolCtx tool.Context, t tool.Tool, fArgs map[string]any) (map[string]any, error) {
	for _, callback := range lf.BeforeToolCallbacks {
		result, err := callback(toolCtx, t, fArgs)
		if err != nil {
			return nil, err
		}
		if result != nil {
			return result, nil
		}
	}
	return nil, nil
}

func (lf *LiveFlow) invokeAfterToolCallbacks(toolCtx tool.Context, t toolinternal.FunctionTool, fArgs, fResult map[string]any, fErr error) (map[string]any, error) {
	for _, callback := range lf.AfterToolCallbacks {
		result, err := callback(toolCtx, t, fArgs, fResult, fErr)
		if err != nil {
			return nil, err
		}
		if result != nil {
			return result, nil
		}
	}
	return fResult, fErr
}

func (lf *LiveFlow) invokeOnToolErrorCallbacks(toolCtx tool.Context, t tool.Tool, fArgs map[string]any, fErr error) (map[string]any, error) {
	for _, callback := range lf.OnToolErrorCallbacks {
		result, err := callback(toolCtx, t, fArgs, fErr)
		if err != nil {
			return nil, err
		}
		if result != nil {
			return result, nil
		}
	}
	return nil, fErr
}

// isEOF returns true if the error is io.EOF or wraps it.
func isEOF(err error) bool {
	return errors.Is(err, io.EOF)
}

// ErrIter returns an iterator that yields a single error.
func ErrIter(err error) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(nil, err)
	}
}
