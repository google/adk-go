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
}

type recvResult struct {
	resp *model.LLMResponse
	err  error
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
func (lf *LiveFlow) RunLive(
	ctx agent.InvocationContext,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		cancelCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		toolsFuncMap := make(map[string]toolinternal.FunctionTool)
		for _, t := range lf.Tools {
			if ft, ok := t.(toolinternal.FunctionTool); ok {
				toolsFuncMap[t.Name()] = ft
			}
		}

		if err := lf.sendHistory(cancelCtx, ctx, conn); err != nil {
			yield(nil, fmt.Errorf("history handoff failed: %w", err))
			return
		}

		eventCh := make(chan eventOrError, 64)
		var wg sync.WaitGroup
		cs := &coalesceState{
			buffer:         make(map[string]*genai.FunctionCall),
			dedup:          make(map[string]string),
			inFlightCancel: make(map[string]context.CancelFunc),
			flushCh:        make(chan struct{}, 1),
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			lf.senderLoop(cancelCtx, conn, queue, eventCh)
			// Close connection when sender is done (queue closed or error).
			// This unblocks the receiver's conn.Receive without cancelling
			// the context used by in-flight tool calls.
			_ = conn.Close()
		}()

		wg.Add(1)
		go func() {
			defer wg.Done()
			lf.receiverLoop(cancelCtx, ctx, conn, queue, cs, toolsFuncMap, eventCh, &wg)
		}()

		// Closer goroutine: eventCh is closed only after ALL producers
		// (sender, receiver, receive worker) have exited. This is the
		// single owner of the close operation, eliminating races.
		go func() {
			wg.Wait()
			close(eventCh)
		}()

		for msg := range eventCh {
			if !yield(msg.event, msg.err) {
				break
			}
		}

		cancel()
		_ = conn.Close()
		wg.Wait()
	}
}

func (lf *LiveFlow) sendHistory(
	cancelCtx context.Context,
	ctx agent.InvocationContext,
	conn model.LiveConnection,
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
		if err := conn.Send(cancelCtx, req); err != nil {
			return err
		}
	}
	return nil
}

// senderLoop forwards queue messages to the live connection.
// On queue.Done(), it drains any remaining buffered messages before returning.
func (lf *LiveFlow) senderLoop(
	ctx context.Context,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	eventCh chan<- eventOrError,
) {
	for {
		select {
		case req := <-queue.Events():
			if err := conn.Send(ctx, req); err != nil {
				sendEvent(ctx, eventCh, eventOrError{err: err})
				return
			}
		case <-queue.Done():
			// Drain any buffered messages before exiting.
			for {
				select {
				case req := <-queue.Events():
					if err := conn.Send(ctx, req); err != nil {
						sendEvent(ctx, eventCh, eventOrError{err: err})
						return
					}
				default:
					return
				}
			}
		case <-ctx.Done():
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
	eventCh chan<- eventOrError,
	wg *sync.WaitGroup,
) {
	defer stopCoalesceTimer(cs)
	recvCh := startReceiveWorker(ctx, conn, cs, wg)

	for {
		select {
		case r := <-recvCh:
			if done := lf.handleRecv(ctx, invCtx, conn, queue, cs, toolsFuncMap, r, eventCh); done {
				return
			}
		case <-cs.flushCh:
			lf.flushToolCalls(ctx, invCtx, conn, cs, toolsFuncMap, eventCh)
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
	r recvResult,
	eventCh chan<- eventOrError,
) bool {
	if r.err != nil {
		if ctx.Err() != nil || isEOF(r.err) {
			return true
		}
		sendEvent(ctx, eventCh, eventOrError{err: r.err})
		return true
	}
	lf.processMessage(ctx, invCtx, conn, queue, cs, toolsFuncMap, r.resp, eventCh)
	return false
}

func (lf *LiveFlow) processMessage(
	ctx context.Context,
	invCtx agent.InvocationContext,
	conn model.LiveConnection,
	queue *agent.LiveRequestQueue,
	cs *coalesceState,
	toolsFuncMap map[string]toolinternal.FunctionTool,
	resp *model.LLMResponse,
	eventCh chan<- eventOrError,
) {
	if ids, ok := resp.CustomMetadata["tool_cancellation_ids"].([]string); ok {
		handleToolCancellation(cs, ids)
		return
	}

	if hasFunctionCallParts(resp) {
		lf.bufferToolCalls(cs, resp)
		return
	}

	lf.flushCoalesceBuffer(ctx, invCtx, conn, cs, toolsFuncMap, eventCh)

	isAudio := false
	if resp.CustomMetadata != nil {
		if v, ok := resp.CustomMetadata["is_audio"].(bool); ok {
			isAudio = v
		}
	}

	if isAudio {
		queue.SetModelSpeaking(true)
	}
	if resp.TurnComplete || resp.Interrupted {
		queue.SetModelSpeaking(false)
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
	sendEvent(ctx, eventCh, eventOrError{event: ev})
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
	eventCh chan<- eventOrError,
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
	lf.flushToolCalls(ctx, invCtx, conn, cs, toolsFuncMap, eventCh)
}

func (lf *LiveFlow) flushToolCalls(
	ctx context.Context,
	invCtx agent.InvocationContext,
	conn model.LiveConnection,
	cs *coalesceState,
	toolsFuncMap map[string]toolinternal.FunctionTool,
	eventCh chan<- eventOrError,
) {
	calls := lf.collectBufferedCalls(cs)
	if len(calls) == 0 {
		return
	}

	groups := deduplicateCalls(calls)
	lf.emitFunctionCallEvent(ctx, invCtx, calls, eventCh)
	responses := lf.executeToolGroups(ctx, invCtx, cs, groups, toolsFuncMap)

	if len(responses) == 0 {
		return
	}

	if err := conn.Send(ctx, &model.LiveRequest{ToolResponse: responses}); err != nil {
		eventCh <- eventOrError{err: fmt.Errorf("failed to send tool response: %w", err)}
		return
	}
	lf.emitFunctionResponseEvent(ctx, invCtx, responses, eventCh)
}

func (lf *LiveFlow) collectBufferedCalls(cs *coalesceState) map[string]*genai.FunctionCall {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	if cs.timer != nil {
		cs.timer.Stop()
		cs.timer = nil
	}
	if len(cs.buffer) == 0 {
		return nil
	}

	calls := make(map[string]*genai.FunctionCall, len(cs.buffer))
	maps.Copy(calls, cs.buffer)
	cs.buffer = make(map[string]*genai.FunctionCall)
	cs.dedup = make(map[string]string)
	return calls
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
