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
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/plugininternal/plugincontext"
	"google.golang.org/adk/internal/telemetry"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

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

func stopCoalesceTimer(cs *coalesceState) {
	cs.mu.Lock()
	if cs.timer != nil {
		cs.timer.Stop()
		cs.timer = nil
	}
	cs.mu.Unlock()
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

func buildToolsFuncMap(tools []tool.Tool) map[string]toolinternal.FunctionTool {
	m := make(map[string]toolinternal.FunctionTool)
	for _, t := range tools {
		if ft, ok := t.(toolinternal.FunctionTool); ok {
			m[t.Name()] = ft
		}
	}
	return m
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

	// Mirror non-Live (base_flow.go:589-596): merged span only when more
	// than one tool call. A single-tool flush goes straight to the per-call
	// execute_tool span — no extra layer.
	var mergedEvent *session.Event
	var mergedErr error
	if len(groups) > 1 {
		mergedCtx, mergedSpan := telemetry.StartTrace(ctx, "execute_tool (merged)")
		ctx = mergedCtx
		defer func() {
			telemetry.TraceMergedToolCallsResult(mergedSpan, mergedEvent, mergedErr)
			mergedSpan.End()
		}()
	}

	lf.emitFunctionCallEvent(ctx, invCtx, calls, queue, ts, eventCh)

	toolStart := time.Now()
	responses := lf.executeToolGroups(ctx, invCtx, cs, groups, toolsFuncMap)
	toolExecTime := time.Since(toolStart)

	if len(responses) == 0 {
		return
	}

	req := &model.LiveRequest{ToolResponse: responses}
	if err := trackedSend(ctx, conn, req, ts); err != nil {
		mergedErr = fmt.Errorf("failed to send tool response: %w", err)
		eventCh <- eventOrError{err: mergedErr}
		return
	}
	mergedEvent = lf.emitFunctionResponseEvent(ctx, invCtx, responses, queue, ts, toolExecTime, eventCh)

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

func hashCall(name string, args map[string]any) string {
	data, err := json.Marshal(args)
	if err != nil {
		data = []byte(fmt.Sprintf("%v", args))
	}
	h := sha256.Sum256(append([]byte(name+":"), data...))
	return fmt.Sprintf("%x", h[:8])
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

		// execute_tool span wraps a single tool dispatch. The span context
		// (sctx) becomes the parent of toolCtx so cancellation and tracing
		// are layered cleanly: cancellation is governed by toolCancel,
		// tracing by sctx — and both reach the tool implementation when
		// callToolLive rebinds invCtx onto toolCtx.
		sctx, span := telemetry.StartExecuteToolSpan(ctx, telemetry.StartExecuteToolSpanParams{
			ToolName: fc.Name,
			Args:     fc.Args,
		})

		toolCtx, toolCancel := context.WithCancel(sctx)
		cs.mu.Lock()
		cs.inFlightCancel[primaryID] = toolCancel
		cs.mu.Unlock()

		result, toolErr := lf.callToolLive(toolCtx, invCtx, fc, toolsFuncMap)

		// Check for external cancellation BEFORE calling toolCancel(),
		// otherwise toolCtx.Err() is always non-nil after our own cleanup cancel.
		wasCancelled := toolCtx.Err() != nil
		toolCancel()
		cs.mu.Lock()
		delete(cs.inFlightCancel, primaryID)
		cs.mu.Unlock()

		if wasCancelled {
			// Model-initiated cancellation is not a failure — leave span at
			// codes.Unset and skip TraceToolResult so we do not record a
			// `<elided>` response payload for a tool that never produced one.
			span.SetAttributes(telemetry.GCPVertexAgentToolCallCancelledKey.Bool(true))
			span.End()
			continue
		}

		// Mirror non-Live (base_flow.go:661-674): if the tool returned a Go
		// error, callToolLive packed it into result["error"] AND surfaced it
		// as toolErr. Pass toolErr to TraceToolResult so the span ends with
		// codes.Error, matching non-Live span semantics.
		perCallEv := buildToolTraceEvent(invCtx, fc, primaryID, result)
		desc := ""
		if funcTool, ok := toolsFuncMap[fc.Name]; ok {
			desc = funcTool.Description()
		}
		telemetry.TraceToolResult(span, telemetry.TraceToolResultParams{
			Description:   desc,
			ResponseEvent: perCallEv,
			Error:         toolErr,
		})
		span.End()

		for _, id := range g.ids {
			responses = append(responses, &genai.FunctionResponse{
				ID: id, Name: fc.Name, Response: result,
			})
		}
	}
	return responses
}

// buildToolTraceEvent constructs the minimal *session.Event needed by
// TraceToolResult to extract the FunctionResponse's ID and (gated) response
// payload. Kept separate from emitFunctionResponseEvent since the trace event
// is per-call, while the emitted event is per-batch.
func buildToolTraceEvent(invCtx agent.InvocationContext, fc *genai.FunctionCall, id string, result map[string]any) *session.Event {
	ev := session.NewEvent(invCtx.InvocationID())
	ev.Author = invCtx.Agent().Name()
	ev.Branch = invCtx.Branch()
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role: "user",
			Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: id, Name: fc.Name, Response: result}},
			},
		},
	}
	return ev
}

// emitFunctionResponseEvent builds and sends the FunctionResponse event,
// returning the event so callers (e.g., flushToolCalls's merged span) can
// pass it to TraceMergedToolCallsResult.
func (lf *LiveFlow) emitFunctionResponseEvent(
	ctx context.Context,
	invCtx agent.InvocationContext,
	responses []*genai.FunctionResponse,
	queue *agent.LiveRequestQueue,
	ts *liveTimingState,
	toolExecTime time.Duration,
	eventCh chan<- eventOrError,
) *session.Event {
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
	return ev
}

// callToolLive runs the tool and returns its response map. The Go error
// (when the tool failed) is returned alongside the map so the caller can pass
// it to telemetry.TraceToolResult — mirroring the non-Live path's
// `result["error"]` extraction at base_flow.go:661-674.
func (lf *LiveFlow) callToolLive(
	ctx context.Context,
	invCtx agent.InvocationContext,
	fc *genai.FunctionCall,
	toolsFuncMap map[string]toolinternal.FunctionTool,
) (map[string]any, error) {
	funcTool, ok := toolsFuncMap[fc.Name]
	if !ok {
		err := fmt.Errorf("tool %q not found", fc.Name)
		return map[string]any{"error": err.Error()}, err
	}

	// Rebind invCtx onto ctx so toolinternal.NewToolContext produces a
	// tool.Context whose Value() chain reaches the execute_tool span.
	// Without this, a tool implementation that calls
	// trace.SpanFromContext(toolCtx) would observe the generate_content
	// or live_session span instead and would parent any sub-spans
	// incorrectly.
	invCtx = invCtx.WithContext(ctx)

	toolCtx := toolinternal.NewToolContext(invCtx, fc.ID, &session.EventActions{StateDelta: make(map[string]any)}, nil)
	wrappedCtx := &cancelableToolContext{Context: toolCtx, cancelCtx: ctx}

	response, err := lf.runToolWithCallbacks(wrappedCtx, invCtx, funcTool, fc.Args)
	if err != nil {
		return map[string]any{"error": err.Error()}, err
	}
	return response, nil
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

// pluginManager mirrors the private interface in internal/llminternal. It is
// duplicated here (rather than exported) so the liveflow extraction does not
// widen the llminternal API surface. Interface satisfaction is structural, so
// the runtime plugin-manager value stored under plugincontext.PluginManagerCtxKey
// satisfies both definitions.
type pluginManager interface {
	RunBeforeModelCallback(cctx agent.CallbackContext, llmRequest *model.LLMRequest) (*model.LLMResponse, error)
	RunAfterModelCallback(cctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)
	RunOnModelErrorCallback(ctx agent.CallbackContext, llmRequest *model.LLMRequest, llmResponseError error) (*model.LLMResponse, error)
	RunBeforeToolCallback(ctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error)
	RunAfterToolCallback(ctx tool.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error)
	RunOnToolErrorCallback(ctx tool.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error)
}

func pluginManagerFromContext(ctx context.Context) pluginManager {
	m, ok := ctx.Value(plugincontext.PluginManagerCtxKey).(pluginManager)
	if !ok {
		return nil
	}
	return m
}
