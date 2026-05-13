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
	"sync"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

type recvResult struct {
	resp *model.LLMResponse
	err  error
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
