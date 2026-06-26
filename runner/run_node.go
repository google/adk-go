// Copyright 2026 Google LLC
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

package runner

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/internal/agent/runconfig"
	artifactinternal "google.golang.org/adk/internal/artifact"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	imemory "google.golang.org/adk/internal/memory"
	"google.golang.org/adk/internal/plugininternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// isLlmAgent reports whether a is an LlmAgent (i.e. backed by the
// internal LLM agent state). The node runtime is used only for LlmAgent
// roots, mirroring adk-python where _run_node_async is reached for an
// LlmAgent.
func isLlmAgent(a agent.Agent) bool {
	_, ok := a.(llminternal.Agent)
	return ok
}

// buildRunnerNode wraps any agent as the workflow node that the node
// runtime drives (Go counterpart of adk-python's build_node). All agent
// kinds share the same wrapper; see newAgentNode.
func buildRunnerNode(a agent.Agent) (workflow.Node, error) {
	if a == nil {
		return nil, fmt.Errorf("node runtime: agent cannot be nil")
	}
	return newAgentNode(a), nil
}

// runNode drives an agent through the workflow engine by wrapping it
// in a single-node workflow (START -> node), reusing the agent path's
// session/plugin/persist pipeline. The wrapping node bridges the agent's
// HITL (long-running tools, which emit LongRunningToolIDs) into a
// workflow pause; see newAgentNode. Go equivalent of adk-python
// Runner._run_node_async.
func (r *Runner) runNode(
	ctx context.Context,
	storedSession session.Session,
	agentToRun agent.Agent,
	msg *genai.Content,
	cfg agent.RunConfig,
	opts runOptions,
	yield func(*session.Event, error) bool,
) {
	node, err := buildRunnerNode(agentToRun)
	if err != nil {
		yield(nil, err)
		return
	}
	// Architectural note: Unlike Go, Python ADK executes standalone agents
	// directly via agent.run_async. Go wraps top-level agents in a synthetic
	// single-node workflow (START -> node) so all execution rides through
	// a unified graph engine. We pass workflow.WithRootWrapper() to prevent
	// this synthetic workflow from stamping an unwanted namespace prefix
	// ("app/agent@1") onto events, ensuring 100% path parity with Python.
	wf, err := workflow.New(rootWorkflowName(r.appName, agentToRun), []workflow.Edge{
		{From: workflow.Start, To: node},
	}, workflow.WithRootWrapper())
	if err != nil {
		yield(nil, fmt.Errorf("failed to build node workflow: %w", err))
		return
	}

	// UserContent is read by Workflow.Run as the workflow's seed input.
	ictx := r.newNodeInvocationContext(ctx, storedSession, agentToRun, msg, cfg)

	// Append the user message to history (also runs the on_user_message
	// plugin callback), same as the agent path.
	ictx, userEvent, err := r.appendMessageToSession(ictx, storedSession, msg, cfg.SaveInputBlobsAsArtifacts, r.pluginManager, opts.stateDelta)
	if err != nil {
		yield(nil, err)
		return
	}
	if opts.yieldUserMessage && userEvent != nil {
		if !yield(userEvent, nil) {
			return
		}
	}

	// Plugin lifecycle: defer after_run, run before_run, honor early exit.
	pluginManager := r.pluginManager
	if pluginManager != nil {
		defer pluginManager.RunAfterRunCallback(ictx)

		earlyExitResult, err := pluginManager.RunBeforeRunCallback(ictx)
		if earlyExitResult != nil || err != nil {
			earlyExitEvent := session.NewEventWithContext(ictx, ictx.InvocationID())
			earlyExitEvent.Author = "user"
			earlyExitEvent.LLMResponse = model.LLMResponse{
				Content: msg,
			}
			if appendErr := r.sessionService.AppendEvent(ictx, storedSession, earlyExitEvent); appendErr != nil {
				yield(nil, fmt.Errorf("failed to add event to session: %w", appendErr))
				return
			}
			yield(earlyExitEvent, err)
			return
		}
	}

	// Resume (HITL continuation) vs Run (fresh): a resume is a turn whose
	// function response answers a node waiting on an open interrupt. The
	// paused state comes from session history, not a persisted blob.
	state, err := wf.ReconstructRunState(storedSession, ictx.InvocationID())
	if err != nil {
		yield(nil, fmt.Errorf("failed to reconstruct workflow run state: %w", err))
		return
	}

	var events iter.Seq2[*session.Event, error]
	responses := buildResumeResponses(msg, state, storedSession)
	if len(responses) > 0 {
		events = wf.Resume(ictx, state, responses)
	} else {
		events = wf.Run(ictx)
	}

	// Consume the workflow's event stream, same loop as the agent path:
	// on_event plugin callback, persist non-partial events, yield.
	for event, evErr := range events {
		if evErr != nil {
			if !yield(nil, evErr) {
				return
			}
			continue
		}

		// Stamp the agent as author on engine control events that have
		// none, else a later turn's findAgentToRun logs a spurious
		// "unknown agent" warning for the empty author.
		if event.Author == "" {
			event.Author = agentToRun.Name()
		}

		if event != nil && !event.LLMResponse.Partial {
			if event.NodeInfo != nil && event.NodeInfo.MessageAsOutput && event.LLMResponse.Content != nil {
				clone := *event
				clone.Output = nil
				event = &clone
			}
		}

		if pluginManager != nil {
			modifiedEvent, perr := pluginManager.RunOnEventCallback(ictx, event)
			if perr != nil {
				if !yield(nil, perr) {
					return
				}
				continue
			}
			if modifiedEvent != nil {
				event = modifiedEvent
			}
		}

		if !event.LLMResponse.Partial {
			if err := r.sessionService.AppendEvent(ictx, storedSession, event); err != nil {
				yield(nil, fmt.Errorf("failed to add event to session: %w", err))
				return
			}
		}

		if !yield(event, nil) {
			return
		}
	}
}

// rootWorkflowName derives the persistence-namespacing name for the
// per-run node workflow. It must be stable across turns (so a paused HITL
// run can be resumed) and unique per agent within a session.
func rootWorkflowName(appName string, a agent.Agent) string {
	return appName + "/" + a.Name()
}

// newNodeInvocationContext builds the node path's InvocationContext,
// mirroring the agent path's wiring (parent map for transfer/tool
// resolution, resolved agent for the flow).
func (r *Runner) newNodeInvocationContext(
	ctx context.Context,
	storedSession session.Session,
	agentToRun agent.Agent,
	msg *genai.Content,
	cfg agent.RunConfig,
) agent.Context {
	ctx = parentmap.ToContext(ctx, r.parents)
	ctx = runconfig.ToContext(ctx, &runconfig.RunConfig{
		StreamingMode: runconfig.StreamingMode(cfg.StreamingMode),
	})
	ctx = plugininternal.ToContext(ctx, r.pluginManager)

	var artifacts agent.Artifacts
	if r.artifactService != nil {
		artifacts = &artifactinternal.Artifacts{
			Service:   r.artifactService,
			SessionID: storedSession.ID(),
			AppName:   storedSession.AppName(),
			UserID:    storedSession.UserID(),
		}
	}

	var memoryImpl agent.Memory
	if r.memoryService != nil {
		memoryImpl = &imemory.Memory{
			Service:   r.memoryService,
			SessionID: storedSession.ID(),
			UserID:    storedSession.UserID(),
			AppName:   storedSession.AppName(),
		}
	}

	ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:    artifacts,
		Memory:       memoryImpl,
		Session:      storedSession,
		Agent:        agentToRun,
		UserContent:  msg,
		RunConfig:    &cfg,
		InvocationID: resolveInvocationID(storedSession, msg),
	})
	resCtx := agent.NewNodeContext(ic, nil)
	return resCtx
}

// buildResumeResponses maps msg's function responses to interruptID ->
// payload, keeping only those that answer a still-pending interrupt.
// Returns nil for a fresh turn (nothing answered). Go analog of
// adk-python _extract_resume_inputs.
//
// Pending = interrupts waiting in RunState PLUS any unanswered
// long-running call open in history. The latter matters when one turn
// emitted several long-running calls: RunState records only one as
// waiting, but the human may answer the others; matching open history
// calls lets each answer drive a resume until all are resolved.
func buildResumeResponses(msg *genai.Content, state *workflow.RunState, sess session.Session) map[string]any {
	if msg == nil {
		return nil
	}
	pending := map[string]struct{}{}
	if state != nil {
		for id := range waitingInterruptIDs(state) {
			pending[id] = struct{}{}
		}
	}
	for id := range openLongRunningCallIDs(sess) {
		pending[id] = struct{}{}
	}
	if len(pending) == 0 {
		return nil
	}

	var out map[string]any
	for _, fr := range utils.FunctionResponses(msg) {
		if fr == nil || fr.ID == "" {
			continue
		}
		if _, ok := pending[fr.ID]; !ok {
			continue
		}
		if out == nil {
			out = map[string]any{}
		}
		// Opaque payload; Workflow.Resume validates against any schema.
		out[fr.ID] = decodeResumeResponse(fr)
	}
	return out
}

// openLongRunningCallIDs returns the set of long-running tool call IDs in
// session history that do not yet have a matching FunctionResponse, i.e.
// the interrupts still awaiting a human answer.
func openLongRunningCallIDs(sess session.Session) map[string]struct{} {
	open := map[string]struct{}{}
	if sess == nil {
		return open
	}
	answered := map[string]struct{}{}
	events := sess.Events()
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		for _, id := range ev.LongRunningToolIDs {
			open[id] = struct{}{}
		}
		for _, fr := range utils.FunctionResponses(ev.Content) {
			if fr != nil && fr.ID != "" {
				answered[fr.ID] = struct{}{}
			}
		}
	}
	for id := range answered {
		delete(open, id)
	}
	return open
}

// decodeResumeResponse extracts the user payload from a function
// response, accepting {"response": v}, {"payload": v}, or the raw map. A
// string under "response" is JSON-parsed when possible.
func decodeResumeResponse(fr *genai.FunctionResponse) any {
	if fr.Response == nil {
		return nil
	}
	if raw, ok := fr.Response["response"]; ok {
		if s, isStr := raw.(string); isStr {
			var decoded any
			if err := json.Unmarshal([]byte(s), &decoded); err == nil {
				return decoded
			}
			return s
		}
		return raw
	}
	if payload, ok := fr.Response["payload"]; ok {
		return payload
	}
	return fr.Response
}

// waitingInterruptIDs returns the set of interrupt IDs for every node
// in state that is currently paused on a long-running interrupt.
func waitingInterruptIDs(state *workflow.RunState) map[string]struct{} {
	ids := map[string]struct{}{}
	for _, ns := range state.Nodes {
		if ns == nil || ns.Status != workflow.NodeWaiting {
			continue
		}
		for _, id := range ns.Interrupts {
			if id != "" {
				ids[id] = struct{}{}
			}
		}
	}
	return ids
}

// resolveInvocationID reuses the paused run's invocation ID when msg is a
// HITL resume (carries a function response), so the resume turn and the
// pause it answers share one ID and rehydration can scope to a single
// run. Returns "" for a fresh turn or when the answered call is not in
// history, letting the caller mint a fresh ID. Go analog of adk-python
// Runner._resolve_invocation_id.
func resolveInvocationID(sess session.Session, msg *genai.Content) string {
	if sess == nil || msg == nil {
		return ""
	}
	for _, fr := range utils.FunctionResponses(msg) {
		if fr == nil || fr.ID == "" {
			continue
		}
		if ev := findEventByFunctionCallID(sess, fr.ID); ev != nil {
			return ev.InvocationID
		}
	}
	return ""
}

// findEventByFunctionCallID returns the most recent event whose content
// carries a FunctionCall with the given id, or nil. Go analog of
// adk-python functions.find_event_by_function_call_id (reverse scan).
func findEventByFunctionCallID(sess session.Session, id string) *session.Event {
	if sess == nil || id == "" {
		return nil
	}
	events := sess.Events()
	for i := events.Len() - 1; i >= 0; i-- {
		ev := events.At(i)
		if ev == nil {
			continue
		}
		for _, fc := range utils.FunctionCalls(utils.Content(ev)) {
			if fc != nil && fc.ID == id {
				return ev
			}
		}
	}
	return nil
}
