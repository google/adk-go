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

// runNode is the Go equivalent of adk-python Runner._run_node_async. It
// drives an LlmAgent (agentToRun) through the workflow engine by wrapping
// it in a node and running a single-node workflow (START -> node),
// reusing the same session/plugin/persist pipeline as the agent path.
//
// Detection and wrapping are automatic (see Run); the user does not
// configure anything. The wrapping node bridges the LlmAgent's own HITL
// (long-running tools / tool confirmation, which emit LongRunningToolIDs)
// into a workflow pause; see llmAgentNode.
//
// The workflow scheduler owns the per-node goroutines and the event
// channel (the analog of Python's ic.event_queue), and cancels in-flight
// nodes when the caller stops consuming the returned iterator. There is
// therefore no need for a done-sentinel or an explicit cleanup task here.
func (r *Runner) runNode(
	ctx context.Context,
	storedSession session.Session,
	agentToRun agent.Agent,
	msg *genai.Content,
	cfg agent.RunConfig,
	opts runOptions,
	yield func(*session.Event, error) bool,
) {
	// Wrap the LlmAgent in the HITL-bridging node and build a single-node
	// workflow (START -> node). The workflow name must be stable and
	// non-empty so its RunState persists across turns for HITL resume; we
	// derive it from the app name and the agent name.
	node := newLlmAgentNode(agentToRun)
	wf, err := workflow.New(rootWorkflowName(r.appName, agentToRun), []workflow.Edge{
		{From: workflow.Start, To: node},
	})
	if err != nil {
		yield(nil, fmt.Errorf("failed to build node workflow: %w", err))
		return
	}

	// 1. Build the invocation context. UserContent carries the user
	// message, which Workflow.Run reads as the workflow's seed input
	// (workflow.userInput); this mirrors Python's node_input.
	ictx := r.newNodeInvocationContext(ctx, storedSession, agentToRun, msg, cfg)

	// Append the incoming user message (fresh text or the function-
	// response reply) to the session for history. Same helper the agent
	// path uses; it also runs the on_user_message plugin callback.
	ictx, userEvent, err := r.appendMessageToSession(ictx, storedSession, msg, cfg.SaveInputBlobsAsArtifacts, r.pluginManager, opts.stateDelta)
	if err != nil {
		yield(nil, err)
		return
	}
	// Optionally yield the user message event before any node events,
	// mirroring adk-python's yield_user_message.
	if opts.yieldUserMessage && userEvent != nil {
		if !yield(userEvent, nil) {
			return
		}
	}

	// 2. Plugin lifecycle: defer after_run, run before_run and honor an
	// early-exit decision. Mirrors the agent path.
	pluginManager := r.pluginManager
	if pluginManager != nil {
		defer pluginManager.RunAfterRunCallback(ictx)

		earlyExitResult, err := pluginManager.RunBeforeRunCallback(ictx)
		if earlyExitResult != nil || err != nil {
			earlyExitEvent := session.NewEvent(ictx.InvocationID())
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

	// 3. Choose the producer: Resume (HITL continuation) vs Run (fresh).
	// We treat the turn as a resume only when a function response in msg
	// matches a node that is currently waiting in the persisted RunState.
	state, err := workflow.LoadRunState(storedSession, wf.Name())
	if err != nil {
		yield(nil, fmt.Errorf("failed to load workflow run state: %w", err))
		return
	}

	var events iter.Seq2[*session.Event, error]
	if responses := buildResumeResponses(msg, state, storedSession); len(responses) > 0 {
		events = wf.Resume(ictx, state, responses)
	} else {
		events = wf.Run(ictx)
	}

	// 4. Consume the workflow's event stream. This is the same loop the
	// agent path runs (the equivalent of Python's _consume_event_queue):
	// run the on_event plugin callback, persist non-partial events, and
	// yield. The RunState StateDelta event emitted by Run/Resume flows
	// through the same persist branch, so HITL pause state is saved here.
	for event, evErr := range events {
		if evErr != nil {
			if !yield(nil, evErr) {
				return
			}
			continue
		}

		// Stamp the agent name as author on internal control events that
		// the workflow engine emits without one (e.g. the RunState
		// StateDelta event from Run/Resume). Otherwise a later turn's
		// findAgentToRun would scan these events and log a spurious
		// "Event from an unknown agent" warning, since an empty author
		// resolves to no agent in the tree.
		if event.Author == "" {
			event.Author = agentToRun.Name()
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

		// Only commit non-partial events to the session service.
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

// newNodeInvocationContext builds the InvocationContext for the node
// path. It mirrors the agent path's context wiring: the parent map is
// attached so the wrapped LlmAgent can transfer / resolve tools, and the
// resolved agent is set so the flow runs against it.
func (r *Runner) newNodeInvocationContext(
	ctx context.Context,
	storedSession session.Session,
	agentToRun agent.Agent,
	msg *genai.Content,
	cfg agent.RunConfig,
) agent.InvocationContext {
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

	return icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:   artifacts,
		Memory:      memoryImpl,
		Session:     storedSession,
		Agent:       agentToRun,
		UserContent: msg,
		RunConfig:   &cfg,
	})
}

// buildResumeResponses maps each function response in msg to its
// InterruptID -> payload, but only for responses that answer a
// still-pending long-running call. This is the Go analog of adk-python's
// _extract_resume_inputs.
//
// A turn is a HITL resume when msg carries a FunctionResponse whose ID
// matches a long-running tool call that the node paused on. We accept a
// match against EITHER the single interrupt currently recorded as waiting
// in RunState OR any unanswered long-running call still open in session
// history. The latter is essential when the model emitted more than one
// long-running call in a turn: the workflow scheduler can only record one
// RequestedInput per activation, so RunState.waiting tracks just one of
// them, but the human may answer the others. Matching against open
// history calls lets each answer drive a resume, after which the node
// re-runs and re-pauses on whatever is still unanswered until all are
// resolved.
//
// Returns nil when the turn answers no pending interrupt (a fresh turn).
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
		// Pass the response payload through opaquely. Workflow.Resume
		// validates it against the request's ResponseSchema if any.
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

// decodeResumeResponse extracts the user-supplied payload from a function
// response. It accepts the shapes the console launcher and other ADK
// runtimes produce: {"response": <value>}, {"payload": <any>}, or the raw
// Response map. A string under "response" is parsed as JSON when possible.
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

// waitingInterruptIDs returns the set of InterruptIDs for every node in
// state that is currently paused on a human-input request.
func waitingInterruptIDs(state *workflow.RunState) map[string]struct{} {
	ids := map[string]struct{}{}
	for _, ns := range state.Nodes {
		if ns == nil {
			continue
		}
		if ns.Status == workflow.NodeWaiting && ns.PendingRequest != nil && ns.PendingRequest.InterruptID != "" {
			ids[ns.PendingRequest.InterruptID] = struct{}{}
		}
	}
	return ids
}

// llmAgentNode wraps an LlmAgent as a workflow node and bridges the
// LlmAgent's own HITL into a workflow pause.
//
// adk-go currently has two separate HITL signals: workflow RequestedInput
// (which pauses the scheduler) and the LlmAgent's LongRunningToolIDs
// (which does not). adk-python unifies them on long_running_tool_ids;
// until adk-go does the same in the workflow core (see
// runner/DESIGN_run_node.md section 10.5), this node performs the bridge
// locally: when the wrapped agent emits an event with LongRunningToolIDs,
// the node also yields a RequestedInput so the scheduler pauses and
// persists RunState. On resume the node re-runs (RerunOnResume) and feeds
// the human's reply back to the agent as a function response.
type llmAgentNode struct {
	workflow.BaseNode
	agent agent.Agent
}

// rerunOnResume is referenced by value via &rerunOnResume below.
var rerunOnResume = true

func newLlmAgentNode(a agent.Agent) *llmAgentNode {
	return &llmAgentNode{
		// RerunOnResume puts the node in re-entry mode: on resume the node
		// re-runs and reads the reply via ctx.ResumedInput, rather than
		// handing the reply to a successor (this node has none).
		BaseNode: workflow.NewBaseNode(a.Name(), a.Description(), workflow.NodeConfig{
			RerunOnResume: &rerunOnResume,
		}),
		agent: a,
	}
}

func (n *llmAgentNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		// The node's input is the runner's user message: fresh text on a
		// new turn, or the human's FunctionResponse(s) on a HITL resume.
		// The runner already appended it to the session, and the LlmAgent
		// flow's contents processor pairs a resume FunctionResponse with
		// the pending long-running call by ID (replacing the original
		// "pending" response). So we feed the message straight through —
		// we do NOT synthesize a second FunctionResponse, which would
		// duplicate it in history and confuse the model.
		userContent := inputToUserContent(input)

		// Interrupts answered by this turn: any long-running call ID for
		// which the message carries a matching FunctionResponse. We must
		// not re-pause for these even though the agent's history still
		// shows the original long-running call.
		resolved := resolvedInterruptIDs(userContent)

		agentCtx := n.newAgentContext(ctx, userContent)

		var pending []string // unresolved long-running interrupt IDs seen this run
		for event, err := range n.agent.Run(agentCtx) {
			if err != nil {
				yield(nil, err)
				return
			}
			synthesizeNodeOutput(event)
			for _, id := range event.LongRunningToolIDs {
				if !resolved[id] {
					pending = append(pending, id)
				}
			}
			if !yield(event, nil) {
				return
			}
		}

		// HITL bridge: if the agent emitted long-running tool calls that
		// were not answered by this turn, pause the workflow by emitting an
		// event that only sets RequestedInput (keyed by the real
		// long-running call ID). The scheduler parks the node
		// (NodeWaiting) on RequestedInput alone and persists RunState.
		//
		// We deliberately do NOT use workflow.NewRequestInputEvent: it
		// injects a synthetic "adk_request_input" FunctionCall into the
		// model conversation, which a real model rejects on resume (the
		// agent already has its own real long-running FunctionCall
		// pending). The pause event carries no model Content, so it does
		// not pollute the LLM history.
		//
		// The scheduler enforces a single RequestedInput per activation,
		// so we pause on the first unresolved interrupt. The console (and
		// other clients) still see every long-running call from the
		// agent's own events and can answer them across turns.
		if len(pending) > 0 {
			yield(newPauseEvent(ctx, pending[0]), nil)
		}
	}
}

// resolvedInterruptIDs returns the set of FunctionResponse IDs carried by
// content, i.e. the long-running interrupts the current turn answers.
func resolvedInterruptIDs(content *genai.Content) map[string]bool {
	if content == nil {
		return nil
	}
	out := map[string]bool{}
	for _, p := range content.Parts {
		if p.FunctionResponse != nil && p.FunctionResponse.ID != "" {
			out[p.FunctionResponse.ID] = true
		}
	}
	return out
}

// newPauseEvent builds a content-free event that only sets RequestedInput,
// so the scheduler pauses the node (NodeWaiting) and persists RunState
// without adding any synthetic FunctionCall to the model conversation. The
// InterruptID is the LlmAgent's real long-running tool call ID, so the
// follow-up FunctionResponse to that call resumes the run.
func newPauseEvent(ctx agent.InvocationContext, interruptID string) *session.Event {
	ev := session.NewEvent(ctx.InvocationID())
	if a := ctx.Agent(); a != nil {
		ev.Author = a.Name()
	}
	ev.Branch = ctx.Branch()
	ev.RequestedInput = &session.RequestInput{
		InterruptID: interruptID,
		Message:     "Awaiting response for long-running tool call.",
	}
	return ev
}

// newAgentContext builds the per-agent InvocationContext, mirroring
// workflow.AgentNode: it inherits services and branch from ctx and runs
// the wrapped agent with the given user content.
func (n *llmAgentNode) newAgentContext(ctx agent.InvocationContext, userContent *genai.Content) agent.InvocationContext {
	return icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:     ctx.Artifacts(),
		Memory:        ctx.Memory(),
		Session:       ctx.Session(),
		Branch:        ctx.Branch(),
		Agent:         n.agent,
		UserContent:   userContent,
		RunConfig:     ctx.RunConfig(),
		EndInvocation: ctx.Ended(),
		InvocationID:  ctx.InvocationID(),
	})
}

// inputToUserContent converts a node input value into a user Content for
// the wrapped agent.
func inputToUserContent(input any) *genai.Content {
	switch v := input.(type) {
	case nil:
		return nil
	case *genai.Content:
		return v
	case string:
		if v == "" {
			return nil
		}
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: v}}}
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return nil
		}
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: string(b)}}}
	}
}

// synthesizeNodeOutput sets Event.Output from a final model text response
// so the node produces an output value (mirrors workflow.AgentNode's
// synthesizeAgentOutput).
func synthesizeNodeOutput(event *session.Event) {
	if event == nil || event.Output != nil {
		return
	}
	if !event.IsFinalResponse() {
		return
	}
	content := event.LLMResponse.Content
	if content == nil || content.Role != "model" {
		return
	}
	var b []byte
	for _, p := range content.Parts {
		if p == nil || p.Text == "" || p.Thought {
			continue
		}
		b = append(b, p.Text...)
	}
	if len(b) == 0 {
		return
	}
	event.Output = string(b)
}
