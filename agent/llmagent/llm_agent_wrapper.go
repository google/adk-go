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

// Package workflowinternal provides utilities for running an LlmAgent as
// a workflow node. Per-mode behaviour:
//
//   - single_turn: the wrapper forces IncludeContents=none, seeds the
//     agent with a single user-content event derived from nodeInput,
//     drives one Agent.Run, post-processes the model reply into the
//     terminal Output, then returns.
//   - task: the wrapper drives Agent.Run and watches for the
//     finish_task FunctionCall; once the matching FinishTaskTool
//     FunctionResponse signals success, the wrapper promotes the FC
//     args (or the wrapped value) as the terminal Output and returns.
//     Non-success FRs let the LLM see the validation error and retry.
//   - chat: the wrapper runs an outer dispatch loop. Before re-entering
//     Agent.Run on each iteration it scans the session for unresolved
//     task delegations (task FCs from this coordinator without a
//     matching FR), dispatches each via workflow.RunNode under a
//     stable WithRunID(fc.ID), and synthesises a user-role FR event so
//     the LLM sees the task result on the next round. The loop ends
//     when the LLM finishes without delegating. transfer_to_agent is
//     handled in-process by llmagent.Run (forwarded through the same
//     iterator via base_flow.go:639-651) so a single runner.Run call
//     returns both the coordinator's transfer event AND the
//     transferred-to sub-agent's first response. See the package
//     comment block in runChat below for why this differs from
//     adk-python's "exit-and-re-pick-next-turn" model.
package llmagent

import (
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/internal/workflowinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/workflow"
)

// RunLLMAgentAsNode runs an LlmAgent as a workflow node.
func RunLLMAgentAsNode(a agent.Agent, ctx agent.InvocationContext, nodeInput any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmA, ok := a.(llminternal.Agent)
		if !ok || llmA == nil {
			yield(nil, fmt.Errorf("RunLLMAgentAsNode: %q is not an LlmAgent", a.Name()))
			return
		}
		state := llminternal.Reveal(llmA)

		if state.Mode == llminternal.ModeUnset {
			state.Mode = llminternal.ModeSingleTurn
		}
		switch state.Mode {
		case llminternal.ModeTask, llminternal.ModeSingleTurn, llminternal.ModeChat:
		default:
			yield(nil, fmt.Errorf("RunLLMAgentAsNode: LlmAgent %q only supports task, single_turn, and chat mode, got %q",
				a.Name(), state.Mode))
			return
		}

		if state.Mode == llminternal.ModeSingleTurn {
			state.IncludeContents = "none"
		}

		if seed := PrepareLLMAgentInput(a, ctx, nodeInput); seed != nil {
			if !yield(seed, nil) {
				return
			}
		}

		// Build the per-agent InvocationContext used to drive
		// Agent.Run. The wrapper:
		//   - rebinds Agent to a (matching adk-python's ic.agent=agent),
		//   - for task/single_turn modes, threads isolation_scope so the
		//     content processor filters session events to this scope only
		//     (chat coordinators stay unscoped and see the full
		//     conversation).
		//   - overrides UserContent for task and single_turn modes so the
		//     content builder has a first-turn fallback when the session
		//     does not (yet) carry one,
		//   - relies on the embedded InvocationContext for everything
		//     else (memory, run config, etc.).
		userContent := ctx.UserContent()
		if (state.Mode == llminternal.ModeTask || state.Mode == llminternal.ModeSingleTurn) && nodeInput != nil {
			userContent = nodeInputToContent(nodeInput)
		}
		isolationScope := ""
		if state.Mode == llminternal.ModeTask || state.Mode == llminternal.ModeSingleTurn {
			isolationScope = ctx.IsolationScope()
		}
		ic := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
			Artifacts:      ctx.Artifacts(),
			Memory:         ctx.Memory(),
			Session:        ctx.Session(),
			Branch:         ctx.Branch(),
			IsolationScope: isolationScope,
			Agent:          a,
			UserContent:    userContent,
			RunConfig:      ctx.RunConfig(),
			InvocationID:   ctx.InvocationID(),
		})

		switch state.Mode {
		case llminternal.ModeSingleTurn:
			runSingleTurn(a, ic, yield)
		case llminternal.ModeChat:
			runChat(a, ic, yield)
		case llminternal.ModeTask:
			runTask(a, ic, yield)
		}
	}
}

// PrepareLLMAgentInput returns the seeded user-role event for the
// single_turn agent's first turn.
func PrepareLLMAgentInput(a agent.Agent, ctx agent.InvocationContext, nodeInput any) *session.Event {
	if nodeInput == nil {
		return nil
	}
	llmA, ok := a.(llminternal.Agent)
	if !ok || llmA == nil {
		return nil
	}
	if llminternal.Reveal(llmA).Mode != llminternal.ModeSingleTurn {
		return nil
	}
	content := nodeInputToContent(nodeInput)
	if content == nil {
		return nil
	}
	ev := session.NewEvent(ctx.InvocationID())
	ev.Author = "user"
	ev.LLMResponse = model.LLMResponse{Content: content}
	if iso := ctx.IsolationScope(); iso != "" {
		ev.IsolationScope = iso
	}
	return ev
}

// ProcessLLMAgentOutput post-processes a model-authored event from the
// LlmAgent.
//
// On the agent's final text turn (no function calls, not partial,
// role=model) we:
//   - parse the text against the agent's OutputSchema if set, else
//     keep the raw text;
//   - record an OutputKey state-delta on the event so the runner
//     persists it (Go writes to ev.Actions.StateDelta; Python writes
//     to ctx.actions.state_delta — the runner reconciles the
//     per-event delta into session state on AppendEvent, so the two
//     paths converge);
//   - stash the parsed value on event.Output for the node's terminal
//     output.
//
// Returns an error iff OutputSchema validation fails on a non-empty
// model reply; the caller (runSingleTurn) propagates it.
func ProcessLLMAgentOutput(a agent.Agent, ev *session.Event) error {
	if ev == nil {
		return nil
	}
	if len(utils.FunctionCalls(ev.Content)) > 0 {
		return nil
	}
	if ev.Partial {
		return nil
	}
	if ev.Content == nil || ev.Content.Role != "model" {
		return nil
	}
	llmA, ok := a.(llminternal.Agent)
	if !ok || llmA == nil {
		return nil
	}
	state := llminternal.Reveal(llmA)

	// Merge non-thought text parts; mirrors adk-python's
	// (p.text for p in parts if p.text and not p.thought) filter.
	var b strings.Builder
	for _, p := range ev.Content.Parts {
		if p == nil || p.Thought {
			continue
		}
		b.WriteString(p.Text)
	}
	text := b.String()

	var output any
	if state.OutputSchema != nil {
		parsed, err := utils.ValidateOutputSchema(text, state.OutputSchema)
		if err != nil {
			// TODO: we should return error, once output_schema with tools is supported. As right now there's no guarantee that the actual LLM output will be compliant with set output_schema.
			//log.Default().Printf("LlmAgent %q output validation failed: %v", a.Name(), err)
			output = text
		} else {
			output = parsed
		}
	} else {
		output = text
	}

	if state.OutputKey != "" && output != nil {
		if ev.Actions.StateDelta == nil {
			ev.Actions.StateDelta = map[string]any{}
		}
		ev.Actions.StateDelta[state.OutputKey] = output
	}

	ev.Output = output

	if ev.NodeInfo == nil {
		ev.NodeInfo = &session.NodeInfo{}
	}
	ev.NodeInfo.MessageAsOutput = true
	return nil
}

// extractFinishTaskFC returns the finish_task FunctionCall
func extractFinishTaskFC(ev *session.Event) *genai.FunctionCall {
	if ev == nil {
		return nil
	}
	for _, fc := range utils.FunctionCalls(ev.Content) {
		if fc != nil && fc.Name == workflowinternal.FinishTaskToolName {
			return fc
		}
	}
	return nil
}

// isFinishTaskSuccessFR reports whether this event is the successful
// FunctionResponse from FinishTaskTool.
// The first finish_task FR decides. A non-success FR (e.g.
// validation error) returns false so the caller keeps iterating and
// the LLM gets a chance to retry.
func isFinishTaskSuccessFR(ev *session.Event) bool {
	if ev == nil {
		return false
	}
	for _, fr := range utils.FunctionResponses(ev.Content) {
		if fr == nil || fr.Name != workflowinternal.FinishTaskToolName {
			continue
		}
		if fr.Response == nil {
			return false
		}
		v, ok := fr.Response["result"]
		if !ok {
			return false
		}
		s, ok := v.(string)
		return ok && s == workflowinternal.FinishTaskSuccessResult
	}
	return false
}

// extractTaskDelegationFCs returns task-delegation FCs in this event.
// A task-delegation FC is one whose tool is a TaskAgentTool.
func extractTaskDelegationFCs(ev *session.Event, toolsDict map[string]tool.Tool) []*genai.FunctionCall {
	if ev == nil {
		return nil
	}
	var out []*genai.FunctionCall
	for _, fc := range utils.FunctionCalls(ev.Content) {
		if fc == nil || fc.ID == "" {
			continue
		}
		if isTaskDelegationTool(toolsDict, fc.Name) {
			out = append(out, fc)
		}
	}
	return out
}

// findUnresolvedTaskDelegations walks session events and returns task FCs
// from owner without a matching FR.
//
// Sequential dispatch means at most one unresolved task delegation at a
// time, but we return a slice so the caller can iterate uniformly.
//
// A chat coordinator's conversation persists across user turns; each turn
// produces a fresh scope, so filtering by the current turn's scope would
// hide the coordinator's own FC from a prior turn. Author + tool-name
// filtering is sufficient.
func findUnresolvedTaskDelegations(sess session.Session, owner string, toolsDict map[string]tool.Tool) []*genai.FunctionCall {
	if sess == nil {
		return nil
	}
	// pendingFCs preserves discovery order (the order the LLM emitted
	// the FCs); fcOrder maps FC id → its index in pendingFCs so we can
	// drop entries when we later see a matching FR.
	var pendingFCs []*genai.FunctionCall
	fcOrder := map[string]int{}
	resolvedIDs := map[string]struct{}{}

	for ev := range sess.Events().All() {
		if ev == nil || ev.Content == nil {
			continue
		}
		if ev.Author != owner && ev.Author != "user" {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p == nil {
				continue
			}
			if fc := p.FunctionCall; fc != nil && fc.ID != "" && isTaskDelegationTool(toolsDict, fc.Name) {
				if _, seen := fcOrder[fc.ID]; !seen {
					fcOrder[fc.ID] = len(pendingFCs)
					pendingFCs = append(pendingFCs, fc)
				}
			}
			if fr := p.FunctionResponse; fr != nil && fr.ID != "" {
				resolvedIDs[fr.ID] = struct{}{}
			}
		}
	}

	out := make([]*genai.FunctionCall, 0, len(pendingFCs))
	for _, fc := range pendingFCs {
		if _, done := resolvedIDs[fc.ID]; done {
			continue
		}
		out = append(out, fc)
	}
	return out
}

func isTaskDelegationTool(toolsDict map[string]tool.Tool, name string) bool {
	t, ok := toolsDict[name]
	if !ok {
		return false
	}
	_, ok = t.(*workflowinternal.TaskAgentTool)
	return ok
}

func findFinishTaskTool(a agent.Agent) *workflowinternal.FinishTaskTool {
	llmA, ok := a.(llminternal.Agent)
	if !ok || llmA == nil {
		return nil
	}
	for _, t := range llminternal.Reveal(llmA).Tools {
		if ft, ok := t.(*workflowinternal.FinishTaskTool); ok {
			return ft
		}
	}
	return nil
}

func safeCanonicalToolsDict(a agent.Agent) map[string]tool.Tool {
	llmA, ok := a.(llminternal.Agent)
	if !ok || llmA == nil {
		return map[string]tool.Tool{}
	}
	tools := llminternal.Reveal(llmA).Tools
	out := make(map[string]tool.Tool, len(tools))
	for _, t := range tools {
		if t == nil {
			continue
		}
		if name := t.Name(); name != "" {
			out[name] = t
		}
	}
	return out
}

func dispatchTaskFC(parentAgent agent.Agent, fc *genai.FunctionCall, ctx workflow.NodeContext) (any, error) {
	if fc == nil {
		return nil, fmt.Errorf("dispatchTaskFC: nil function call")
	}
	target := parentAgent.FindAgent(fc.Name)
	if target == nil {
		return nil, fmt.Errorf("dispatchTaskFC: task target agent %q not found", fc.Name)
	}
	node, err := workflow.NewAgentNode(target, workflow.NodeConfig{})
	if err != nil {
		return nil, fmt.Errorf("dispatchTaskFC: build node for %q: %w", fc.Name, err)
	}
	out, err := workflow.RunNode[any](ctx, node, fc.Args,
		workflow.WithRunID(fc.ID),
		workflow.WithIsolationScope(fc.ID),
	)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// synthesizeTaskFREvent builds the synthesised FR event for a completed
// task delegation.
func synthesizeTaskFREvent(invocationID string, fc *genai.FunctionCall, output any) *session.Event {
	var response map[string]any
	if m, ok := output.(map[string]any); ok {
		response = m
	} else {
		response = map[string]any{"output": output}
	}
	ev := session.NewEvent(invocationID)
	ev.Author = "user"
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role: genai.RoleUser,
			Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{
					ID:       fc.ID,
					Name:     fc.Name,
					Response: response,
				},
			}},
		},
	}
	return ev
}

func nodeInputToContent(nodeInput any) *genai.Content {
	if nodeInput == nil {
		return nil
	}
	switch v := nodeInput.(type) {
	case *genai.Content:
		// Force role=user; reuse parts verbatim.
		return &genai.Content{Role: genai.RoleUser, Parts: v.Parts}
	case genai.Content:
		return &genai.Content{Role: genai.RoleUser, Parts: v.Parts}
	case string:
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: v}}}
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: fmt.Sprint(v)}}}
		}
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: string(b)}}}
	}
}

// runSingleTurn drives Agent.Run for a single round and post-processes
// each event so the terminal Output is set on the model reply.
func runSingleTurn(a agent.Agent, ic agent.InvocationContext, yield func(*session.Event, error) bool) {
	for ev, err := range a.Run(ic) {
		if err != nil {
			yield(nil, err)
			return
		}
		if perr := ProcessLLMAgentOutput(a, ev); perr != nil {
			yield(nil, perr)
			return
		}
		if !yield(ev, nil) {
			return
		}
	}
}

// runChat drives the outer dispatch loop for chat coordinators.
//
// One coordinator invocation may contain multiple LLM rounds chained
// by task delegations. Sequential delegation example:
//
//  1. Pre-LLM scan: replay any unresolved task FCs from prior turns.
//     Their dispatched sub-agents may complete or pause (HITL).
//  2. Run Agent.Run; on every fresh task FC, dispatch via RunNode and
//     synthesise an FR event so the LLM sees the result on the next
//     round.
//  3. Re-enter Agent.Run after each dispatch round; the loop ends when
//     the LLM finishes without delegating.
func runChat(a agent.Agent, ic agent.InvocationContext, yield func(*session.Event, error) bool) {
	toolsDict := safeCanonicalToolsDict(a)

	nodeCtx, ok := workflow.NodeContextFromGoContext(ic)
	if !ok {
		yield(nil, fmt.Errorf("runChat: failed to recover NodeContext from InvocationContext"))
		return
	}

	dispatchAndYield := func(fc *genai.FunctionCall) (ok bool) {
		out, err := dispatchTaskFC(a, fc, nodeCtx)
		if err != nil {
			// HITL pause propagates as ErrNodeInterrupted; bubble it
			// up so the outer scheduler can park the coordinator.
			yield(nil, err)
			return false
		}
		return yield(synthesizeTaskFREvent(ic.InvocationID(), fc, out), nil)
	}

	// Step 1: pre-LLM scan for unresolved task FCs from prior turns.
	for _, fc := range findUnresolvedTaskDelegations(nodeCtx.Session(), a.Name(), toolsDict) {
		if !dispatchAndYield(fc) {
			return
		}
	}

	// Step 2: run Agent.Run; on every fresh task FC, dispatch and
	// re-enter Agent.Run with the FR now in session.
	//
	// transfer_to_agent — deliberate divergence from adk-python.
	//
	// adk-python's wrapper (_llm_agent_wrapper.py:389-403) terminates
	// on event.actions.transfer_to_agent and (when is_resumable)
	// emits an agent-state event so the runner's next-turn picker
	// can resume the transferred-to agent on a later user turn.
	// adk-go does NOT do this — the wrapper simply forwards events
	// through, relying on base_flow.go:639-651's inline-forward to
	// run the transferred-to sub-agent within the same runner.Run
	// iterator.
	//
	// Why we diverge intentionally:
	//
	//   1. adk-go has no resumability machinery (no is_resumable,
	//      set_agent_state, or _create_agent_state_event). Mirroring
	//      Python's wrapper-exit without resumability would yield
	//      Python's non-resumable behaviour, which is: round 0
	//      emits only the transfer event; round 1's findAgentToRun
	//      walks back to find the most-recent non-user author —
	//      which is the *root agent* (who authored the transfer
	//      event) — so round 1 re-runs the root agent, NOT the
	//      transferred-to agent. The transfer becomes a "hint" the
	//      LLM has to re-decide each turn. That UX is strictly
	//      worse than the current inline-forwarding behaviour.
	//
	//   2. adk-go's inline forwarding produces a useful single-turn
	//      multi-hop UX: the user gets the transferred-to agent's
	//      response immediately. This is well-suited to streaming
	//      and to A2A multi-hop scenarios that rely on single-message
	//      hand-offs (see agent/remoteagent/v2/a2a_e2e_test.go).
	for {
		hadTaskFC := false
		for ev, err := range a.Run(ic) {
			if err != nil {
				yield(nil, err)
				return
			}
			if !yield(ev, nil) {
				return
			}
			taskFCs := extractTaskDelegationFCs(ev, toolsDict)
			for _, fc := range taskFCs {
				if !dispatchAndYield(fc) {
					return
				}
			}
			if len(taskFCs) > 0 {
				// Close this inner iteration; outer loop re-enters
				// Agent.Run so the LLM sees the synthesised FR(s).
				hadTaskFC = true
				break
			}
		}
		if !hadTaskFC {
			return
		}
	}
}

// runTask drives a task-mode LlmAgent: it sniffs the finish_task FC,
// waits for FinishTaskTool's success FR, then promotes the FC's args
// as the task output and exits.
//
// If validation fails (FR carries an "error" key), the LLM sees the
// error and retries on the next round. The finish_task tool's
// declaration mirrors the agent's OutputSchema: for wrapped primitives
// the value lives at the wrapper key; for object schemas it's at the
// top level of args.
func runTask(a agent.Agent, ic agent.InvocationContext, yield func(*session.Event, error) bool) {
	finishTool := findFinishTaskTool(a)
	var pendingFCArgs map[string]any
	for ev, err := range a.Run(ic) {
		if err != nil {
			yield(nil, err)
			return
		}
		if fc := extractFinishTaskFC(ev); fc != nil {
			// Remember the latest FC's args; wait for FinishTaskTool's
			// FR before terminating. On validation failure the FR will
			// NOT be the success message — the LLM sees the error and
			// retries.
			pendingFCArgs = maps.Clone(fc.Args)
			if !yield(ev, nil) {
				return
			}
			continue
		}
		if pendingFCArgs != nil && isFinishTaskSuccessFR(ev) {
			wrapperKey := ""
			if finishTool != nil {
				wrapperKey = finishTool.WrapperKey()
			}
			if wrapperKey != "" {
				if v, ok := pendingFCArgs[wrapperKey]; ok {
					ev.Output = v
				} else {
					ev.Output = pendingFCArgs
				}
			} else {
				ev.Output = pendingFCArgs
			}
			yield(ev, nil)
			return
		}
		if !yield(ev, nil) {
			return
		}
	}
}
