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

// Package workflowagent adapts a workflow.Workflow (a node/graph
// workflow) to the agent.Agent interface, so a graph can be run by a
// runner.Runner like any other agent.
package workflowagent

import (
	"encoding/json"
	"fmt"
	"iter"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// Config is the configuration for creating a new Workflow agent.
type Config struct {
	Name                 string
	Description          string
	SubAgents            []agent.Agent
	BeforeAgentCallbacks []agent.BeforeAgentCallback
	AfterAgentCallbacks  []agent.AfterAgentCallback
	Edges                []workflow.Edge

	// InputSchema, when non-nil, validates the workflow's input. See workflow.WithInputSchema.
	InputSchema *jsonschema.Resolved
	// OutputSchema, when non-nil, validates the workflow's terminal output. See workflow.WithOutputSchema.
	OutputSchema *jsonschema.Resolved
}

// New creates a new Workflow agent. A single returned agent
// instance can serve many concurrent sessions: the per-invocation
// run state is rebuilt from session event history on a resume turn
// rather than held on the agent. A paused
// workflow resumes on a follow-up turn when the user submits a
// FunctionResponse targeting the InterruptID emitted by the
// paused node.
//
// cfg.InputSchema and cfg.OutputSchema are passed to the workflow as
// [workflow.WithInputSchema] and [workflow.WithOutputSchema]: a run whose
// input fails the input schema yields an error wrapping
// [workflow.ErrInputValidation] before any node runs, and a run whose
// terminal output fails the output schema ends with an error wrapping
// [workflow.ErrNodeFailed].
//
// In a resumable run (runner.Config.Resumable) the agent yields one
// end-of-agent event, authored by itself, after the workflow has finished with
// no node waiting for input. It yields none when the run is not resumable, a
// node failed, the caller stopped reading, or a node still waits for a
// long-running call or an input request, including one that has waited since
// an earlier turn.
//
// The resume behavior described above holds only when this agent is the run's
// root; in every other tree, the runner decides which agent runs a resumed turn
// (see runner.WithInvocationID and Runner.Run), which may leave this agent's
// remaining work undone.
// Resuming an invocation (runner.WithInvocationID, or a function response to a
// paused call or an input request) is not supported when a graph workflow
// agent, made by workflowagent.New, runs below a ParallelAgent,
// SequentialAgent or LoopAgent, or when a graph workflow agent, ParallelAgent,
// SequentialAgent or LoopAgent runs as an agent node (workflow.AgentNode) of a
// graph workflow agent: in those trees the resume behavior described above
// does not apply, and a resumed turn may, for example, fail, run nodes or
// sub-agents again, or end before every branch has finished.
func New(cfg Config) (agent.Agent, error) {
	w, err := workflow.New(cfg.Name, cfg.Edges,
		workflow.WithInputSchema(cfg.InputSchema),
		workflow.WithOutputSchema(cfg.OutputSchema))
	if err != nil {
		return nil, err
	}

	wa := &workflowAgent{workflow: w}

	wfAgent, err := agent.New(agent.Config{
		Name:                 cfg.Name,
		Description:          cfg.Description,
		SubAgents:            cfg.SubAgents,
		BeforeAgentCallbacks: cfg.BeforeAgentCallbacks,
		AfterAgentCallbacks:  cfg.AfterAgentCallbacks,
		Run:                  wa.run,
	})
	if err != nil {
		return nil, err
	}

	// Tag the agent state so the telemetry layer can emit
	// "invoke_workflow <name>" spans instead of "invoke_agent <name>"
	// for workflow-backed agents.
	internalAgent, ok := wfAgent.(agentinternal.Agent)
	if !ok {
		return nil, fmt.Errorf("internal error: failed to convert to internal agent")
	}
	state := agentinternal.Reveal(internalAgent)
	state.AgentType = agentinternal.TypeWorkflowAgent
	state.Config = cfg

	return wfAgent, nil
}

// workflowAgent is the wrapper that dispatches between
// Workflow.Run (fresh turn) and Workflow.Resume (resume turn).
// The dispatch decision is made by inspecting ctx.UserContent for
// a FunctionResponse targeting a previously-emitted RequestInput.
// The workflow's RunState is reconstructed from session event
// history on a resume turn, rather than held on this struct, so a
// single *workflowAgent safely services many concurrent sessions.
type workflowAgent struct {
	workflow *workflow.Workflow
}

// run is the agent.Config.Run callback. It dispatches between
// Workflow.Resume (when the inbound user content carries a
// FunctionResponse to a previously-emitted RequestInput) and
// Workflow.Run (every other turn).
//
// In a resumable run it ends a workflow that finished with no error and
// no node waiting on an interrupt with the agent's end-of-agent event, as
// adk-python's Workflow does (workflow/_workflow.py), so that a resumable
// ParallelAgent above it skips it on resume. The scheduler parks a node on
// an interrupt for each event with LongRunningToolIDs, so the stream tells
// whether a node of this turn waits; after a Resume, state also holds the
// nodes that wait from an earlier turn.
func (a *workflowAgent) run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		responses, state, ok, err := a.detectResume(ctx)
		if err != nil {
			yield(nil, err)
			return
		}
		var events iter.Seq2[*session.Event, error]
		if ok {
			events = a.workflow.Resume(agent.Promote(ctx), state, responses)
		} else {
			events = a.workflow.Run(ctx)
		}
		finished := true
		for ev, err := range events {
			if err != nil || ev != nil && len(ev.LongRunningToolIDs) > 0 {
				finished = false
			}
			if !yield(ev, err) {
				return
			}
		}
		if rc := runconfig.FromContext(ctx); finished && rc != nil && rc.Resumable && !hasWaitingNode(state) {
			yield(llminternal.NewEndOfAgentEvent(ctx, ctx.Agent().Name()), nil)
		}
	}
}

// hasWaitingNode reports whether state, which a Resume run updated in place,
// holds a node that still waits on an interrupt. A nil state has none.
//
// A node parked by WaitForOutput is NodeWaiting with no interrupts and does
// not count, as in adk-python's _collect_remaining_interrupts
// (workflow/_workflow.py).
func hasWaitingNode(state *workflow.RunState) bool {
	if state == nil {
		return false
	}
	for _, ns := range state.Nodes {
		if ns.Status == workflow.NodeWaiting && len(ns.Interrupts) > 0 {
			return true
		}
	}
	return false
}

// detectResume inspects the inbound user message for FunctionResponses
// targeting a previously-emitted RequestInput. Returns the
// responses map keyed by InterruptID (suitable for
// Workflow.Resume), the RunState reconstructed from session history,
// and true if this turn is a resume; (nil, nil, false) for a fresh
// turn.
func (a *workflowAgent) detectResume(ctx agent.InvocationContext) (map[string]any, *workflow.RunState, bool, error) {
	frs := utils.FunctionResponses(ctx.UserContent())
	if len(frs) == 0 {
		return nil, nil, false, nil
	}

	responses := map[string]any{}
	for _, fr := range frs {
		if fr.Name != workflow.WorkflowInputFunctionCallName {
			continue
		}
		responses[fr.ID] = decodeWorkflowInputResponse(fr)
	}
	if len(responses) == 0 {
		return nil, nil, false, nil
	}

	// Scope rehydration to this run's invocation (the runner reuses the
	// paused run's ID on resume) so a prior completed run in the same
	// session does not leak in.
	state, err := a.workflow.ReconstructRunState(ctx.Session(), ctx.InvocationID())
	if err != nil {
		// A bad resume (e.g. failed schema validation) must fail,
		// not silently fall through to a fresh Run.
		return nil, nil, false, err
	}
	if state == nil {
		return nil, nil, false, nil
	}
	// The runner has already persisted this turn's answers. Rehydration
	// unwraps and validates them; the ingress decoder must not replace
	// those values with a differently decoded copy of the same answer.
	for _, ns := range state.Nodes {
		for id, response := range ns.ResumedInputs {
			if _, matched := responses[id]; matched {
				responses[id] = response
			}
		}
	}

	return responses, state, true, nil
}

// decodeWorkflowInputResponse extracts the user-supplied payload
// from a FunctionResponse targeting a workflow input request.
//
// Three accepted shapes, in priority order:
//
//  1. {"response": <value>}  — when value is a string, it is
//     parsed as JSON and the result returned; if the string is
//     not valid JSON it is returned verbatim. When value is any
//     other type, it is returned as-is.
//  2. {"payload": <any>}     — value returned verbatim.
//  3. anything else           — the whole Response map is returned.
func decodeWorkflowInputResponse(fr *genai.FunctionResponse) any {
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
