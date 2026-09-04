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
	"fmt"
	"iter"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
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
}

// New creates a new Workflow agent. A single returned agent
// instance can serve many concurrent sessions: the per-invocation
// run state lives in session.State, not on the agent. A paused
// workflow resumes on a follow-up turn when the user submits a
// FunctionResponse targeting the InterruptID emitted by the
// paused node.
func New(cfg Config) (agent.Agent, error) {
	w, err := workflow.New(cfg.Name, cfg.Edges)
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
// a FunctionResponse that answers an interrupt this run can still
// act on — see detectResume. The workflow's RunState lives in
// session.State, not on this struct, so a single *workflowAgent
// safely services many concurrent sessions.
type workflowAgent struct {
	workflow *workflow.Workflow
}

// run is the agent.Config.Run callback. It dispatches between
// Workflow.Resume (when the inbound user content answers one of
// this run's open interrupts) and Workflow.Run (every other turn).
func (a *workflowAgent) run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		responses, state, ok, err := a.detectResume(ctx)
		if err != nil {
			yield(nil, err)
			return
		}
		if ok {
			exCtx := agent.Promote(ctx)
			for ev, err := range a.workflow.Resume(exCtx, state, responses) {
				if !yield(ev, err) {
					return
				}
			}
			return
		}
		for ev, err := range a.workflow.Run(ctx) {
			if !yield(ev, err) {
				return
			}
		}
	}
}

// detectResume inspects the inbound user message for FunctionResponses
// that answer a paused node's long-running interrupt. Returns the
// responses map keyed by InterruptID (suitable for Workflow.Resume),
// the RunState loaded from session, and true if this turn is a resume;
// (nil, nil, false) for a fresh turn.
func (a *workflowAgent) detectResume(ctx agent.InvocationContext) (map[string]any, *workflow.RunState, bool, error) {
	frs := utils.FunctionResponses(ctx.UserContent())
	if len(frs) == 0 {
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

	// Key by interrupt ID, not function name: an agent node pauses on a
	// tool's long-running request (adk_request_credential /
	// adk_request_confirmation), not just the workflow's own
	// adk_request_input, and the engine's pause is name-agnostic
	// (Event.LongRunningToolIDs).
	//
	// A response only routes this turn to Resume when it is actually aimed at
	// this run: either its ID is an interrupt the run knows about, or it is
	// explicitly an answer to the workflow's own input request. Otherwise the
	// turn falls through to a fresh Run, as it did before ID-keyed matching —
	// a turn may legitimately carry both text and an unrelated tool reply, and
	// routing that to Resume would fail it with ErrNothingToResume and discard
	// the text. Mirrors runner.buildResumeResponses, which filters the same way.
	//
	// A wrong-but-deliberate answer (right name, unknown ID) still reaches
	// Resume, so the caller keeps the ErrNothingToResume diagnostic.
	known := knownInterruptIDs(state)
	responses := map[string]any{}
	for _, fr := range frs {
		if fr == nil || fr.ID == "" {
			continue
		}
		if _, ok := known[fr.ID]; !ok && fr.Name != workflow.WorkflowInputFunctionCallName {
			continue
		}
		responses[fr.ID] = utils.UnwrapResponse(fr.Response)
	}
	if len(responses) == 0 {
		return nil, nil, false, nil
	}

	return responses, state, true, nil
}

// knownInterruptIDs collects the interrupt IDs of nodes the rehydrated run can
// still act on — those waiting for an answer, and those a re-entry node is
// about to be re-run with. A node that already settled is excluded, so a reply
// to an interrupt this run has finished with no longer counts as a resume.
//
// ResumedInputs is included, not just Interrupts: the runner appends the reply
// to the session before the agent runs, so on a genuine first resume the
// interrupt is already resolved and rehydration hands it back on the node.
func knownInterruptIDs(state *workflow.RunState) map[string]struct{} {
	ids := map[string]struct{}{}
	for _, ns := range state.Nodes {
		if ns == nil || (ns.Status != workflow.NodeWaiting && ns.Status != workflow.NodePending) {
			continue
		}
		for _, id := range ns.Interrupts {
			if id != "" {
				ids[id] = struct{}{}
			}
		}
		for id := range ns.ResumedInputs {
			if id != "" {
				ids[id] = struct{}{}
			}
		}
	}
	return ids
}
