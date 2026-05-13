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

package workflowagent

import (
	"encoding/json"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
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

	return agent.New(agent.Config{
		Name:                 cfg.Name,
		Description:          cfg.Description,
		SubAgents:            cfg.SubAgents,
		BeforeAgentCallbacks: cfg.BeforeAgentCallbacks,
		AfterAgentCallbacks:  cfg.AfterAgentCallbacks,
		Run:                  wa.run,
	})
}

// workflowAgent is the wrapper that dispatches between
// Workflow.Run (fresh turn) and Workflow.Resume (resume turn).
// The dispatch decision is made by inspecting ctx.UserContent for
// a FunctionResponse targeting a previously-emitted RequestInput.
// The workflow's RunState lives in session.State, not on this
// struct, so a single *workflowAgent safely services many
// concurrent sessions.
type workflowAgent struct {
	workflow *workflow.Workflow
}

// run is the agent.Config.Run callback. It dispatches between
// Workflow.Resume (when the inbound user content carries a
// FunctionResponse to a previously-emitted RequestInput) and
// Workflow.Run (every other turn).
func (a *workflowAgent) run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if responses, state, ok := a.detectResume(ctx); ok {
			for ev, err := range a.workflow.Resume(ctx, state, responses) {
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
// targeting a previously-emitted RequestInput. Returns the
// responses map keyed by InterruptID (suitable for
// Workflow.Resume), the RunState loaded from session, and true if
// this turn is a resume; (nil, nil, false) for a fresh turn.
func (a *workflowAgent) detectResume(ctx agent.InvocationContext) (map[string]any, *workflow.RunState, bool) {
	frs := utils.FunctionResponses(ctx.UserContent())
	if len(frs) == 0 {
		return nil, nil, false
	}

	responses := map[string]any{}
	for _, fr := range frs {
		if fr.Name != workflow.WorkflowInputFunctionCallName {
			continue
		}
		responses[fr.ID] = decodeWorkflowInputResponse(fr)
	}
	if len(responses) == 0 {
		return nil, nil, false
	}

	state, err := workflow.LoadRunState(ctx.Session(), a.workflow.Name())
	if err != nil || state == nil {
		// No persisted state means there is nothing to resume;
		// fall through to a fresh Workflow.Run.
		return nil, nil, false
	}

	return responses, state, true
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
