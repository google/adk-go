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

// This file is the agent-node wrapper: it turns an LlmAgent into a
// workflow node and runs its body, forwarding the agent's events. It is
// the Go counterpart of adk-python's workflow/_llm_agent_wrapper.py and
// holds no runner-side orchestration (session/plugin/persist wiring and
// resume-input extraction live in run_node.go).

package runner

import (
	"encoding/json"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

var rerunOnResume = true

// newLlmAgentNode wraps an LlmAgent as a dynamic workflow node.
//
// It is dynamic (not static) so its body gets a NodeContext whose
// sub-scheduler can later delegate transfer_to_agent / request_task to a
// child llmAgentNode through the event queue; the current body does not
// delegate yet and forwards the agent's events verbatim.
//
// HITL rides on LongRunningToolIDs (matching adk-python): the scheduler
// parks the node (NodeWaiting) when an event carries unresolved IDs, with
// no synthetic RequestInput. On a later turn the human's FunctionResponse
// is matched back by ID and the node re-runs (RerunOnResume), continuing
// from history.
func newLlmAgentNode(a agent.Agent) workflow.Node {
	// RerunOnResume = re-entry mode: on resume the node re-runs and reads
	// the reply from history rather than handing it to a successor (it has
	// none). IN/OUT are any: seed input is the user content (nil on
	// resume), output is the agent's final text.
	//
	// EmitsOwnSpan: the wrapped agent's Run already emits an invoke_agent
	// span, so the scheduler must not add an invoke_node wrapper — the
	// span tree stays invoke_agent > generate_content like the direct
	// agent path (and adk-python's intended node dispatch), without the
	// redundant outer span.
	return workflow.NewDynamicNode[any, any](
		a.Name(),
		runLlmAgentNodeBody(a),
		workflow.NodeConfig{RerunOnResume: &rerunOnResume, EmitsOwnSpan: true},
	)
}

// runLlmAgentNodeBody returns the dynamic-node body that drives the
// wrapped agent for one activation, emitting its events through emit and
// returning the agent's final text as the node output.
func runLlmAgentNodeBody(a agent.Agent) workflow.DynamicFn[any, any] {
	return func(ctx workflow.NodeContext, input any, emit func(*session.Event) error) (any, error) {
		// On resume, input is the ORIGINAL user text; re-feeding it would
		// loop (model calls the long-running tool again). So pass no user
		// content and let the agent continue from history.
		resolved := answeredOpenInterrupts(ctx.Session())

		var userContent *genai.Content
		if len(resolved) == 0 {
			userContent = inputToUserContent(input) // fresh turn
		}

		agentCtx := newAgentContext(ctx, a, userContent)

		// Forward events verbatim. LongRunningToolIDs is the pause
		// signal: return ErrNodeInterrupted so the node parks (NodeWaiting)
		// without a terminal event. Otherwise return nil — like
		// adk-python's chat mode, a root agent sets no Output.
		paused := false
		for event, err := range a.Run(agentCtx) {
			if err != nil {
				return nil, err
			}
			if len(event.LongRunningToolIDs) > 0 {
				paused = true
			}
			if emitErr := emit(event); emitErr != nil {
				return nil, emitErr
			}
		}
		if paused {
			return nil, workflow.ErrNodeInterrupted
		}
		return nil, nil
	}
}

// answeredOpenInterrupts returns the long-running interrupt IDs that a
// FunctionResponse in history answers. Non-empty means this turn is a
// HITL resume (continue from history, don't re-process the user text).
func answeredOpenInterrupts(sess session.Session) map[string]bool {
	if sess == nil {
		return nil
	}
	longRunning := map[string]struct{}{}
	answered := map[string]bool{}
	events := sess.Events()
	for i := 0; i < events.Len(); i++ {
		ev := events.At(i)
		for _, id := range ev.LongRunningToolIDs {
			longRunning[id] = struct{}{}
		}
		for _, fr := range utils.FunctionResponses(ev.Content) {
			if fr == nil || fr.ID == "" {
				continue
			}
			if _, ok := longRunning[fr.ID]; ok {
				answered[fr.ID] = true
			}
		}
	}
	return answered
}

// newAgentContext builds the per-agent InvocationContext, inheriting
// services and branch from ctx (mirrors workflow.AgentNode).
func newAgentContext(ctx agent.InvocationContext, a agent.Agent, userContent *genai.Content) agent.InvocationContext {
	return icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Artifacts:     ctx.Artifacts(),
		Memory:        ctx.Memory(),
		Session:       ctx.Session(),
		Branch:        ctx.Branch(),
		Agent:         a,
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
