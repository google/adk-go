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

package llminternal

import (
	"fmt"
	"iter"
	"strings"

	"google.golang.org/genai"

	icontext "google.golang.org/adk/internal/context"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// runTaskRequests handles any session.TaskRequest entries the coordinator
// produced via task.NewRequestTaskTool. For each (function_call_id,
// TaskRequest), the named task agent is located in the agent tree and
// invoked with a rendered user message; events from the task agent are
// forwarded to yield. After the task agent emits a session.TaskResult
// (via task.NewFinishTaskTool), a synthetic FunctionResponse event is
// produced and yielded so the coordinator's next turn observes the
// result keyed by the original call ID.
//
// The implementation is conservative: it consumes events serially per
// task to keep ordering deterministic. Parallel delegation across
// multiple call IDs is enabled by simply scheduling each task in the
// merged event one after another — the LLM still sees them as one
// batch because they share the model turn that originated them.
func (f *Flow) runTaskRequests(
	ctx agent.InvocationContext,
	ev *session.Event,
	yield func(*session.Event, error) bool,
) bool {
	if ev == nil || len(ev.Actions.RequestTask) == 0 {
		return true
	}
	parents := parentmap.FromContext(ctx)
	for callID, req := range ev.Actions.RequestTask {
		taskAgent := findAgentByName(ctx.Agent(), parents, req.AgentName)
		if taskAgent == nil {
			yield(nil, fmt.Errorf("task: agent %q not found in tree", req.AgentName))
			return false
		}
		// Build a child invocation context whose UserContent renders the
		// task input. The task agent runs as a sub-call: ownership stays
		// with the coordinator, so we don't go through the runner's
		// findAgentToRun path.
		childCtx := newTaskChildContext(ctx, taskAgent, req)
		var finish *session.TaskResult
		for childEv, err := range taskAgent.Run(childCtx) {
			if !yield(childEv, err) {
				return false
			}
			if err != nil {
				return false
			}
			if childEv != nil && len(childEv.Actions.FinishTask) > 0 {
				// Pick the latest FinishTask entry; tasks normally produce one.
				for _, r := range childEv.Actions.FinishTask {
					r := r
					finish = &r
				}
			}
		}
		// Synthesize a FunctionResponse for the coordinator. If the task
		// never called finish_task, synthesize an empty result so the
		// coordinator can decide how to proceed rather than stalling.
		respPayload := map[string]any{}
		if finish != nil && finish.Output != nil {
			respPayload = finish.Output
		}
		fr := session.NewEvent(ctx.InvocationID())
		fr.Author = ctx.Agent().Name()
		fr.Branch = ctx.Branch()
		fr.LLMResponse = model.LLMResponse{
			Content: &genai.Content{
				Role: genai.RoleUser,
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       callID,
						Name:     req.AgentName,
						Response: respPayload,
					},
				}},
			},
		}
		if !yield(fr, nil) {
			return false
		}
	}
	return true
}

// findAgentByName searches the agent tree (root via parents) for the
// named agent. Returns nil if not found.
func findAgentByName(start agent.Agent, parents parentmap.Map, name string) agent.Agent {
	if start == nil {
		return nil
	}
	root := start
	for {
		p := parents[root.Name()]
		if p == nil {
			break
		}
		root = p
	}
	return root.FindAgent(name)
}

// newTaskChildContext builds a sub-invocation context for a task agent.
// Mirrors the scaffolding in agent.Run: a fresh InvocationContext with
// the same session/services, but UserContent rendered from the task
// input so the task agent's contents-builder picks it up naturally.
func newTaskChildContext(parent agent.InvocationContext, taskAgent agent.Agent, req session.TaskRequest) agent.InvocationContext {
	rendered := renderTaskInput(req)
	uc := &genai.Content{
		Role:  genai.RoleUser,
		Parts: []*genai.Part{{Text: rendered}},
	}
	return icontext.NewInvocationContext(parent, icontext.InvocationContextParams{
		Artifacts:    parent.Artifacts(),
		Memory:       parent.Memory(),
		Session:      parent.Session(),
		Agent:        taskAgent,
		UserContent:  uc,
		RunConfig:    parent.RunConfig(),
		InvocationID: parent.InvocationID(),
	})
}

// renderTaskInput formats a TaskRequest's input as a human-readable
// string. Mirrors adk-python's render_task_input — labels each field,
// flagged with a single-turn nudge when applicable. Single-turn vs
// multi-turn distinction will be wired once the task agent's Mode is
// surfaced through agent.Agent (Phase 6D2); for Phase 6D the renderer
// emits the labelled body without the single-turn warning.
func renderTaskInput(req session.TaskRequest) string {
	var b strings.Builder
	b.WriteString("[Delegated Task]\n")
	for k, v := range req.Input {
		fmt.Fprintf(&b, "%s: %v\n", k, v)
	}
	return b.String()
}

// silence unused-variable in incremental development.
var _ iter.Seq2[*session.Event, error]
