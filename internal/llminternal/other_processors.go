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

package llminternal

import (
	"fmt"
	"iter"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/planner"
	"google.golang.org/adk/v2/session"
)

// nlPlanningRequestProcessor prepares req for the agent's planner: a built-in
// planner (planner.BuiltIn) sets the thinking config, of which req then gets
// its own copy; any other planner appends its planning instruction and clears
// the thought flags of the request history (adk-python
// flows/llm_flows/extensions/_planning.py).
func nlPlanningRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		p := getPlanner(ctx)
		if p == nil {
			return
		}
		if bp, ok := p.(planner.BuiltIn); ok {
			bp.ApplyThinkingConfig(req)
			// ApplyThinkingConfig may store a value that the planner shares with
			// every request and every agent using it. Before-model callbacks,
			// plugins and the model may edit req, so req gets a copy, as
			// basicRequestProcessor gives it a copy of GenerateContentConfig.
			if req.Config != nil {
				req.Config.ThinkingConfig = clone(req.Config.ThinkingConfig)
			}
			return
		}
		inst, err := p.BuildPlanningInstruction(icontext.NewReadonlyContext(ctx), req)
		if err != nil {
			yield(nil, fmt.Errorf("build planning instruction: %w", err))
			return
		}
		if inst != "" {
			utils.AppendInstructions(req, inst)
		}
		removeThoughtFromRequest(req)
	}
}

// removeThoughtFromRequest clears Thought on every part of req.Contents, so
// parts a planner marked as thoughts reach the model as ordinary history.
// req.Contents holds copies of the session events, so stored history keeps
// its flags.
func removeThoughtFromRequest(req *model.LLMRequest) {
	for _, content := range req.Contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if part != nil {
				part.Thought = false
			}
		}
	}
}

func codeExecutionRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk/flows/llm_flows/_code_execution.py)
	return func(yield func(*session.Event, error) bool) {}
}

// nlPlanningResponseProcessor rewrites resp's parts through the agent's
// planner when it implements planner.RewritingPlanner. If the planner writes
// state, it yields one content-less event that carries the state and artifact
// deltas, which the flow emits before the model response event (adk-python
// flows/llm_flows/extensions/_planning.py).
func nlPlanningResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
			return
		}
		p := responseRewritingPlanner(ctx)
		if p == nil {
			return
		}
		stateDelta := make(map[string]any)
		artifactDelta := make(map[string]int64)
		cctx := icontext.NewCallbackContextWithDelta(ctx, stateDelta, artifactDelta)
		processedParts, err := p.ProcessPlanningResponse(cctx, resp.Content.Parts)
		if err != nil {
			yield(nil, fmt.Errorf("process planning response: %w", err))
			return
		}
		// An empty result keeps the model's parts, as adk-python's
		// "if processed_parts:" does.
		if len(processedParts) > 0 {
			resp.Content.Parts = processedParts
		}
		if len(stateDelta) == 0 {
			return
		}
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Author = ctx.Agent().Name()
		ev.Branch = ctx.Branch()
		ev.Actions = session.EventActions{StateDelta: stateDelta, ArtifactDelta: artifactDelta}
		yield(ev, nil)
	}
}

func codeExecutionResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
	// TODO: implement (adk-python src/google/adk_code_execution.py)
	return func(yield func(*session.Event, error) bool) {}
}

// getPlanner returns the planner of the agent in ctx, or nil when the agent is
// not an LLM agent or has no planner.
func getPlanner(ctx agent.InvocationContext) planner.Planner {
	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return nil
	}
	return llmAgent.internal().Planner
}

// responseRewritingPlanner returns the agent's planner if it rewrites model
// responses, or nil.
func responseRewritingPlanner(ctx agent.InvocationContext) planner.RewritingPlanner {
	rp, _ := getPlanner(ctx).(planner.RewritingPlanner)
	return rp
}
