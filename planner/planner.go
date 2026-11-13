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

package planner

import (
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
)

// Planner guides an LLM agent to plan and reason across model calls.
//
// Before each model call, the agent's flow invokes BuildPlanningInstruction
// and appends any non-empty instruction to the request's system instruction,
// unless the planner implements BuiltIn (a *BuiltInPlanner or a struct that
// embeds one; see BuiltIn). The flow rewrites model responses through the
// planner only if it also implements RewritingPlanner.
//
// In adk-python both planner methods are abstract, so every Python planner
// implements the response step. A planner ported from Python therefore
// implements RewritingPlanner in Go and behaves as in Python. A planner that
// implements only Planner (an instruction-only planner) can be written only in
// Go: the flow does not treat it as a response-rewriting planner, so it calls
// no response step for it and keeps re-calling the model on a thought-only
// turn.
//
// Implementations must be safe for concurrent use across multiple invocations
// if the configured agent is run concurrently.
type Planner interface {
	// BuildPlanningInstruction returns the system instruction text to append to
	// req for planning, or "" if no planning instruction should be added.
	//
	// During an agent run ctx and req are non-nil. req is the request about to
	// be sent to the model; implementations must not modify it. The flow
	// appends a non-empty result to the existing system instruction, separated
	// from it by "\n\n", and appends nothing for ("", nil).
	//
	// A non-nil error means the instruction could not be built: the flow
	// aborts the current step, does not call the model, and yields an error
	// that wraps it.
	BuildPlanningInstruction(ctx agent.ReadonlyContext, req *model.LLMRequest) (string, error)
}

// RewritingPlanner is a Planner that also rewrites the model's response parts.
//
// After each model response with content parts, the flow invokes
// ProcessPlanningResponse if the agent's planner implements RewritingPlanner.
// PlanReActPlanner implements it; BuiltInPlanner does not. A struct that embeds
// *BuiltInPlanner and defines its own ProcessPlanningResponse implements
// RewritingPlanner, and the flow calls it.
//
// In adk-python both planner methods are abstract, so every Python planner
// implements the response step. A planner ported from Python therefore
// implements RewritingPlanner in Go and behaves as in Python. An
// instruction-only planner, which implements only Planner, can be written only
// in Go.
//
// Implementations must be safe for concurrent use across multiple invocations
// if the configured agent is run concurrently.
type RewritingPlanner interface {
	Planner

	// ProcessPlanningResponse rewrites the parts of one model response for
	// planning.
	//
	// During an agent run ctx is a non-nil callback context and responseParts
	// is non-empty. Direct callers, such as unit tests or callers of
	// PlanReActPlanner, may pass a nil ctx to an implementation that does not
	// access session state or artifacts.
	//
	// If the returned slice is non-empty, the flow replaces the model
	// response's parts with it. If it is nil or empty, the flow keeps the
	// response's existing parts slice, including any changes the
	// implementation made in place to elements of responseParts. If the
	// implementation writes to ctx.State(), the flow emits a separate event
	// carrying that state change before the model response event.
	//
	// A non-nil error means response processing failed: the flow aborts the
	// current step and yields an error that wraps it, without emitting the
	// model response event.
	ProcessPlanningResponse(ctx agent.Context, responseParts []*genai.Part) ([]*genai.Part, error)
}
