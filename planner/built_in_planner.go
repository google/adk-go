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

// BuiltIn is implemented by *BuiltInPlanner and by the types that embed it.
// The agent's flow uses it to tell a built-in planner from any other Planner,
// as adk-python does with isinstance(planner, BuiltInPlanner).
//
// The unexported method of BuiltIn cannot be declared outside this package, so
// a type implements BuiltIn only through a *BuiltInPlanner it embeds, directly
// or through an embedded BuiltIn value. Go promotes the methods of
// *BuiltInPlanner to a struct that embeds *BuiltInPlanner and to a pointer to a
// struct that embeds BuiltInPlanner, and the methods of BuiltIn to a struct
// that embeds a BuiltIn value. Those types implement BuiltIn unless they
// declare a method of the same name with another signature. A type that has an
// ApplyThinkingConfig method but embeds none of these is an ordinary Planner.
//
// Before each model call, the flow calls ApplyThinkingConfig on a planner that
// implements BuiltIn, and neither calls its BuildPlanningInstruction nor clears
// the Thought flags on request history parts. The call goes to the planner's
// own ApplyThinkingConfig if its type declares one, and otherwise to the
// embedded BuiltInPlanner or BuiltIn value; a nil embedded BuiltIn value makes
// the call panic. An embedding type's own BuildPlanningInstruction is never
// called.
//
// After ApplyThinkingConfig returns, the flow replaces a non-nil request
// Config, thinking config included, with a deep copy, whichever
// ApplyThinkingConfig ran and whether or not it replaced the request's Config.
// So before-model callbacks, plugins and the model can edit the thinking
// config of the request they receive without changing the planner, a later
// request, or another agent that shares the planner. As with the copy the
// flow makes of llmagent.Config.GenerateContentConfig, values held in
// interface-typed fields of the Config, such as ResponseJsonSchema, stay
// shared.
type BuiltIn interface {
	// ApplyThinkingConfig sets the thinking config of req, the request about
	// to be sent to the model.
	ApplyThinkingConfig(req *model.LLMRequest)

	builtIn()
}

// BuiltInPlanner configures an LLM agent to use the model's built-in thinking
// feature via a genai.ThinkingConfig.
//
// Unlike PlanReActPlanner and custom Planner implementations, BuiltInPlanner
// does not append a planning instruction and does not clear Thought flags on
// request history parts. It implements BuiltIn and Planner but not
// RewritingPlanner, so the flow does not rewrite model response parts for it.
//
// A struct that embeds *BuiltInPlanner implements BuiltIn, so the flow calls
// the struct's own ApplyThinkingConfig if it defines one and never calls its
// BuildPlanningInstruction, even one the struct defines itself. If the struct
// defines its own ProcessPlanningResponse, it implements RewritingPlanner, and
// the flow calls that method.
//
// If both llmagent.Config.GenerateContentConfig.ThinkingConfig and
// BuiltInPlanner.ThinkingConfig are non-nil, BuiltInPlanner.ThinkingConfig
// takes precedence and overwrites the request's ThinkingConfig on every model
// call.
//
// The agent's flow gives each model request its own copy of ThinkingConfig
// (see BuiltIn), so a before-model callback or plugin that edits the request's
// thinking config changes neither ThinkingConfig nor any other request.
//
// The zero value (ThinkingConfig == nil) is valid and acts as a no-op.
// BuiltInPlanner is safe for concurrent use as long as ThinkingConfig and the
// value it points to are not mutated concurrently.
type BuiltInPlanner struct {
	// ThinkingConfig is applied to each outbound model.LLMRequest.Config.
	// If nil, ApplyThinkingConfig leaves the request unchanged.
	ThinkingConfig *genai.ThinkingConfig
}

// NewBuiltInPlanner returns a non-nil *BuiltInPlanner whose ThinkingConfig is
// thinkingConfig. A nil thinkingConfig is allowed and gives a planner that
// changes no request.
func NewBuiltInPlanner(thinkingConfig *genai.ThinkingConfig) *BuiltInPlanner {
	return &BuiltInPlanner{ThinkingConfig: thinkingConfig}
}

// ApplyThinkingConfig sets req.Config.ThinkingConfig to p.ThinkingConfig,
// allocating req.Config if it is nil and overwriting any thinking config the
// request already has. Afterwards req.Config is non-nil and
// req.Config.ThinkingConfig is p.ThinkingConfig itself, not a copy, so a change
// made through req changes p.ThinkingConfig. The agent's flow replaces it with
// a copy before callbacks, plugins or the model see the request (see BuiltIn);
// a caller that applies the config to a request of its own and then edits the
// request's thinking config must replace it with a copy first.
//
// p and req may be nil. When p, req or p.ThinkingConfig is nil,
// ApplyThinkingConfig leaves req unmodified.
func (p *BuiltInPlanner) ApplyThinkingConfig(req *model.LLMRequest) {
	if p == nil || req == nil || p.ThinkingConfig == nil {
		return
	}
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	req.Config.ThinkingConfig = p.ThinkingConfig
}

// BuildPlanningInstruction implements Planner and always returns ("", nil).
func (p *BuiltInPlanner) BuildPlanningInstruction(ctx agent.ReadonlyContext, req *model.LLMRequest) (string, error) {
	return "", nil
}

func (*BuiltInPlanner) builtIn() {}
