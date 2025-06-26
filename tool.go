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

package adk

import (
	"context"
)

// ToolContext is the tool invocation context.
type ToolContext struct {
	// The invocation context of the tool call.
	InvocationContext *InvocationContext

	// The function call id of the current tool call.
	// This id was returned in the function call event from LLM to identify
	// a function call. If LLM didn't return an id, ADK will assign one t it.
	// This id is used to map function call response to the original function call.
	FunctionCallID string

	// The event actions of the current tool call.
	EventActions *EventActions
}

// Tool is the ADK tool interface.
type Tool interface {
	Name() string
	Description() string
	// ProcessRequest processes the outgoing LLM request for this tool.
	// Use cases:
	//  * Adding this tool schema to the LLM request.
	//  * Preprocess the LLM request before it's sent out.
	ProcessRequest(ctx context.Context, tc *ToolContext, req *LLMRequest) error

	// Run runs the tool with the given argument and returns the result.
	Run(ctx context.Context, tc *ToolContext, args map[string]any) (map[string]any, error)

	// note for reviewers: this interface corresponds to adk-python's BaseTool (tools/base_tool.py)
}
