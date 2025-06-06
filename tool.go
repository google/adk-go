package adk

import "context"

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
	EventActions []*EventAction
}

// Tool is the ADK tool interface.
type Tool interface {
	Name() string
	Description() string

	// ProcessRequest processes the outgoing LLM request for this tool.
	// Use cases:
	//  * Adding this tool to the LLM request.
	//  * Preprocess the LLM request before it's sent out.
	ProcessRequest(ctx context.Context, tc *ToolContext, req *LLMRequest) error

	// TODO: IsLongRunning, or LongRunningTool interface?
	// TODO: interface vs concrete (golang.org/x/tools/internal/mcp.Tool)
}

// TODO: func Declaration(Tool) JSONSchema ?