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

package agent

import (
	"context"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

/*
InvocationContext represents the context of an agent invocation.

An invocation:
 1. Starts with a user message and ends with a final response.
 2. Can contain one or multiple agent calls.
 3. Is handled by runner.Run().

An invocation runs an agent until it does not request to transfer to another
agent.

An agent call:
 1. Is handled by agent.Run().
 2. Ends when agent.Run() ends.

An agent call can contain one or multiple steps.
For example, LLM agent runs steps in a loop until:
 1. A final response is generated.
 2. The agent transfers to another agent.
 3. EndInvocation() was called by the invocation context.

A step:
 1. Calls the LLM only once and yields its response.
 2. Calls the tools and yields their responses if requested.

The summarization of the function response is considered another step, since
it is another LLM call.
A step ends when it's done calling LLM and tools, or if the EndInvocation() was
called by invocation context at any time.

	┌─────────────────────── invocation ──────────────────────────┐
	┌──────────── llm_agent_call_1 ────────────┐ ┌─ agent_call_2 ─┐
	┌──── step_1 ────────┐ ┌───── step_2 ──────┐
	[call_llm] [call_tool] [call_llm] [transfer]
*/
type InvocationContext interface {
	context.Context

	// Agent of this invocation context.
	Agent() Agent

	// Artifacts of the current session.
	Artifacts() Artifacts

	// Memory is scoped to sessions of the current user_id.
	Memory() Memory

	// Session of the current invocation context.
	Session() session.Session

	InvocationID() string

	// Branch of the invocation context.
	// The format is like agent_1.agent_2.agent_3, where agent_1 is the parent
	// of agent_2, and agent_2 is the parent of agent_3.
	//
	// Branch is used when multiple sub-agents shouldn't see their peer agents'
	// conversation history.
	//
	// Applicable to parallel agent because its sub-agents run concurrently.
	Branch() string

	// IsolationScope of the invocation context. When set, the agent's LLM
	// prompt history includes only session events whose IsolationScope
	// matches exactly. Empty means unscoped.
	IsolationScope() string

	// UserContent that started this invocation.
	UserContent() *genai.Content

	// RunConfig stores the runtime configuration used during this invocation.
	RunConfig() *RunConfig

	// EndInvocation ends the current invocation. This stops any planned agent
	// calls.
	EndInvocation()
	// Ended returns whether the invocation has ended.
	Ended() bool

	// ResumedInput returns the user-supplied response payload
	// associated with the given InterruptID for the current
	// activation, or (nil, false) if none. Implementations that
	// do not carry resume payloads always return (nil, false).
	ResumedInput(interruptID string) (any, bool)

	// WithContext returns a new instance of the context with overridden embedded context.
	// NOTE: This is a temporary solution and will be removed later. The proper solution
	// we plan is to stop embedding go context in adk context types and split it.
	WithContext(ctx context.Context) InvocationContext

	Apply(d *InvocationContextDelta) InvocationContext
}

// ReadonlyContext provides read-only access to invocation context data.
type ReadonlyContext interface {
	context.Context

	// UserContent that started this invocation.
	UserContent() *genai.Content
	InvocationID() string
	AgentName() string
	ReadonlyState() session.ReadonlyState

	UserID() string
	AppName() string
	SessionID() string
	// Branch of the current invocation.
	Branch() string
}

// Context is the unified context passed to user callbacks during agent
// execution and to tools when they are called. It provides access to the
// originating function call, mutable event actions, long-term memory search,
// and the Human-in-the-Loop (HITL) confirmation flow.
type Context interface {
	ReadonlyContext
	InvocationContext

	// Callback context
	Artifacts() Artifacts
	State() session.State

	// Tool context section

	// FunctionCallID returns the unique identifier of the function call
	// that triggered this tool execution.
	FunctionCallID() string

	// Actions returns the EventActions for the current event. This can be
	// used by the tool to modify the agent's state, transfer to another
	// agent, or perform other actions.
	Actions() *session.EventActions

	// SearchMemory performs a semantic search on the agent's memory.
	SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error)

	// ToolConfirmation returns a handler for checking the Human-in-the-Loop
	// confirmation status for the current tool context. This should be used
	// within a tool's logic *before* performing any sensitive operations that
	// require user approval.
	//
	// Example Usage:
	//   if confirmation := ctx.ToolConfirmation(); confirmation == nil {
	//       // Confirmation required, create confirmation or handle appropriately
	//       ctx.RequestConfirmation("hint", payload)
	//   }
	//
	// The returned *toolconfirmation.ToolConfirmation object provides methods
	// to check the actual confirmation state.
	ToolConfirmation() *toolconfirmation.ToolConfirmation

	// RequestConfirmation initiates the Human-in-the-Loop (HITL) process to
	// ask the user for approval before the tool proceeds with a specific
	// action. Call this method when a tool needs explicit user consent.
	//
	// This will typically result in the ADK emitting a special event
	// (e.g., a FunctionCall like "adk_request_confirmation") to the client
	// application/UI, prompting the user for a decision.
	//
	// Args:
	//   - hint: A human-readable string explaining why confirmation is needed.
	//     This is usually displayed to the user in the confirmation prompt.
	//   - payload: Any additional data or context about the action requiring
	//     confirmation.
	//
	// Returns:
	//   - nil: If the confirmation request was successfully enqueued or
	//     initiated within the ADK. This indicates that the process of asking
	//     the user has begun. It does NOT mean the action is approved. The
	//     tool's execution will likely pause or be suspended until the user
	//     responds.
	//   - error: If there was a failure in initiating the confirmation process
	//     itself (e.g., invalid arguments, issue with the event system). The
	//     request to ask the user has not been sent.
	RequestConfirmation(hint string, payload any) error

	// Workflow node section

	// ResumedInput returns the response payload for a re-entry resume
	// activation keyed by InterruptID, or (nil, false) otherwise.
	ResumedInput(interruptID string) (any, bool)

	// Path returns the composite path of the currently-executing node.
	// Empty for top-level static nodes; "<parent_path>/<child_name>@<run_id>"
	// for dynamic children.
	Path() string

	// RunID returns the per-invocation identifier. Empty for top-level
	// static nodes; auto-counter or user-supplied via WithRunID for
	// dynamic children.
	RunID() string

	// WithBranch returns a Context whose Branch() returns the
	// given value; all other fields (path, runID, subScheduler,
	// resumeInputs, embedded InvocationContext) are preserved.
	WithBranch(branch string) Context

	// SubScheduler is non-nil only when this context belongs to a
	// dynamic-node activation; RunNode uses it to schedule children.
	SubScheduler() DynamicSubScheduler

	// WithAgentContext creates a new context as a shallow copy setting the internal contexts to ctx.
	WithAgentContext(ctx context.Context) Context

	// WithAgentTimeout creates a new context as a shallow copy, adding timeout to the top of the underlying context.Context.
	WithAgentTimeout(timeout time.Duration) (Context, context.CancelFunc)

	// WithAgentCancel creates a new context as a shallow copy, adding cancellation to the top of the underlying context.Context.
	WithAgentCancel() (Context, context.CancelFunc)

	// OutputForAncestors are the delegating-ancestor paths carried
	// into this activation when it runs as a WithUseAsOutput child;
	// its dynamic sub-scheduler reads them to stamp OutputFor.
	OutputForAncestors() []string

	Apply(d *CommonContextDelta) Context
}
