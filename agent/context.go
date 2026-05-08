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

	"google.golang.org/genai"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// Context is the unified context type that ADK passes to user code:
// agents, tools, callbacks, and workflow nodes all receive a Context.
// It is the single name a developer needs to learn for the ADK API.
//
// Context exposes every capability ADK ever offers a user:
//   - invocation lifecycle (Agent, Memory, Session, EndInvocation, …),
//   - read and write access to session state and artifacts,
//   - tool-call metadata (FunctionCallID, Actions, ToolConfirmation, …),
//   - Human-in-the-Loop hooks (RequestConfirmation).
//
// Capabilities that only make sense inside a particular call site (for
// example, a tool callback's FunctionCallID) follow a "mix" policy:
//   - pollable accessors return zero values when called outside their
//     mode (e.g. FunctionCallID() returns "" inside an agent callback);
//   - mutating actions return a non-nil error in the same situation
//     (e.g. RequestConfirmation outside a tool returns an error).
//
// The historical sub-interfaces ReadonlyContext, CallbackContext, and
// InvocationContext are declared as type aliases of Context below.
// Existing code using those names continues to compile and gets the
// same widened capability set; new code should prefer Context.
type Context interface {
	context.Context

	// Identity and provenance.
	Agent() Agent
	AgentName() string
	InvocationID() string
	Branch() string
	UserContent() *genai.Content
	UserID() string
	AppName() string
	SessionID() string

	// Session, state, artifacts, memory.
	Session() session.Session
	State() session.State
	ReadonlyState() session.ReadonlyState
	Artifacts() Artifacts
	Memory() Memory
	SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error)

	// Lifecycle.
	RunConfig() *RunConfig
	EndInvocation()
	Ended() bool

	// WithContext returns a new instance of the context with overridden
	// embedded context.Context. NOTE: this is a temporary shim and will
	// be removed once ADK stops embedding context.Context in its own
	// context types.
	WithContext(ctx context.Context) Context
	// WithAgent returns a copy of the context with Agent overridden.
	// Used by Agent.Run wrappers (telemetry, sub-agent dispatch).
	WithAgent(a Agent) Context

	// Tool-call site capabilities. Outside of a tool call:
	//   - FunctionCallID returns "".
	//   - Actions returns nil.
	//   - ToolConfirmation returns nil.
	//   - RequestConfirmation returns a non-nil error.
	FunctionCallID() string
	Actions() *session.EventActions
	ToolConfirmation() *toolconfirmation.ToolConfirmation
	RequestConfirmation(hint string, payload any) error
}

/*
InvocationContext is a backward-compatibility alias for the unified
Context type. Use Context in new code.

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
type InvocationContext = Context

// ReadonlyContext is a backward-compatibility alias for the unified
// Context type. Use Context in new code.
//
// Historically ReadonlyContext exposed only a read-only subset of the
// invocation surface. Under the unified Context API the full surface
// is available; ADK 2.0 trades compile-time capability narrowing for
// a single type to learn.
type ReadonlyContext = Context

// CallbackContext is a backward-compatibility alias for the unified
// Context type. Use Context in new code.
type CallbackContext = Context
