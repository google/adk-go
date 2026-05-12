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
	"errors"

	"google.golang.org/genai"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// ErrOutsideToolCall is returned by Context.RequestConfirmation (and
// other tool-call-only mutating operations) when called on a Context
// that wasn't produced by a tool dispatcher. Callbacks and agent-level
// code can detect this to know they're not inside a tool call.
var ErrOutsideToolCall = errors.New("agent: tool-only operation called outside a tool call")

// ReadonlyContext provides read-only access to invocation context data.
//
// Used in narrow callsites where mutation must be impossible at compile
// time (e.g. InstructionProvider, Predicate, Toolset.Tools). For the
// common case, use Context (which embeds ReadonlyContext and adds the
// full capability surface).
//
// Mirrors adk-python's `class ReadonlyContext` (kept as the parent
// class of `Context` for the same compile-time read-only purpose).
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

/*
Context is the unified user-facing context for ADK 2.0. Every callback,
tool, and agent code path receives a value implementing this interface.
The historical context types (InvocationContext, CallbackContext,
tool.Context) are now type aliases of Context.

Context embeds ReadonlyContext to expose its 8 read-only methods, then
adds mutating session/state/artifact access, memory, lifecycle control,
and tool-call-site capabilities. Tool-call-only methods (FunctionCallID,
Actions, ToolConfirmation, RequestConfirmation) follow a runtime-check
pattern: when called on a Context not produced by a tool dispatcher,
pollable accessors return zero values and mutating operations return
ErrOutsideToolCall.

Mirrors adk-python's `class Context(ReadonlyContext)` model.

An invocation:
 1. Starts with a user message and ends with a final response.
 2. Can contain one or multiple agent calls.
 3. Is handled by runner.Run().

An invocation runs an agent until it does not request to transfer to
another agent.

An agent call:
 1. Is handled by agent.Run().
 2. Ends when agent.Run() ends.

An agent call can contain one or multiple steps. For example, an LLM
agent runs steps in a loop until:
 1. A final response is generated.
 2. The agent transfers to another agent.
 3. EndInvocation() was called by the invocation context.

A step:
 1. Calls the LLM only once and yields its response.
 2. Calls the tools and yields their responses if requested.

The summarization of the function response is considered another step,
since it is another LLM call. A step ends when it's done calling the
LLM and tools, or if EndInvocation() was called by the invocation
context at any time.

	┌─────────────────────── invocation ──────────────────────────┐
	┌──────────── llm_agent_call_1 ────────────┐ ┌─ agent_call_2 ─┐
	┌──── step_1 ────────┐ ┌───── step_2 ──────┐
	[call_llm] [call_tool] [call_llm] [transfer]
*/
type Context interface {
	ReadonlyContext

	// Identity / framework objects.
	Agent() Agent

	// Session and state (mutable view).
	Session() session.Session
	State() session.State

	// Artifacts service (Save records into Actions().ArtifactDelta when
	// the Context was produced by a callback or tool dispatcher).
	Artifacts() Artifacts

	// Memory access.
	Memory() Memory
	SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error)

	// Lifecycle.
	RunConfig() *RunConfig
	// EndInvocation ends the current invocation. This stops any planned
	// agent calls.
	EndInvocation()
	// Ended reports whether the invocation has ended.
	Ended() bool

	// WithContext returns a new Context with the embedded
	// context.Context replaced.
	//
	// NOTE: This is a temporary solution and will be removed later. The
	// proper solution we plan is to stop embedding go context in adk
	// context types and split it.
	WithContext(ctx context.Context) Context

	// Tool-call site capabilities. When called on a Context not
	// produced by a tool dispatcher, FunctionCallID returns "", Actions
	// and ToolConfirmation return nil, and RequestConfirmation returns
	// ErrOutsideToolCall.
	FunctionCallID() string
	Actions() *session.EventActions
	ToolConfirmation() *toolconfirmation.ToolConfirmation
	RequestConfirmation(hint string, payload any) error
}

// InvocationContext is an alias for Context. Kept for backward
// compatibility; new code should use Context directly.
type InvocationContext = Context

// CallbackContext is an alias for Context. Kept for backward
// compatibility; new code should use Context directly.
type CallbackContext = Context
