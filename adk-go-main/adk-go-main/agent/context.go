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

	"google.golang.org/adk/session"
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

	// UserContent that started this invocation.
	UserContent() *genai.Content

	// RunConfig stores the runtime configuration used during this invocation.
	RunConfig() *RunConfig

	// EndInvocation ends the current invocation. This stops any planned agent
	// calls.
	EndInvocation()
	// Ended returns whether the invocation has ended.
	Ended() bool
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

// CallbackContext is passed to user callbacks during agent execution.
type CallbackContext interface {
	ReadonlyContext

	Artifacts() Artifacts
	State() session.State
}
