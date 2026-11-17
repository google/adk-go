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
	"fmt"
	"maps"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/internal/eventmetadata"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
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

	// Artifacts of the current session. It is nil when the invocation has no
	// artifact service, except on a Context, whose Artifacts is never nil.
	Artifacts() Artifacts

	// Memory is scoped to sessions of the current user_id. It is nil when the
	// invocation has no memory service, except on a Context, whose Memory is
	// never nil.
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

	WithICDelta(d *InvocationContextDelta) InvocationContext
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
//
// On the contexts ADK builds, Memory and Artifacts are never nil. When the
// invocation has no memory or artifact service, their methods fail with an
// error that says so, such as "artifact service is not initialized", and
// SearchMemory fails with "memory service is not available". The memory
// calls that are not methods of Memory are package functions over a Context:
// AddSessionToMemory, AddEventsToMemory and AddMemory. A callback's Context
// refuses SearchMemory while a memory service is configured; a tool's Context
// searches. A tool's Context reports no session (Session returns nil), while
// AddSessionToMemory still reaches the invocation's session. A callback's
// Context returns the invocation's session, and both return the run config of
// the invocation.
type Context interface {
	ReadonlyContext
	InvocationContext

	// Callback context

	// State returns this context's view of the session state: a read sees the
	// writes made through this context first, then the session's state; a write
	// goes to this context's state delta and to the session's state.
	//
	// When a model turn holds several function calls, ADK runs the calls
	// concurrently, and State does not order them: Get, Set and All never wait
	// for another call, and a call reads what the calls that have already
	// written put there. A tool whose calls must see the writes of the earlier
	// calls of their turn, such as one that reads a key and writes a value
	// computed from it while its sibling calls write the same key, declares
	// itself sequential (tool.SequentialRunner, functiontool.Config.Sequential),
	// whether or not it also takes a lock of its own. A call of a sequential
	// tool starts only after every earlier call of the turn has returned,
	// whether those calls are sequential or not, so it sees all of their
	// writes. A lock of the tool's own, which it may hold across State, does
	// not order the calls.
	//
	// When several calls write one key, the event that merges the turn's
	// responses takes the value of the later call in the model's response. A
	// lock alone therefore keeps each read-modify-write atomic but can leave a
	// scalar one update short: two calls that each read n and write n+1 under
	// the lock leave n+1 when the later call takes the lock first. A
	// sequential tool's calls leave n+2. Use of State after the call has
	// returned (a streaming tool's goroutine in live mode) is not ordered.
	//
	// The returned State, like the call's context, belongs to one call at a
	// time: Set writes the call's state delta, a plain map, so a call must not
	// use it from several goroutines at once.
	//
	// On the context of a workflow node, Set checks each write against the
	// workflow's state schema (workflow.WithStateSchema) before it writes
	// anything, and returns the schema's error when the write is refused. The
	// refused write is not made, so the node must not ignore the error. The
	// same rule applies there: one node context's Set is not safe for use from
	// several goroutines at once.
	State() session.State

	// Tool context section

	// FunctionCallID returns the unique identifier of the function call
	// that triggered this tool execution.
	FunctionCallID() string

	// Actions returns the EventActions for the current event. This can be
	// used by the tool to modify the agent's state, transfer to another
	// agent, or perform other actions.
	//
	// On the context of a workflow node, Actions returns the pending actions
	// of the node's activation. ADK puts only their StateDelta and
	// ArtifactDelta on the node's next event; SkipSummarization, Escalate,
	// TransferToAgent, RequestedAuthConfigs, RenderUIWidgets and Compaction
	// are not used in a node.
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

	// Path returns the composite path of the currently-executing node.
	// Empty for top-level static nodes; "<parent_path>/<child_name>@<run_id>"
	// for dynamic children.
	Path() string

	// RunID returns the per-invocation identifier. Empty for top-level
	// static nodes; auto-counter or user-supplied via WithRunID for
	// dynamic children.
	RunID() string

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

	// WithDelta returns a copy of source context with applied delta d
	WithDelta(d *CommonContextDelta) Context
}

// AddSessionToMemory adds the session of ctx's invocation to ctx's memory
// service, so later invocations of the same user can recall it. It works from
// callback and tool contexts alike: from a tool context, whose Session method
// reports no session, it still passes the invocation's session, and so it does
// from a type of the caller's own that embeds a tool context. A Context whose
// own Session method reports a session passes that session instead.
//
// It returns an error, without calling any service, when the invocation has no
// memory service. Otherwise it returns the error of the service's
// AddSessionToMemory unchanged.
func AddSessionToMemory(ctx Context) error {
	m, ok := memoryService(ctx)
	if !ok {
		return errNoMemoryForSession
	}
	return m.AddSessionToMemory(ctx, invocationSession(ctx))
}

// invocationSession returns the session of ctx's invocation. A tool context
// reports no session through Session, and other contexts wrap one (a live
// streaming tool runs under an ADK wrapper, a user's decorator under its own),
// so the session comes from Value, which every ADK context forwards down to the
// commonContext that holds the invocation. ADK's own context types are asked
// through Value first, which spares a tool context's Session its "not
// supported" log line. Any other Context's Session wins when it is not nil, so
// a type that overrides Session passes its own session; when it is nil, that
// Context is asked through Value too.
func invocationSession(ctx Context) session.Session {
	if _, ours := ctx.(adkcontext.Source); ours {
		if s, ok := ctx.Value(adkcontext.InvocationSessionKey).(session.Session); ok {
			return s
		}
	}
	if s := ctx.Session(); s != nil {
		return s
	}
	s, _ := ctx.Value(adkcontext.InvocationSessionKey).(session.Session)
	return s
}

// eventAdder is the method of a Memory that can add events, such as the
// memory the runner builds over a [memory.EventAdder].
type eventAdder interface {
	AddEventsToMemory(ctx context.Context, req *memory.AddEventsRequest) error
}

// memoryAdder is the method of a Memory that accepts direct writes, such as
// the memory the runner builds over a [memory.MemoryAdder].
type memoryAdder interface {
	AddMemory(ctx context.Context, req *memory.AddMemoryRequest) error
}

// AddEventsToMemory adds events to ctx's memory service, scoped to the app,
// user and session of ctx (ctx.AppName, ctx.UserID and ctx.SessionID).
// customMetadata is passed to the service unchanged; which keys a service
// reads is up to the service. events and customMetadata are passed on as they
// are, without copying; nil or empty events are the service's to accept or
// refuse.
//
// It returns an error, without calling any service, when the invocation has no
// memory service. It returns [memory.ErrAddEventsUnsupported] when the memory
// service cannot add events (see [memory.EventAdder]).
// Otherwise it returns the service's error unchanged.
func AddEventsToMemory(ctx Context, events []*session.Event, customMetadata map[string]any) error {
	m, ok := memoryService(ctx)
	if !ok {
		return errNoMemoryForEvents
	}
	adder, ok := m.(eventAdder)
	if !ok {
		return memory.ErrAddEventsUnsupported
	}
	return adder.AddEventsToMemory(ctx, &memory.AddEventsRequest{
		AppName:        ctx.AppName(),
		UserID:         ctx.UserID(),
		SessionID:      ctx.SessionID(),
		Events:         events,
		CustomMetadata: customMetadata,
	})
}

// AddMemory writes memories directly to ctx's memory service, scoped to the app
// and user of ctx (ctx.AppName and ctx.UserID). customMetadata is passed to the
// service unchanged. memories and customMetadata are passed on as they are,
// without copying; nil or empty memories are the service's to accept or
// refuse.
//
// It returns an error, without calling any service, when the invocation has no
// memory service. It returns [memory.ErrAddMemoryUnsupported] when the memory
// service does not accept direct writes (see [memory.MemoryAdder]); the
// in-memory service is one of those. Otherwise it
// returns the service's error unchanged.
func AddMemory(ctx Context, memories []*memory.Entry, customMetadata map[string]any) error {
	m, ok := memoryService(ctx)
	if !ok {
		return errNoMemoryForEntries
	}
	adder, ok := m.(memoryAdder)
	if !ok {
		return memory.ErrAddMemoryUnsupported
	}
	return adder.AddMemory(ctx, &memory.AddMemoryRequest{
		AppName:        ctx.AppName(),
		UserID:         ctx.UserID(),
		Memories:       memories,
		CustomMetadata: customMetadata,
	})
}

// artifactVersionReader is the method of an Artifacts that can look up the
// record of an artifact version, such as the Artifacts the runner builds over
// its [artifact.Service].
type artifactVersionReader interface {
	ArtifactVersion(ctx context.Context, name string, version *int64) (*artifact.ArtifactVersion, error)
}

// ArtifactVersion looks up a version of an artifact by name in the invocation's
// artifact service. If version is nil, it returns the latest version.
// It returns an error wrapping [fs.ErrNotExist] if the artifact or the version
// does not exist. A non-nil version of 0 or less returns an error wrapping
// [fs.ErrNotExist] without calling the service.
// If the context's artifacts do not support version lookup, it returns an error
// wrapping [errors.ErrUnsupported].
// If the invocation has no artifact service configured, it returns an error
// stating "artifact service is not initialized".
//
// The artifact is looked up with the app, user and session of the invocation,
// and a name with the "user:" prefix names an artifact of the user. When a
// tool from package tool/agenttool runs an agent, the lookups of that agent's
// tools go to the artifact service of the parent invocation, with the
// parent's app, user and session.
func ArtifactVersion(ctx Context, name string, version *int64) (*artifact.ArtifactVersion, error) {
	arts := ctx.Artifacts()
	for {
		switch a := arts.(type) {
		case nil, noServiceArtifacts:
			return nil, errNoArtifactService
		case artifactVersionReader:
			return a.ArtifactVersion(ctx, name, version)
		case interface{ Unwrap() Artifacts }:
			arts = a.Unwrap()
		default:
			return nil, fmt.Errorf("artifacts of type %T cannot look up versions: %w", arts, errors.ErrUnsupported)
		}
	}
}

// CustomMetadata returns the custom metadata of the run ctx belongs to:
// [RunConfig.CustomMetadata] without the keys ADK reserves for its own event
// metadata (the "__adk_internal_" and "adk.workflow." prefixes and the
// "adk:before_run_reply" key).
//
// Inside a run started by the runner it returns the copy the runner took when
// the run started. Without a runner it filters the run config of ctx, when ctx
// has a RunConfig method, as callback, tool and instruction-provider contexts
// do. It returns an empty, non-nil map when there is no metadata.
//
// The result is a new map on every call: changing it changes neither the run
// nor what later calls return. Values are not deep-copied. CustomMetadata is
// safe for concurrent use.
func CustomMetadata(ctx ReadonlyContext) map[string]any {
	var md map[string]any
	if snap, ok := ctx.Value(adkcontext.RunConfigKey).(customMetadataSnapshotter); ok {
		md = maps.Clone(snap.CustomMetadataSnapshot())
	} else if rc, ok := ctx.(interface{ RunConfig() *RunConfig }); ok && rc.RunConfig() != nil && rc.RunConfig().CustomMetadata != nil {
		md = eventmetadata.WithoutInternal(*rc.RunConfig().CustomMetadata)
	}
	if md == nil {
		return map[string]any{}
	}
	return md
}

// customMetadataSnapshotter is the method through which the runner's run
// config, stored under [adkcontext.RunConfigKey], hands out the custom
// metadata it filtered when the run started.
type customMetadataSnapshotter interface {
	CustomMetadataSnapshot() map[string]any
}
