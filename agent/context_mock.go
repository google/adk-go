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

package agent

import (
	"context"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// StrictContextMock is a strict test double for the context interfaces
// ([ToolContext], [CallbackContext], [ReadonlyContext]).
//
// Embed it in a test fake and override only the methods your test actually
// uses. Because it implements the whole surface, embedders keep compiling as
// the interfaces grow.
//
// An un-overridden method panics with "not implemented" — an unexpected call
// fails the test loudly instead of silently returning a zero value.
//
// The exception is the standard library's context.Context methods (Deadline,
// Done, Err and Value): those read from the supplied Ctx rather than panicking,
// so the mock carries a usable context payload. If Ctx is nil they panic like
// everything else.
type StrictContextMock struct {
	// Ctx supplies the values returned by Deadline, Done, Err and Value.
	Ctx context.Context
}

// NewStrictContextMock returns a StrictContextMock backed by ctx. It keeps test
// literals concise when the mock is embedded as the only field of a fake.
func NewStrictContextMock(ctx context.Context) StrictContextMock {
	return StrictContextMock{Ctx: ctx}
}

func (m *StrictContextMock) ctx() context.Context {
	if m.Ctx == nil {
		panic("agent.StrictContextMock: Ctx is nil")
	}
	return m.Ctx
}

// context.Context methods, served from Ctx instead of panicking.

// Deadline implements [ToolContext].
func (m *StrictContextMock) Deadline() (deadline time.Time, ok bool) { return m.ctx().Deadline() }

// Done implements [ToolContext].
func (m *StrictContextMock) Done() <-chan struct{} { return m.ctx().Done() }

// Err implements [ToolContext].
func (m *StrictContextMock) Err() error { return m.ctx().Err() }

// Value implements [ToolContext].
func (m *StrictContextMock) Value(key any) any { return m.ctx().Value(key) }

// InvocationContext methods.

// Agent implements [ToolContext].
func (m *StrictContextMock) Agent() Agent { panic("not implemented") }

// Memory implements [ToolContext].
func (m *StrictContextMock) Memory() Memory { panic("not implemented") }

// Session implements [ToolContext].
func (m *StrictContextMock) Session() session.Session { panic("not implemented") }

// IsolationScope implements [ToolContext].
func (m *StrictContextMock) IsolationScope() string { panic("not implemented") }

// RunConfig implements [ToolContext].
func (m *StrictContextMock) RunConfig() *RunConfig { panic("not implemented") }

// EndInvocation implements [ToolContext].
func (m *StrictContextMock) EndInvocation() { panic("not implemented") }

// Ended implements [ToolContext].
func (m *StrictContextMock) Ended() bool { panic("not implemented") }

// ResumedInput implements [ToolContext].
func (m *StrictContextMock) ResumedInput(interruptID string) (any, bool) { panic("not implemented") }

// WithContext implements [ToolContext].
func (m *StrictContextMock) WithContext(ctx context.Context) InvocationContext {
	panic("not implemented")
}

// ReadonlyContext methods.

// UserContent implements [ToolContext].
func (m *StrictContextMock) UserContent() *genai.Content { panic("not implemented") }

// InvocationID implements [ToolContext].
func (m *StrictContextMock) InvocationID() string { panic("not implemented") }

// AgentName implements [ToolContext].
func (m *StrictContextMock) AgentName() string { panic("not implemented") }

// ReadonlyState implements [ToolContext].
func (m *StrictContextMock) ReadonlyState() session.ReadonlyState { panic("not implemented") }

// UserID implements [ToolContext].
func (m *StrictContextMock) UserID() string { panic("not implemented") }

// AppName implements [ToolContext].
func (m *StrictContextMock) AppName() string { panic("not implemented") }

// SessionID implements [ToolContext].
func (m *StrictContextMock) SessionID() string { panic("not implemented") }

// Branch implements [ToolContext].
func (m *StrictContextMock) Branch() string { panic("not implemented") }

// CallbackContext methods.

// Artifacts implements [ToolContext].
func (m *StrictContextMock) Artifacts() Artifacts { panic("not implemented") }

// State implements [ToolContext].
func (m *StrictContextMock) State() session.State { panic("not implemented") }

// ToolContext methods.

// FunctionCallID implements [ToolContext].
func (m *StrictContextMock) FunctionCallID() string { panic("not implemented") }

// Actions implements [ToolContext].
func (m *StrictContextMock) Actions() *session.EventActions { panic("not implemented") }

// SearchMemory implements [ToolContext].
func (m *StrictContextMock) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	panic("not implemented")
}

// ToolConfirmation implements [ToolContext].
func (m *StrictContextMock) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	panic("not implemented")
}

// RequestConfirmation implements [ToolContext].
func (m *StrictContextMock) RequestConfirmation(hint string, payload any) error {
	panic("not implemented")
}

// NodeContext methods.

// Path implements [ToolContext].
func (m *StrictContextMock) Path() string { panic("not implemented") }

// RunID implements [ToolContext].
func (m *StrictContextMock) RunID() string { panic("not implemented") }

// WithBranch implements [ToolContext].
func (m *StrictContextMock) WithBranch(branch string) Context { panic("not implemented") }

// SubScheduler implements [ToolContext].
func (m *StrictContextMock) SubScheduler() DynamicSubScheduler { panic("not implemented") }

// InvocationContext implements [ToolContext].
func (m *StrictContextMock) InvocationContext() InvocationContext { panic("not implemented") }

// SetInvocationContext implements [ToolContext].
func (m *StrictContextMock) SetInvocationContext(InvocationContext) { panic("not implemented") }

// WithAgentContext implements [ToolContext].
func (m *StrictContextMock) WithAgentContext(ctx context.Context) Context { panic("not implemented") }

// WithAgentTimeout implements [ToolContext].
func (m *StrictContextMock) WithAgentTimeout(timeout time.Duration) (Context, context.CancelFunc) {
	panic("not implemented")
}

// WithAgentCancel implements [ToolContext].
func (m *StrictContextMock) WithAgentCancel() (Context, context.CancelFunc) {
	panic("not implemented")
}

// OutputForAncestors implements [ToolContext].
func (m *StrictContextMock) OutputForAncestors() []string { panic("not implemented") }

var (
	_ ToolContext     = (*StrictContextMock)(nil)
	_ CallbackContext = (*StrictContextMock)(nil)
	_ ReadonlyContext = (*StrictContextMock)(nil)
)
