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

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// ContextMock defines mocking logic (makes creating your own mocks easier if embedded)
type ContextMock struct{}

// WithAgentCancel implements [Context].
func (c *ContextMock) WithAgentCancel() (Context, context.CancelFunc) {
	return nil, nil
}

// WithAgentTimeout implements [Context].
func (c *ContextMock) WithAgentTimeout(timeout time.Duration) (Context, context.CancelFunc) {
	return nil, nil
}

// Actions implements [Context].
func (c *ContextMock) Actions() *session.EventActions {
	return nil
}

// Agent implements [Context].
func (c *ContextMock) Agent() Agent {
	return nil
}

// AgentName implements [Context].
func (c *ContextMock) AgentName() string {
	return ""
}

// AppName implements [Context].
func (c *ContextMock) AppName() string {
	return ""
}

// Artifacts implements [Context].
func (c *ContextMock) Artifacts() Artifacts {
	return nil
}

// Branch implements [Context].
func (c *ContextMock) Branch() string {
	return ""
}

// Deadline implements [Context].
func (c *ContextMock) Deadline() (deadline time.Time, ok bool) {
	panic("unimplemented")
}

// Done implements [Context].
func (c *ContextMock) Done() <-chan struct{} {
	panic("unimplemented")
}

// EndInvocation implements [Context].
func (c *ContextMock) EndInvocation() {
}

// Ended implements [Context].
func (c *ContextMock) Ended() bool {
	return false
}

// Err implements [Context].
func (c *ContextMock) Err() error {
	return nil
}

// FunctionCallID implements [Context].
func (c *ContextMock) FunctionCallID() string {
	return ""
}

// InvocationContext implements [Context].
func (c *ContextMock) InvocationContext() InvocationContext {
	return nil
}

// InvocationID implements [Context].
func (c *ContextMock) InvocationID() string {
	return ""
}

// IsolationScope implements [Context].
func (c *ContextMock) IsolationScope() string {
	return ""
}

// Memory implements [Context].
func (c *ContextMock) Memory() Memory {
	return nil
}

// Path implements [Context].
func (c *ContextMock) Path() string {
	return ""
}

// ReadonlyState implements [Context].
func (c *ContextMock) ReadonlyState() session.ReadonlyState {
	return nil
}

// RequestConfirmation implements [Context].
func (c *ContextMock) RequestConfirmation(hint string, payload any) error {
	return nil
}

// ResumedInput implements [Context].
func (c *ContextMock) ResumedInput(interruptID string) (any, bool) {
	return nil, false
}

// RunConfig implements [Context].
func (c *ContextMock) RunConfig() *RunConfig {
	return nil
}

// RunID implements [Context].
func (c *ContextMock) RunID() string {
	return ""
}

// SearchMemory implements [Context].
func (c *ContextMock) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	return nil, nil
}

// Session implements [Context].
func (c *ContextMock) Session() session.Session {
	return nil
}

// SessionID implements [Context].
func (c *ContextMock) SessionID() string {
	return ""
}

// SetInvocationContext implements [Context].
func (c *ContextMock) SetInvocationContext(InvocationContext) {
}

// State implements [Context].
func (c *ContextMock) State() session.State {
	return nil
}

// SubScheduler implements [Context].
func (c *ContextMock) SubScheduler() DynamicSubScheduler {
	return nil
}

// ToolConfirmation implements [Context].
func (c *ContextMock) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	return nil
}

// UserContent implements [Context].
func (c *ContextMock) UserContent() *genai.Content {
	return nil
}

// UserID implements [Context].
func (c *ContextMock) UserID() string {
	return ""
}

// Value implements [Context].
func (c *ContextMock) Value(key any) any {
	return nil
}

// WithBranch implements [Context].
func (c *ContextMock) WithBranch(branch string) Context {
	return nil
}

// WithContext implements [Context].
func (c *ContextMock) WithContext(ctx context.Context) InvocationContext {
	return nil
}

// WithContext implements [Context].
func (c *ContextMock) WithAgentContext(ctx context.Context) Context {
	return nil
}

func (c *ContextMock) OutputForAncestors() []string {
	return nil
}

func (c *ContextMock) WithDelta(d *CommonContextDelta) Context {
	return c
}

func (c *ContextMock) WithICDelta(d *InvocationContextDelta) InvocationContext {
	return c
}

var (
	_ Context           = (*ContextMock)(nil)
	_ InvocationContext = (*ContextMock)(nil)
)
