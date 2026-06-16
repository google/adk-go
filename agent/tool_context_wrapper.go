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
	"log"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// toolContextWrapper is used to emit log entries for unexpected calls - those
// related to NodeContext when Context is used as tool context
type toolContextWrapper struct {
	context Context
}

// WithAgentCancel implements [Context].
func (c *toolContextWrapper) WithAgentCancel() (Context, context.CancelFunc) {
	// this one is needed for tool context.
	return c.context.WithAgentCancel()
}

// WithAgentTimeout implements [Context].
func (c *toolContextWrapper) WithAgentTimeout(timeout time.Duration) (Context, context.CancelFunc) {
	log.Print("WithAgentTimeout() is not supported for ToolContext")
	return nil, nil
}

// InvocationContext implements [Context].
func (c *toolContextWrapper) InvocationContext() InvocationContext {
	log.Print("InvocationContext() is not supported for ToolContext")
	return nil
}

// SubScheduler implements [Context].
func (c *toolContextWrapper) SubScheduler() DynamicSubScheduler {
	return c.context.SubScheduler()
}

// Agent implements [Context].
func (c *toolContextWrapper) Agent() Agent {
	log.Print("Agent() is not supported for ToolContext")
	return nil
}

// EndInvocation implements [Context].
func (c *toolContextWrapper) EndInvocation() {
	log.Print("EndInvocation() is not supported for ToolContext")
}

// Ended implements [Context].
func (c *toolContextWrapper) Ended() bool {
	log.Print("Ended() is not supported for ToolContext")
	return false
}

// IsolationScope implements [Context].
func (c *toolContextWrapper) IsolationScope() string {
	log.Print("IsolationScope() is not supported for ToolContext")
	return ""
}

// Memory implements [Context].
func (c *toolContextWrapper) Memory() Memory {
	log.Print("Memory() is not supported for ToolContext")
	return nil
}

// Path implements [Context].
func (c *toolContextWrapper) Path() string {
	log.Print("Path() is not supported for ToolContext")
	return ""
}

// ResumedInput implements [Context].
func (c *toolContextWrapper) ResumedInput(interruptID string) (any, bool) {
	log.Print("ResumedInput() is not supported for ToolContext")
	return nil, false
}

// RunConfig implements [Context].
func (c *toolContextWrapper) RunConfig() *RunConfig {
	log.Print("RunConfig() is not supported for ToolContext")
	return nil
}

// RunID implements [Context].
func (c *toolContextWrapper) RunID() string {
	log.Print("RunID() is not supported for ToolContext")
	return ""
}

// Session implements [Context].
func (c *toolContextWrapper) Session() session.Session {
	log.Print("Session() is not supported for ToolContext")
	return nil
}

// WithBranch implements [Context].
func (c *toolContextWrapper) WithBranch(branch string) Context {
	log.Print("WithBranch() is not supported for ToolContext")
	return nil
}

// WithContext implements [Context].
func (c *toolContextWrapper) WithContext(ctx context.Context) InvocationContext {
	log.Print("WithContext() is not supported for ToolContext")
	return nil
}

// WithAgentContext implements [Context].
func (c *toolContextWrapper) WithAgentContext(ctx context.Context) Context {
	log.Print("WithAgentContext() is not supported for ToolContext")
	return nil
}

// ToolContext-related: emit logs and return empty data

// Actions implements [Context].
func (c *toolContextWrapper) Actions() *session.EventActions {
	return c.context.Actions()
}

// FunctionCallID implements [Context].
func (c *toolContextWrapper) FunctionCallID() string {
	return c.context.FunctionCallID()
}

// RequestConfirmation implements [Context].
func (c *toolContextWrapper) RequestConfirmation(hint string, payload any) error {
	return c.context.RequestConfirmation(hint, payload)
}

// SearchMemory implements [Context].
func (c *toolContextWrapper) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	return c.context.SearchMemory(ctx, query)
}

// ToolConfirmation implements [Context].
func (c *toolContextWrapper) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	return c.context.ToolConfirmation()
}

// NodeContext

func (c *toolContextWrapper) OutputForAncestors() []string {
	log.Print("OutputForAncestors() is not supported for ToolContext")
	return nil
}

// AgentName implements [Context].
func (c *toolContextWrapper) AgentName() string {
	return c.context.AgentName()
}

// AppName implements [Context].
func (c *toolContextWrapper) AppName() string {
	return c.context.AppName()
}

// Artifacts implements [Context].
func (c *toolContextWrapper) Artifacts() Artifacts {
	return c.context.Artifacts()
}

// Branch implements [Context].
func (c *toolContextWrapper) Branch() string {
	return c.context.Branch()
}

// Deadline implements [Context].
func (c *toolContextWrapper) Deadline() (deadline time.Time, ok bool) {
	return c.context.Deadline()
}

// Done implements [Context].
func (c *toolContextWrapper) Done() <-chan struct{} {
	return c.context.Done()
}

// Err implements [Context].
func (c *toolContextWrapper) Err() error {
	return c.context.Err()
}

// InvocationID implements [Context].
func (c *toolContextWrapper) InvocationID() string {
	return c.context.InvocationID()
}

// ReadonlyState implements [Context].
func (c *toolContextWrapper) ReadonlyState() session.ReadonlyState {
	return c.context.ReadonlyState()
}

// SessionID implements [Context].
func (c *toolContextWrapper) SessionID() string {
	return c.context.SessionID()
}

// State implements [Context].
func (c *toolContextWrapper) State() session.State {
	return c.context.State()
}

// UserContent implements [Context].
func (c *toolContextWrapper) UserContent() *genai.Content {
	return c.context.UserContent()
}

// UserID implements [Context].
func (c *toolContextWrapper) UserID() string {
	return c.context.UserID()
}

// Value implements [Context].
func (c *toolContextWrapper) Value(key any) any {
	return c.context.Value(key)
}

var _ Context = (*toolContextWrapper)(nil)
