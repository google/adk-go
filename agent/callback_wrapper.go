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
	"fmt"
	"log"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
)

// callbackContextWrapper is used to emit log entries for unexpected calls - those
// related to ToolContext when Context is used as callback context
type callbackContextWrapper struct {
	context CallbackContext
}

// InvocationContext implements [Context].
func (c *callbackContextWrapper) InvocationContext() InvocationContext {
	log.Print("InvocationContext() is not supported for CallbackContext")
	return nil
}

// SetInvocationContext implements [Context].
func (c *callbackContextWrapper) SetInvocationContext(InvocationContext) {
	log.Print("SetInvocationContext() is not supported for CallbackContext")
}

// SubScheduler implements [Context].
func (c *callbackContextWrapper) SubScheduler() DynamicSubScheduler {
	log.Print("SubScheduler() is not supported for CallbackContext")
	return nil
}

// Agent implements [Context].
func (c *callbackContextWrapper) Agent() Agent {
	log.Print("Agent() is not supported for CallbackContext")
	return nil
}

// EndInvocation implements [Context].
func (c *callbackContextWrapper) EndInvocation() {
	log.Print("EndInvocation() is not supported for CallbackContext")
}

// Ended implements [Context].
func (c *callbackContextWrapper) Ended() bool {
	log.Print("Ended() is not supported for CallbackContext")
	return false
}

// IsolationScope implements [Context].
func (c *callbackContextWrapper) IsolationScope() string {
	log.Print("IsolationScope() is not supported for CallbackContext")
	return ""
}

// Memory implements [Context].
func (c *callbackContextWrapper) Memory() Memory {
	log.Print("Memory() is not supported for CallbackContext")
	return nil
}

// Path implements [Context].
func (c *callbackContextWrapper) Path() string {
	log.Print("Path() is not supported for CallbackContext")
	return ""
}

// ResumedInput implements [Context].
func (c *callbackContextWrapper) ResumedInput(interruptID string) (any, bool) {
	log.Print("ResumedInput() is not supported for CallbackContext")
	return nil, false
}

// RunConfig implements [Context].
func (c *callbackContextWrapper) RunConfig() *RunConfig {
	log.Print("RunConfig() is not supported for CallbackContext")
	return nil
}

// RunID implements [Context].
func (c *callbackContextWrapper) RunID() string {
	log.Print("RunID() is not supported for CallbackContext")
	return ""
}

// Session implements [Context].
func (c *callbackContextWrapper) Session() session.Session {
	log.Print("Session() is not supported for CallbackContext")
	return nil
}

// WithBranch implements [Context].
func (c *callbackContextWrapper) WithBranch(branch string) Context {
	log.Print("WithBranch() is not supported for CallbackContext")
	return nil
}

// WithContext implements [Context].
func (c *callbackContextWrapper) WithContext(ctx context.Context) InvocationContext {
	log.Print("WithContext() is not supported for CallbackContext")
	return nil
}

// ToolContext-related: emit logs and return empty data

// Actions implements [Context].
func (c *callbackContextWrapper) Actions() *session.EventActions {
	// return nil, Actions() do not make any sense for CallbackContext
	log.Print("Actions() is not supported for CallbackContext")
	return nil
}

// FunctionCallID implements [Context].
func (c *callbackContextWrapper) FunctionCallID() string {
	// return "", FunctionCallID() do not make any sense for CallbackContext
	log.Print("FunctionCallID() is not supported for CallbackContext")
	return ""
}

// RequestConfirmation implements [Context].
func (c *callbackContextWrapper) RequestConfirmation(hint string, payload any) error {
	//  RequestConfirmation() does not make any sense for CallbackContext
	log.Print("RequestConfirmation() is not supported for CallbackContext")
	return fmt.Errorf("RequestConfirmation() is not supported for callback context")
}

// SearchMemory implements [Context].
func (c *callbackContextWrapper) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	//  SearchMemory() does not make any sense for CallbackContext
	log.Print("SearchMemory() is not supported for CallbackContext")
	return nil, fmt.Errorf("SearchMemory() is not supported for callback context")
}

// ToolConfirmation implements [Context].
func (c *callbackContextWrapper) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	// ToolConfirmation() does not make any sense for CallbackContext
	log.Print("ToolConfirmation() is not supported for CallbackContext")
	return nil
}

// non ToolContext-related - call embedded context.

// AgentName implements [Context].
func (c *callbackContextWrapper) AgentName() string {
	return c.context.AgentName()
}

// AppName implements [Context].
func (c *callbackContextWrapper) AppName() string {
	return c.context.AppName()
}

// Artifacts implements [Context].
func (c *callbackContextWrapper) Artifacts() Artifacts {
	return c.context.Artifacts()
}

// Branch implements [Context].
func (c *callbackContextWrapper) Branch() string {
	return c.context.Branch()
}

// Deadline implements [Context].
func (c *callbackContextWrapper) Deadline() (deadline time.Time, ok bool) {
	return c.context.Deadline()
}

// Done implements [Context].
func (c *callbackContextWrapper) Done() <-chan struct{} {
	return c.context.Done()
}

// Err implements [Context].
func (c *callbackContextWrapper) Err() error {
	return c.context.Err()
}

// InvocationID implements [Context].
func (c *callbackContextWrapper) InvocationID() string {
	return c.context.InvocationID()
}

// ReadonlyState implements [Context].
func (c *callbackContextWrapper) ReadonlyState() session.ReadonlyState {
	return c.context.ReadonlyState()
}

// SessionID implements [Context].
func (c *callbackContextWrapper) SessionID() string {
	return c.context.SessionID()
}

// State implements [Context].
func (c *callbackContextWrapper) State() session.State {
	return c.context.State()
}

// UserContent implements [Context].
func (c *callbackContextWrapper) UserContent() *genai.Content {
	return c.context.UserContent()
}

// UserID implements [Context].
func (c *callbackContextWrapper) UserID() string {
	return c.context.UserID()
}

// Value implements [Context].
func (c *callbackContextWrapper) Value(key any) any {
	return c.context.Value(key)
}

var _ Context = (*callbackContextWrapper)(nil)
