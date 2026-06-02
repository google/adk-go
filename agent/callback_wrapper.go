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
	"time"

	"log"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/genai"
)

type callbackContextWrapper struct {
	context CallbackContext
}

// Actions implements [Context].
func (c *callbackContextWrapper) Actions() *session.EventActions {
	// return nil, Actions() do not make any sense for CallbackContext
	log.Print("Actions() is not supported for CallbackContext")
	return nil
}

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

// FunctionCallID implements [Context].
func (c *callbackContextWrapper) FunctionCallID() string {
	// return "", FunctionCallID() do not make any sense for CallbackContext
	log.Print("FunctionCallID() is not supported for CallbackContext")
	return ""
}

// InvocationID implements [Context].
func (c *callbackContextWrapper) InvocationID() string {
	return c.context.InvocationID()
}

// ReadonlyState implements [Context].
func (c *callbackContextWrapper) ReadonlyState() session.ReadonlyState {
	return c.context.ReadonlyState()
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

// SessionID implements [Context].
func (c *callbackContextWrapper) SessionID() string {
	return c.context.SessionID()
}

// State implements [Context].
func (c *callbackContextWrapper) State() session.State {
	return c.context.State()
}

// ToolConfirmation implements [Context].
func (c *callbackContextWrapper) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	// ToolConfirmation() does not make any sense for CallbackContext
	log.Print("ToolConfirmation() is not supported for CallbackContext")
	return nil
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
