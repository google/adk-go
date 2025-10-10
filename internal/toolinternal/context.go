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

package toolinternal

import (
	"context"

	"github.com/google/uuid"
	"google.golang.org/adk/agent"
	contextinternal "google.golang.org/adk/internal/context"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

func NewToolContext(ctx agent.InvocationContext, functionCallID string, actions *session.EventActions) tool.Context {
	if functionCallID == "" {
		functionCallID = uuid.NewString()
	}
	cbCtx := contextinternal.NewCallbackContext(ctx)
	return &toolContext{
		CallbackContext:   cbCtx,
		invocationContext: ctx,
		functionCallID:    functionCallID,
		eventActions:      actions,
	}
}

type toolContext struct {
	agent.CallbackContext
	invocationContext agent.InvocationContext
	functionCallID    string
	eventActions      *session.EventActions
}

func (c *toolContext) FunctionCallID() string {
	return c.functionCallID
}

func (c *toolContext) Actions() *session.EventActions {
	return c.eventActions
}

func (c *toolContext) AgentName() string {
	return c.invocationContext.Agent().Name()
}

func (c *toolContext) SearchMemory(ctx context.Context, query string) ([]memory.Entry, error) {
	return c.invocationContext.Memory().Search(query)
}
