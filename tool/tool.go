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

package tool

import (
	"github.com/google/uuid"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// Tool defines the interface for a callable tool.
type Tool interface {
	// Name returns the name of the tool.
	Name() string
	// Description returns a description of the tool.
	Description() string
	// IsLongRunning indicates whether the tool is a long-running operation,
	// which typically returns a resource id first and finishes the operation later.
	IsLongRunning() bool
}

type Context interface {
	agent.Context
	FunctionCallID() string

	// TODO: remove
	EventActions() *session.Actions
}

type Set interface {
	Tool // to allow passing a toolset to agent tools
	Tools(ctx agent.Context) ([]Tool, error)
}

func NewContext(ctx agent.Context, functionCallID string, actions *session.Actions) Context {
	if functionCallID == "" {
		functionCallID = uuid.NewString()
	}
	return &toolContext{
		Context:        ctx,
		functionCallID: functionCallID,
		eventActions:   actions,
	}
}

type toolContext struct {
	agent.Context
	functionCallID string
	eventActions   *session.Actions
}

func (c *toolContext) FunctionCallID() string {
	return c.functionCallID
}

func (c *toolContext) EventActions() *session.Actions {
	return c.eventActions
}
