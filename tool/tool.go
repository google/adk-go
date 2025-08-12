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
	"google.golang.org/adk/llm"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type Tool interface {
	Name() string
	Description() string
	Declaration() *genai.FunctionDeclaration
	Run(ctx Context, args any) (result any, err error)

	// TODO: remove
	ProcessRequest(ctx Context, req *llm.Request) error
}

type Context interface {
	agent.Context
	FunctionCallID() string

	// TODO: remove
	EventActions() *session.Actions
}

// TODO: implement
type Set struct{}

func NewSet(t ...Tool) Set { return Set{} }

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
