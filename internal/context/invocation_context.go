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

package context

import (
	"context"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

type InvocationContextParams struct {
	Artifacts agent.Artifacts
	Memory    agent.Memory
	Session   session.Session

	Branch string
	Agent  agent.Agent

	UserContent   *genai.Content
	RunConfig     *agent.RunConfig
	EndInvocation bool
	Values        map[string]any
}

func NewInvocationContext(ctx context.Context, params InvocationContextParams) agent.InvocationContext {
	var values map[string]any
	if len(params.Values) > 0 {
		values = make(map[string]any, len(params.Values))
		for k, v := range params.Values {
			values[k] = v
		}
	}
	return &InvocationContext{
		Context:      ctx,
		params:       params,
		invocationID: "e-" + uuid.NewString(),
		values:       values,
	}
}

type InvocationContext struct {
	context.Context

	params       InvocationContextParams
	invocationID string
	values       map[string]any
}

func (c *InvocationContext) Artifacts() agent.Artifacts {
	return c.params.Artifacts
}

// Value returns the value associated with key in the invocation context.
// Custom values provided via InvocationContextParams.Values take precedence;
// otherwise fall back to the underlying context.
func (c *InvocationContext) Value(key any) any {
	if k, ok := key.(string); ok {
		if v, found := c.values[k]; found {
			return v
		}
	}
	return c.Context.Value(key)
}

// SetInvocationValue stores a custom value scoped to this invocation.
func (c *InvocationContext) SetInvocationValue(key string, value any) {
	c.setValue(key, value)
}

func (c *InvocationContext) setValue(key string, value any) {
	if value == nil {
		delete(c.values, key)
		return
	}
	if c.values == nil {
		c.values = make(map[string]any)
	}
	c.values[key] = value
}

func (c *InvocationContext) Agent() agent.Agent {
	return c.params.Agent
}

func (c *InvocationContext) Branch() string {
	return c.params.Branch
}

func (c *InvocationContext) InvocationID() string {
	return c.invocationID
}

func (c *InvocationContext) Memory() agent.Memory {
	return c.params.Memory
}

func (c *InvocationContext) Session() session.Session {
	return c.params.Session
}

func (c *InvocationContext) UserContent() *genai.Content {
	return c.params.UserContent
}

func (c *InvocationContext) RunConfig() *agent.RunConfig {
	return c.params.RunConfig
}

func (c *InvocationContext) EndInvocation() {
	c.params.EndInvocation = true
}

func (c *InvocationContext) Ended() bool {
	return c.params.EndInvocation
}
