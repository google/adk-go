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

// Package context provides helper functions for constructing ADK contexts.
package context

import (
	"context"

	"google.golang.org/adk/agent"
	agentinternal "google.golang.org/adk/internal/agent"
	"google.golang.org/adk/session"
)

// NewCallbackContext returns a new agent.CallbackContext for callbacks
// run in the context of the agent invocation.
func NewCallbackContext(ctx agent.InvocationContext) agent.CallbackContext {
	return agentinternal.NewCallbackContext(contextAdapter{ctx}, &session.EventActions{})
}

// contextAdapter is an adapter that converts [agent.InvocationContext]
// to [agentinternal.ContextBase].
type contextAdapter struct {
	agent.InvocationContext
}

func (a contextAdapter) AgentName() string {
	return a.Agent().Name()
}

func (a contextAdapter) Context() context.Context {
	return a.InvocationContext
}
