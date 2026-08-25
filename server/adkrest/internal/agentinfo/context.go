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

package agentinfo

import (
	"context"
	"iter"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// staticContext is a minimal [agent.ReadonlyContext] for resolving toolsets
// outside of an invocation. Describing an app runs no agent, so there is no
// user content, no session and no invocation to expose; a toolset that consults
// any of them sees empty values rather than a nil dereference.
type staticContext struct {
	context.Context

	appName   string
	agentName string
}

func newStaticContext(ctx context.Context, appName, agentName string) agent.ReadonlyContext {
	return staticContext{Context: ctx, appName: appName, agentName: agentName}
}

func (c staticContext) AppName() string   { return c.appName }
func (c staticContext) AgentName() string { return c.agentName }

func (staticContext) UserContent() *genai.Content { return nil }
func (staticContext) InvocationID() string        { return "" }
func (staticContext) UserID() string              { return "" }
func (staticContext) SessionID() string           { return "" }
func (staticContext) Branch() string              { return "" }

func (staticContext) ReadonlyState() session.ReadonlyState { return emptyState{} }

// emptyState is a [session.ReadonlyState] that holds nothing.
type emptyState struct{}

func (emptyState) Get(string) (any, error) { return nil, session.ErrStateKeyNotExist }

func (emptyState) All() iter.Seq2[string, any] {
	return func(func(string, any) bool) {}
}
