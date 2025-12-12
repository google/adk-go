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

package adka2a

import (
	"context"
	"iter"

	"github.com/a2aproject/a2a-go/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// ExecutorContext provides read-only information about the context of an A2A agent execution.
// An execution starts with a user message and ends with a task in a terminal or input-required state.
type ExecutorContext interface {
	context.Context

	// SessionID is ID of the session. It is passed as contextID in A2A request.
	SessionID() string
	// UserID is ID of the user who made the request. The information is either extracted from [a2asrv.CallContext]
	// or derived from session ID for unauthenticated requests.
	UserID() string
	// AgentName is the name of the root agent.
	AgentName() string
	// ReadonlyState provides a view of the current session state.
	ReadonlyState() session.ReadonlyState
	// UserContent is a converted A2A message which is passed to runner.Run.
	UserContent() *genai.Content
	// RequestContext containts information about the original A2A Request, the current task and related tasks.
	RequestContext() *a2asrv.RequestContext
}

type executorContext struct {
	context.Context
	meta        invocationMeta
	session     session.ReadonlyState
	userContent *genai.Content
}

func newExecutorContext(ctx context.Context, meta invocationMeta, session session.ReadonlyState, userContent *genai.Content) ExecutorContext {
	return &executorContext{
		Context:     ctx,
		meta:        meta,
		session:     session,
		userContent: userContent,
	}
}

func (ec *executorContext) SessionID() string {
	return ec.meta.sessionID
}

func (ec *executorContext) UserID() string {
	return ec.meta.userID
}

func (ec *executorContext) AgentName() string {
	return ec.meta.agentName
}

func (ec *executorContext) ReadonlyState() session.ReadonlyState {
	return ec.session
}

func (ec *executorContext) RequestContext() *a2asrv.RequestContext {
	return ec.meta.reqCtx
}

func (ec *executorContext) UserContent() *genai.Content {
	return ec.userContent
}

type emptySessionState struct{}

func (emptySessionState) Get(string) (any, error) {
	return nil, session.ErrStateKeyNotExist
}

func (emptySessionState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {}
}
